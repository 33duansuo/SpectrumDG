"""SpectrumDG 训练入口: 两阶段 (Stage1 firstU → Stage2 secondU) 训练 + 自动评估。

验证集从训练场景 85/15 划分; 训练结束后做 in-domain(val) + cross-domain(未见场景)
评估, 报告写入 {save_dir}/dg_report.json, 训练配置快照写入 {save_dir}/config.json
(供 test.py 复现评估)。

用法:
  cd /data/userSZ/duanrj/SpectrumDG
  conda activate wifi-diff
  # 基线
  CUDA_VISIBLE_DEVICES=5 python train.py --train_terrains 05.Suburban,06.DenseUrban,08.OrdinaryUrban --test_terrain 01.Grassland --save_dir run/baseline_3urban
  # CIRL 因果分解 (瓶颈层)
  CUDA_VISIBLE_DEVICES=5 python train.py --exp causal --save_dir run/causal_3urban
  # Fish 梯度对齐 (损失函数层)
  CUDA_VISIBLE_DEVICES=5 python train.py --exp fish --save_dir run/fish_3urban
  # Fish+CIRL / Fish+CIRL+SWAD: --exp fish_causal / --exp fish_causal_swad
"""
import argparse
import json
import os
from dataclasses import asdict, fields

import numpy as np
import torch

from configs import EXPERIMENTS, TrainConfig
from data_process.dataset import build_data
from lib.dg import build_domain_map
from lib.evaluate import evaluate_report
from lib.trainer import Trainer
from model.radiownet import RadioWNet


def parse_args():
    p = argparse.ArgumentParser(description="SpectrumDG: SpectrumNet 跨域泛化训练+评估")
    # 实验预设 (与命令行 DG 开关等价, 二选一; 开关优先)
    p.add_argument("--exp", type=str, default="baseline",
                   choices=sorted(EXPERIMENTS.keys()),
                   help="实验预设: baseline/causal/fish/fish_causal/fish_causal_swad")
    p.add_argument("--gpu", type=int, default=None, help="可见 GPU 内的编号 (建议用 CUDA_VISIBLE_DEVICES 代替)")
    # 数据
    p.add_argument("--train_terrains", type=str, default=None, help="训练场景, 逗号分隔")
    p.add_argument("--test_terrain", "--test_terrains", dest="test_terrains", type=str, default=None,
                   help="跨域测试场景, 逗号分隔可多场景")
    p.add_argument("--z_layer", type=str, default=None, help="z 层: 00/01/02")
    p.add_argument("--freqs", type=str, default=None, help="freq 白名单, 逗号分隔; 空=全部")
    p.add_argument("--data_root", type=str, default=None, help="PPData5D-success 根目录")
    p.add_argument("--max_per_terrain", type=int, default=None,
                   help="每场景样本上限(快速验证); None=全部")
    p.add_argument("--num_workers", type=int, default=None)
    # 优化
    p.add_argument("--num_epochs", type=int, default=None, help="每阶段最大 epoch 数 (早停可能提前结束)")
    p.add_argument("--patience", type=int, default=None, help="验证集 patience 轮无改善则早停")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    # Fish 梯度对齐损失 (损失函数层, 参考 Fishr)
    p.add_argument("--use_fish", action="store_true", default=None,
                   help="启用 Fish: 在ERM损失上增加梯度对齐惩罚, 约束跨域梯度方向一致")
    p.add_argument("--fish_lambda", type=float, default=None,
                   help="梯度对齐惩罚系数 (参考 Fishr, 典型 0.01~1.0)")
    # CIRL 因果分解 (瓶颈层, 参考 CIRL)
    p.add_argument("--use_causal", action="store_true", default=None,
                   help="启用 CIRL 因果分解: bottleneck 分解 z_inv(域不变→解码器)/z_spe(域特定→域分类), "
                        "GRL 对抗 + 正交解相关")
    p.add_argument("--causal_lambda_dom", type=float, default=None, help="z_spe 域分类 CE 权重 (应可分)")
    p.add_argument("--causal_lambda_inv", type=float, default=None,
                   help="z_inv 对抗域分类 CE 权重 (经 GRL, DANN 风格, 典型 0.01~1.0)")
    p.add_argument("--causal_lambda_orth", type=float, default=None, help="z_inv/z_spe 正交解相关权重")
    p.add_argument("--causal_lambda_grl", type=float, default=None, help="GRL 梯度反转系数")
    # SWAD 权重平均 (训练策略层, 参考 SWAD, Cha et al. 2021)
    p.add_argument("--use_swad", action="store_true", default=None,
                   help="启用 SWAD: 训练后期对权重做等权平均, 寻找平坦极小值")
    p.add_argument("--swad_start_ratio", type=float, default=None,
                   help="开始权重平均的 epoch 占比 (默认 0.25)")
    # 两阶段 DG 分工 + 初始化复用
    p.add_argument("--dg_second", "--no_dg_second", dest="dg_second", action="store_false",
                   default=None,
                   help="Stage2 (secondU) 纯 ERM 域内精修, 不挂 Causal/Fish/SWAD (dg_gate 实验)")
    p.add_argument("--init_firstU", type=str, default=None,
                   help="加载已有 firstU 权重 (如 run/xxx/Trained_Model_FirstU.pt) 并跳过 Stage1; "
                        "Stage1/Stage2 梯度完全隔离, 数学上等价于重跑")
    # 断点续训
    p.add_argument("--resume", type=str, default=None,
                   help="从 checkpoint 恢复训练; 传 'auto' 自动在 save_dir 下查找 checkpoint_firstU.pt / checkpoint_secondU.pt")
    p.add_argument("--checkpoint_interval", type=int, default=10,
                   help="每隔多少 epoch 自动保存 checkpoint (默认 10, 0=不保存)")
    # 输出
    p.add_argument("--save_dir", type=str, default=None, help="模型 + 报告保存目录")
    p.add_argument("--report", type=str, default=None, help="DG 报告 JSON 路径")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def build_config(args):
    """实验预设 + 命令行覆盖 (仅显式传入的参数覆盖预设)。"""
    overrides = {}
    for f in fields(TrainConfig):
        val = getattr(args, f.name, None)
        if val is not None:
            overrides[f.name] = val
    return TrainConfig(**{**EXPERIMENTS[args.exp], **overrides})


def build_model(cfg, num_domains, device):
    return RadioWNet(inputs=2, phase="firstU",
                     use_causal=cfg.use_causal,
                     num_domains=num_domains if cfg.use_causal else 0,
                     causal_lambda_grl=cfg.causal_lambda_grl,
                     use_causal_2nd=cfg.use_causal and cfg.dg_second).to(device)


def main():
    args = parse_args()
    cfg = build_config(args)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | exp={args.exp} train={cfg.train_terrains} test={cfg.test_terrains} "
          f"fish={cfg.use_fish} causal={cfg.use_causal} swad={cfg.use_swad}")

    # 配置快照: test.py 依据它复现评估
    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_terrs, test_terrs, train_dl, val_dl, test_dls = build_data(cfg)

    domain_map = build_domain_map(train_terrs) if cfg.use_causal else {}
    if cfg.use_causal:
        print(f"domain_map (global→local): {domain_map}")

    model = build_model(cfg, len(train_terrs), device)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    trainer = Trainer(model, {'train': train_dl, 'val': val_dl}, device, cfg, domain_map)

    ckpt_interval = args.checkpoint_interval

    # ===== Stage 1: firstU 训练 =====
    print("\n===== Stage 1: firstU 训练 =====")
    resume_1st = None
    if args.resume:
        auto_path = os.path.join(cfg.save_dir, "checkpoint_firstU.pt")
        resume_1st = auto_path if args.resume == "auto" else args.resume
        if not os.path.isfile(resume_1st):
            print(f"[Resume] checkpoint 不存在: {resume_1st}, 从头开始训练")
            resume_1st = None
    if args.init_firstU and resume_1st is None:
        # 复用已训练的 firstU 权重, 跳过 Stage1 (两阶段梯度隔离, 等价于重跑)
        state = torch.load(args.init_firstU, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        unexpected = [k for k in unexpected if k.startswith("causal_2nd")]
        print(f"[Init] firstU 权重加载自 {args.init_firstU} "
              f"(missing={len(missing)}, ignored causal_2nd={len(unexpected)}), 跳过 Stage1")
    else:
        model = trainer.train_stage("firstU", resume_from=resume_1st,
                                    checkpoint_interval=ckpt_interval)
    ckpt1 = os.path.join(cfg.save_dir, "Trained_Model_FirstU.pt")
    torch.save(model.state_dict(), ckpt1)
    print(f"saved: {ckpt1}")

    # ===== Stage 2: secondU 训练 =====
    print("\n===== Stage 2: secondU 训练 =====")
    model.phase = "secondU"  # firstU 路径 detached, 只训练 W-Net (两阶段设计)
    resume_2nd = None
    if args.resume:
        auto_path = os.path.join(cfg.save_dir, "checkpoint_secondU.pt")
        resume_2nd = auto_path if args.resume == "auto" else None
        if resume_2nd and not os.path.isfile(resume_2nd):
            print(f"[Resume] checkpoint 不存在: {resume_2nd}, 从头开始 Stage 2")
            resume_2nd = None
    model = trainer.train_stage("secondU", resume_from=resume_2nd,
                                checkpoint_interval=ckpt_interval)
    ckpt2 = os.path.join(cfg.save_dir, "Trained_Model_SecondU.pt")
    torch.save(model.state_dict(), ckpt2)
    print(f"saved: {ckpt2}")

    # ===== 评估: in-domain + cross-domain (多场景) =====
    print("\n===== Test: 分域评估 (firstU + secondU + gated) =====")
    report = evaluate_report(model, val_dl, test_dls, test_terrs, device)

    rep_path = cfg.report or os.path.join(cfg.save_dir, "dg_report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDG 报告保存至 {rep_path}")
    print(f"\n[结果] in_domain NMSE_firstU = {report['indomain']['NMSE_firstU']:.6f} "
          f"NMSE_secondU = {report['indomain']['NMSE_secondU']:.6f}")
    print(f"[结果] cross_avg NMSE_firstU = {report['cross_avg']['NMSE_firstU']:.6f} "
          f"NMSE_secondU = {report['cross_avg']['NMSE_secondU']:.6f}")
    if 'NMSE_gated' in report['cross_avg']:
        cross_fracs = [report['cross_' + t].get('gate_second_frac', 0) for t in test_terrs]
        frac_str = '/'.join(f'{v:.3f}' for v in cross_fracs)
        print(f"[结果] in_domain NMSE_gated = {report['indomain']['NMSE_gated']:.6f} "
              f"cross_avg NMSE_gated = {report['cross_avg']['NMSE_gated']:.6f} "
              f"(gate_second_frac: {report['indomain'].get('gate_second_frac', 0):.3f} "
              f"in-domain / {frac_str} cross)")


if __name__ == "__main__":
    main()
