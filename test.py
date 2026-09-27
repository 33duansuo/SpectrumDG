"""SpectrumDG 评估入口: 从训练产物 (config.json + checkpoint) 复现评估。

默认读取 {run_dir}/config.json (train.py 训练时写入) 重建数据与模型,
加载 {run_dir}/Trained_Model_SecondU.pt (含两阶段全部权重), 输出
in-domain + cross-domain 的 MAE/MSE/PSNR 报告。

用法:
  cd /data/userSZ/duanrj/SpectrumDG
  conda activate wifi-diff
  CUDA_VISIBLE_DEVICES=4 python test.py --run_dir run/dual_expert_4to7
  # 覆盖测试场景
  python test.py --run_dir run/dual_expert_4to7 --test_terrain 01.Grassland,02.Island
  # 无 config.json 的旧实验目录需用 --exp 指明预设 (仅 baseline/dg_gate/dual_expert)
  python test.py --run_dir run/xxx --exp dg_gate
"""
import argparse
import json
import os
from dataclasses import fields

import torch
import numpy as np

from configs import EXPERIMENTS, TrainConfig
from data_process.dataset import SpectrumNetDataset, TERRAIN_ORDER
from torch.utils.data import DataLoader
from lib.evaluate import evaluate_report
from model.radiownet import RadioWNet
from model.dual_expert import DualExpertNet


def parse_args():
    p = argparse.ArgumentParser(description="SpectrumDG: 从 checkpoint 复现 DG 评估")
    p.add_argument("--run_dir", type=str, required=True,
                   help="训练输出目录 (含 config.json 与 Trained_Model_SecondU.pt)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="模型权重路径; 默认 {run_dir}/Trained_Model_SecondU.pt")
    p.add_argument("--exp", type=str, default=None, choices=sorted(EXPERIMENTS.keys()),
                   help="旧实验目录无 config.json 时, 指明训练时的实验预设")
    p.add_argument("--gpu", type=int, default=None, help="可见 GPU 内的编号 (建议用 CUDA_VISIBLE_DEVICES 代替)")
    # 可覆盖项 (其余数据配置从 config.json 继承)
    p.add_argument("--test_terrain", type=str, default=None, help="覆盖跨域测试场景, 逗号分隔可多场景")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_per_terrain", type=int, default=None)
    p.add_argument("--report", type=str, default=None, help="DG 报告 JSON 路径; 默认 {run_dir}/dg_report.json")
    # Router 门控 (不给则默认多物理先验 v4 规则)
    p.add_argument("--router", type=str, default=None,
                   help="RouterBin.pt 路径 (train_router_bin.py 产物); 给出则用学习式门控替代 v4 规则")
    p.add_argument("--tau", type=str, default="auto",
                   help="Router 门控阈值; 'auto' = val 集 95%% 召回自动校准, 或浮点数")
    return p.parse_args()


def load_config(args):
    """run_dir/config.json + 命令行覆盖; 旧目录回退到 --exp 预设。"""
    overrides = {}
    for f in fields(TrainConfig):
        val = getattr(args, f.name, None)
        if val is not None:
            overrides[f.name] = val

    cfg_path = os.path.join(args.run_dir, "config.json")
    valid_fields = {f.name for f in fields(TrainConfig)}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
        # 过滤旧版本 config.json 中已删除的字段 (如 use_bridge), 保证跨版本可复现
        saved = {k: v for k, v in saved.items() if k in valid_fields}
        cfg = TrainConfig(**{**saved, **overrides})
    elif args.exp is not None:
        cfg = TrainConfig(**{**EXPERIMENTS[args.exp],
                             "save_dir": args.run_dir, **overrides})
    else:
        raise FileNotFoundError(
            f"{cfg_path} 不存在; 旧实验目录请用 --exp 指明训练时的实验预设")
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | run_dir={args.run_dir} train={cfg.train_terrains} "
          f"test={cfg.test_terrains} causal={cfg.use_causal}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- 数据: val (in-domain) + 每个测试场景独立 loader (返回 terrain 以支持多物理先验门控) ----
    freqs = [s.strip() for s in cfg.freqs.split(",") if s.strip()] or None
    train_terrs = [s.strip() for s in cfg.train_terrains.split(",") if s.strip()]
    test_terrs = [s.strip() for s in cfg.test_terrains.split(",") if s.strip()]
    common = dict(z_layer=cfg.z_layer, freqs=freqs, max_per_terrain=cfg.max_per_terrain,
                  root=cfg.data_root, seed=cfg.seed, return_terrain=True)
    val_ds = SpectrumNetDataset(train_terrs, split="val", split_ratio=0.85, **common)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)
    test_dls = {}
    for tt in test_terrs:
        test_ds = SpectrumNetDataset([tt], split="test", split_ratio=1.0, **common)
        test_dls[tt] = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers)
    print(f"val={len(val_dl.dataset)} "
          f"test={{{', '.join(f'{t}={len(test_dls[t].dataset)}' for t in test_terrs)}}}")

    # ---- 模型: 与训练时同构, 加载权重 ----
    if cfg.arch == "dual":
        model = DualExpertNet(inputs=2, num_domains=len(train_terrs),
                              causal_lambda_grl=cfg.causal_lambda_grl).to(device)
        default_ckpt = "Trained_Model_DualExpert.pt"
    else:
        model = RadioWNet(inputs=2, phase="secondU",
                          use_causal=cfg.use_causal,
                          num_domains=len(train_terrs) if cfg.use_causal else 0,
                          causal_lambda_grl=cfg.causal_lambda_grl,
                          use_causal_2nd=cfg.use_causal and cfg.dg_second).to(device)
        default_ckpt = "Trained_Model_SecondU.pt"
    ckpt_path = args.ckpt or os.path.join(args.run_dir, default_ckpt)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"loaded: {ckpt_path}")

    # ---- Router 门控 (可选; 默认 v4 规则) ----
    router, tau = None, None
    if args.router is not None:
        from model.router import BinaryRouter
        rckpt = torch.load(args.router, map_location=device)
        router = BinaryRouter(width=rckpt.get("width", 32)).to(device)
        router.load_state_dict(rckpt["state_dict"])
        router.eval()
        tau = None if args.tau == "auto" else float(args.tau)
        print(f"loaded router: {args.router} "
              f"(protocol={rckpt.get('protocol', '?')}, tau={'auto' if tau is None else tau})")

    # ---- 评估 ----
    print("\n===== Test: 分域评估 (firstU + secondU + gated) =====")
    report = evaluate_report(model, val_dl, test_dls, test_terrs, device,
                             router=router, tau=tau)
    report["arch"] = cfg.arch

    rep_path = args.report or os.path.join(args.run_dir, "dg_report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDG 报告保存至 {rep_path}")
    print(f"\n[结果] in_domain MAE_firstU = {report['indomain']['MAE_firstU']:.6f} "
          f"MSE_firstU = {report['indomain']['MSE_firstU']:.6f} PSNR_firstU = {report['indomain']['PSNR_firstU']:.4f}dB")
    print(f"[结果] in_domain MAE_secondU = {report['indomain']['MAE_secondU']:.6f} "
          f"MSE_secondU = {report['indomain']['MSE_secondU']:.6f} PSNR_secondU = {report['indomain']['PSNR_secondU']:.4f}dB")
    print(f"[结果] cross_avg MAE_firstU = {report['cross_avg']['MAE_firstU']:.6f} "
          f"MSE_firstU = {report['cross_avg']['MSE_firstU']:.6f} PSNR_firstU = {report['cross_avg']['PSNR_firstU']:.4f}dB")
    print(f"[结果] cross_avg MAE_secondU = {report['cross_avg']['MAE_secondU']:.6f} "
          f"MSE_secondU = {report['cross_avg']['MSE_secondU']:.6f} PSNR_secondU = {report['cross_avg']['PSNR_secondU']:.4f}dB")
    if 'MAE_gated' in report['cross_avg']:
        cross_fracs = [report['cross_' + t].get('gate_second_frac', 0) for t in test_terrs]
        frac_str = '/'.join(f'{v:.3f}' for v in cross_fracs)
        print(f"[结果] in_domain MAE_gated = {report['indomain']['MAE_gated']:.6f} "
              f"MSE_gated = {report['indomain']['MSE_gated']:.6f} PSNR_gated = {report['indomain']['PSNR_gated']:.4f}dB")
        print(f"[结果] cross_avg MAE_gated = {report['cross_avg']['MAE_gated']:.6f} "
              f"MSE_gated = {report['cross_avg']['MSE_gated']:.6f} PSNR_gated = {report['cross_avg']['PSNR_gated']:.4f}dB "
              f"(gate_second_frac: {report['indomain'].get('gate_second_frac', 0):.3f} "
              f"in-domain / {frac_str} cross)")


if __name__ == "__main__":
    main()
