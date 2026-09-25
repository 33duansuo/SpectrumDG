"""SpectrumDG 评估入口: 从训练产物 (config.json + checkpoint) 复现评估。

默认读取 {run_dir}/config.json (train.py 训练时写入) 重建数据与模型,
加载 {run_dir}/Trained_Model_SecondU.pt (含两阶段全部权重), 输出
in-domain + cross-domain 的 MSE/NMSE 报告。

用法:
  cd /data/userSZ/duanrj/SpectrumDG
  conda activate wifi-diff
  CUDA_VISIBLE_DEVICES=5 python test.py --run_dir run/baseline_3urban
  # 覆盖测试场景 / checkpoint
  python test.py --run_dir run/baseline_3urban --test_terrain 01.Grassland,02.Island
  # 旧实验目录 (无 config.json, 需用 --exp 指明训练时预设)
  python test.py --run_dir run/causal_3urban --exp causal
"""
import argparse
import json
import os
from dataclasses import fields

import torch
import numpy as np

from configs import EXPERIMENTS, TrainConfig
from data_process.dataset import build_dataloaders
from lib.evaluate import evaluate_report
from model.radiownet import RadioWNet


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
    return p.parse_args()


def load_config(args):
    """run_dir/config.json + 命令行覆盖; 旧目录回退到 --exp 预设。"""
    overrides = {}
    for f in fields(TrainConfig):
        val = getattr(args, f.name, None)
        if val is not None:
            overrides[f.name] = val

    cfg_path = os.path.join(args.run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = TrainConfig(**{**json.load(f), **overrides})
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

    # ---- 数据: val (in-domain) + 每个测试场景独立 loader ----
    freqs = [s.strip() for s in cfg.freqs.split(",") if s.strip()] or None
    train_terrs = [s.strip() for s in cfg.train_terrains.split(",") if s.strip()]
    test_terrs = [s.strip() for s in cfg.test_terrains.split(",") if s.strip()]
    common = dict(z_layer=cfg.z_layer, freqs=freqs, batch_size=cfg.batch_size,
                  num_workers=cfg.num_workers, max_per_terrain=cfg.max_per_terrain,
                  root=cfg.data_root, seed=cfg.seed)
    _, val_dl, *_ = build_dataloaders(train_terrs, test_terrs[0], **common)
    test_dls = {}
    for tt in test_terrs:
        _, _, tdl, *_ = build_dataloaders(train_terrs, tt, **common)
        test_dls[tt] = tdl
    print(f"val={len(val_dl.dataset)} "
          f"test={{{', '.join(f'{t}={len(test_dls[t].dataset)}' for t in test_terrs)}}}")

    # ---- 模型: 与训练时同构, 加载两阶段权重 ----
    model = RadioWNet(inputs=2, phase="secondU",
                      use_causal=cfg.use_causal,
                      num_domains=len(train_terrs) if cfg.use_causal else 0,
                      causal_lambda_grl=cfg.causal_lambda_grl,
                      use_causal_2nd=cfg.use_causal and cfg.dg_second).to(device)
    ckpt_path = args.ckpt or os.path.join(args.run_dir, "Trained_Model_SecondU.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"loaded: {ckpt_path}")

    # ---- 评估 ----
    print("\n===== Test: 分域评估 (firstU + secondU + gated) =====")
    report = evaluate_report(model, val_dl, test_dls, test_terrs, device)

    rep_path = args.report or os.path.join(args.run_dir, "dg_report.json")
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
