"""门控信号诊断: 对比不同特征/协方差配置下源域 vs 未见域的可分性。

在已有训练产物上离线运行 (无需重训), 输出各配置对每个测试域的
AUROC 与 FPR@95%TPR 对应的检测率, 用于选择门控信号。

用法:
  cd /data/userSZ/duanrj/SpectrumDG
  CUDA_VISIBLE_DEVICES=0 python scripts/gate_diag.py --run_dir run/cbam_dg_4to4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs import TrainConfig
from data_process.dataset import build_dataloaders
from lib.evaluate import collect_gate_features
from lib.gate import MahalanobisGate
from model.radiownet import RadioWNet


def auroc(src_scores, ood_scores):
    """AUROC: OOD 分数应高于源域分数。"""
    all_scores = torch.cat([src_scores, ood_scores])
    ranks = all_scores.argsort().argsort().float() + 1
    r_src, r_ood = ranks[:len(src_scores)], ranks[len(src_scores):]
    return ((r_ood.sum() - len(ood_scores) * (len(ood_scores) + 1) / 2)
            / (len(src_scores) * len(ood_scores))).item()


def detect_at_fpr(src_scores, ood_scores, fpr=0.05):
    """固定源域误报率 fpr 下的 OOD 检出率 (阈值 = 源域 1-fpr 分位)。"""
    thr = torch.quantile(src_scores, 1 - fpr)
    return (ood_scores > thr).float().mean().item()


def main():
    p = argparse.ArgumentParser(description="SpectrumDG 门控信号诊断")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_per_terrain", type=int, default=None,
                   help="每域采样上限 (快速诊断)")
    args = p.parse_args()

    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = TrainConfig(**json.load(f))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_terrs = [s.strip() for s in cfg.train_terrains.split(",") if s.strip()]
    test_terrs = [s.strip() for s in cfg.test_terrains.split(",") if s.strip()]
    common = dict(z_layer=cfg.z_layer, freqs=None, batch_size=args.batch_size,
                  num_workers=args.num_workers, max_per_terrain=args.max_per_terrain,
                  root=cfg.data_root, seed=cfg.seed)
    _, val_dl, *_ = build_dataloaders(train_terrs, test_terrs[0], **common)
    test_dls = {}
    for tt in test_terrs:
        _, _, tdl, *_ = build_dataloaders(train_terrs, tt, **common)
        test_dls[tt] = tdl

    model = RadioWNet(inputs=2, phase="secondU",
                      use_causal=cfg.use_causal,
                      num_domains=len(train_terrs) if cfg.use_causal else 0,
                      causal_lambda_grl=cfg.causal_lambda_grl,
                      use_causal_2nd=cfg.use_causal and cfg.dg_second).to(device)
    ckpt = os.path.join(args.run_dir, "Trained_Model_SecondU.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    print(f"loaded: {ckpt}")

    # ---- 收集特征: 源域 val + 各测试域 ----
    datasets = {"val(src)": val_dl}
    datasets.update({t: test_dls[t] for t in test_terrs})

    configs = [
        ("z_spe+shared", model.gate_features, dict(shared_cov=True)),
        ("z_spe+perdom", model.gate_features, dict(shared_cov=False)),
        ("z_inv+shared", model.gate_inv_features, dict(shared_cov=True)),
        ("z_inv+perdom", model.gate_inv_features, dict(shared_cov=False)),
    ]
    for name, feat_fn, kw in configs:
        feats = {}
        for tag, dl in datasets.items():
            collected = collect_gate_features(model, dl, device, feat_fn=feat_fn)
            feats[tag] = collected[0] if collected else None
        if any(v is None for v in feats.values()):
            print(f"\n=== {name}: 无信号, 跳过 ===")
            continue

        # val 内含 4 个源域, 按真实域 id 拟合高斯
        val_feats, val_dom = collect_gate_features(model, val_dl, device, feat_fn=feat_fn)
        gate = MahalanobisGate(**kw).fit(val_feats, val_dom)
        src_scores = gate.score(feats["val(src)"])

        print(f"\n=== {name} (threshold 未校准, 仅看分数分离度) ===")
        print(f"  {'domain':20s} {'score_med':>10s} {'AUROC':>7s} {'det@FPR5%':>10s}")
        for tag in test_terrs:
            ood_scores = gate.score(feats[tag])
            print(f"  {tag:20s} {ood_scores.median():10.1f} "
                  f"{auroc(src_scores, ood_scores):7.3f} "
                  f"{detect_at_fpr(src_scores, ood_scores):10.3f}")
        print(f"  {'(src val)':20s} {src_scores.median():10.1f}")


if __name__ == "__main__":
    main()
