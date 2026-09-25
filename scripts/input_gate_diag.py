"""输入空间门控信号诊断: buildings 通道统计量的源域/未见域分离度。

瓶颈特征 (z_spe/z_inv) 的高斯几何对未见域无判别力 (AUROC < 0.57),
但源域均为城市/郊区类地形 (建筑丰富), 测试域均为自然地形 ——
本质差异在输入空间。本脚本验证 buildings 通道统计量 (覆盖率/块数/
块大小等) 的分离度, 无需模型前向。

用法:
  cd /data/userSZ/duanrj/SpectrumDG
  python scripts/input_gate_diag.py
  python scripts/input_gate_diag.py --max_per_terrain 200  # 快速
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import ndimage

from data_process.dataset import SpectrumNetDataset, TERRAIN_ORDER


def input_stats(inp):
    """单样本输入统计量 (buildings + tx_prox 两通道)。

    Returns:
        (7,) 向量: [bldg_cov, bldg_ncomp, bldg_med_area, bldg_area_std,
                    bldg_frac_log, tx_prox_p90, tx_prox_entropy]
    """
    bldg = inp[0]  # (H, W) 0/1
    prox = inp[1]
    cov = float(bldg.mean())
    labeled, ncomp = ndimage.label(bldg > 0.5)
    areas = np.bincount(labeled.ravel())[1:] if ncomp > 0 else np.array([0.0])
    med_area = float(np.median(areas))
    area_std = float(areas.std())
    # 建筑覆盖率取 log 避免量纲悬殊 (0 → 用小值)
    frac_log = float(np.log10(cov + 1e-3))
    p90 = float(np.quantile(prox, 0.9))
    hist = np.histogram(prox, bins=16, range=(0, 1))[0].astype(float)
    hist = hist / max(hist.sum(), 1)
    entropy = float(-(hist * np.log10(hist + 1e-9)).sum())
    return np.array([cov, ncomp, med_area, area_std, frac_log, p90, entropy])


def main():
    p = argparse.ArgumentParser(description="输入空间门控信号诊断")
    p.add_argument("--max_per_terrain", type=int, default=200,
                   help="每域样本上限 (统计量稳定即可)")
    args = p.parse_args()

    src_terrs = ["05.Suburban", "06.DenseUrban", "07.Rural", "08.OrdinaryUrban"]
    ood_terrs = ["01.Grassland", "09.Desert", "10.Mountainous", "11.Forest"]

    feats, doms = [], []
    for tag, terrs, is_ood in (("src", src_terrs, 0), ("ood", ood_terrs, 1)):
        for t in terrs:
            ds = SpectrumNetDataset([t], split="all", z_layer="00",
                                    max_per_terrain=args.max_per_terrain)
            for i in range(len(ds)):
                inp, _, _ = ds[i]
                feats.append(input_stats(inp.numpy()))
                doms.append(TERRAIN_ORDER.index(t))
            print(f"[{tag}] {t}: n={len(ds)}")
    feats = np.stack(feats)
    doms = np.array(doms)

    names = ["bldg_cov", "ncomp", "med_area", "area_std",
             "frac_log", "prox_p90", "prox_ent"]
    src_ids = [TERRAIN_ORDER.index(t) for t in src_terrs]
    src_mask = np.isin(doms, src_ids)
    src_m = feats[src_mask]
    print(f"\n{'feature':10s} {'src_med':>10s} {'Grass':>8s} {'Desert':>8s} "
          f"{'Mount':>8s} {'Forest':>8s}")
    ood_by_terrain = {t: feats[doms == TERRAIN_ORDER.index(t)] for t in ood_terrs}
    for j, name in enumerate(names):
        row = [f"{ood_by_terrain[t][:, j].mean():8.3f}" for t in ood_terrs]
        print(f"{name:10s} {np.median(src_m[:, j]):10.3f} {' '.join(row)}")

    # ---- 分离度: 逐特征 AUROC (简单rank法) ----
    def auroc(s, o):
        all_v = np.concatenate([s, o])
        ranks = all_v.argsort().argsort().astype(float) + 1
        r_o = ranks[len(s):]
        return ((r_o.sum() - len(o) * (len(o) + 1) / 2) / (len(s) * len(o)))

    print(f"\n{'feature':10s} {'AUROC':>7s} (每测试域)")
    for j, name in enumerate(names):
        aucs = [auroc(src_m[:, j], ood_by_terrain[t][:, j]) for t in ood_terrs]
        auc_str = " ".join(f"{a:6.3f}" for a in aucs)
        print(f"{name:10s} {auc_str}")

    # ---- 简单组合: buildings 覆盖率 + ncomp 的 kNN/马氏距离 ----
    # 用统计量向量 (标准化) 做与 gate.py 相同的马氏评分
    from lib.gate import MahalanobisGate
    sel = [0, 1, 4]  # cov, ncomp, frac_log
    f_sel = feats[:, sel]
    mu, sd = f_sel[src_mask].mean(0), f_sel[src_mask].std(0) + 1e-8
    f_norm = (f_sel - mu) / sd
    gate = MahalanobisGate(shrinkage=0.1, shared_cov=False).fit(
        torch.from_numpy(f_norm[src_mask]).float(),
        torch.tensor(doms[src_mask]))
    src_scores = gate.score(torch.from_numpy(f_norm[src_mask]).float())
    print(f"\n=== 输入统计量 + 逐域马氏 (src val median = {src_scores.median():.2f}) ===")
    print(f"{'domain':16s} {'score_med':>10s} {'AUROC':>7s} {'det@FPR5%':>10s}")
    for t in ood_terrs:
        m = doms == TERRAIN_ORDER.index(t)
        ood_scores = gate.score(torch.from_numpy(f_norm[m]).float())
        all_v = torch.cat([src_scores, ood_scores])
        ranks = all_v.argsort().argsort().float() + 1
        r_o = ranks[len(src_scores):]
        auc = ((r_o.sum() - len(ood_scores) * (len(ood_scores) + 1) / 2)
               / (len(src_scores) * len(ood_scores))).item()
        thr = torch.quantile(src_scores, 0.95)
        det = (ood_scores > thr).float().mean().item()
        print(f"{t:16s} {ood_scores.median():10.2f} {auc:7.3f} {det:10.3f}")


if __name__ == "__main__":
    main()
