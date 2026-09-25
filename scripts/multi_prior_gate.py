"""多物理先验域分类器设计与验证。

从输入 2 通道 (buildings + tx_prox) 提取丰富的统计量,
评估各特征的源域/未见域分离度, 并设计融合策略:
  1. 规则融合: 多阈值 AND/OR 逻辑
  2. 一类分类: Isolation Forest 自动学习源域分布

用法:
  cd /data/userSZ/duanrj/SpectrumDG
  python scripts/multi_prior_gate.py
  python scripts/multi_prior_gate.py --max_per_terrain 200  # 快速
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import ndimage

from data_process.dataset import SpectrumNetDataset, TERRAIN_ORDER


def extract_features(inp, npz_path=None):
    """从单样本输入 (2, H, W) + npz 文件提取多物理先验特征。

    Returns:
        dict of scalar features
    """
    bldg = inp[0]  # (H, W) 0/1
    prox = inp[1]  # (H, W) [0, 1]

    feats = {}

    # ---- 建筑统计量 ----
    feats['bldg_cov'] = float(bldg.mean())
    labeled, ncomp = ndimage.label(bldg > 0.5)
    feats['bldg_ncomp'] = float(ncomp)
    if ncomp > 0:
        areas = np.bincount(labeled.ravel())[1:]
        feats['bldg_med_area'] = float(np.median(areas))
        feats['bldg_area_std'] = float(areas.std())
        feats['bldg_max_area'] = float(areas.max())
    else:
        feats['bldg_med_area'] = 0.0
        feats['bldg_area_std'] = 0.0
        feats['bldg_max_area'] = 0.0
    feats['bldg_frac_log'] = float(np.log10(feats['bldg_cov'] + 1e-3))

    # 建筑边缘密度 (Sobel)
    if feats['bldg_cov'] > 0.001:
        sx = ndimage.sobel(bldg, axis=0, mode='constant')
        sy = ndimage.sobel(bldg, axis=1, mode='constant')
        edges = np.hypot(sx, sy)
        feats['bldg_edge_density'] = float((edges > 0.5).mean())
    else:
        feats['bldg_edge_density'] = 0.0

    # 建筑空间均匀性 (分块覆盖率变异系数)
    block_size = 16
    n_blocks_h = bldg.shape[0] // block_size
    n_blocks_w = bldg.shape[1] // block_size
    if n_blocks_h > 0 and n_blocks_w > 0:
        block_covers = []
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                block = bldg[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_covers.append(block.mean())
        block_covers = np.array(block_covers)
        feats['bldg_block_cv'] = float(block_covers.std() / (block_covers.mean() + 1e-8))
    else:
        feats['bldg_block_cv'] = 0.0

    # ---- Tx 邻近统计量 ----
    feats['tx_prox_max'] = float(prox.max())
    feats['tx_prox_mean'] = float(prox.mean())
    feats['tx_prox_std'] = float(prox.std())
    feats['tx_prox_p90'] = float(np.quantile(prox, 0.9))
    # Tx 邻近图梯度 (空间变化率)
    gx = np.abs(np.diff(prox, axis=1)).mean()
    gy = np.abs(np.diff(prox, axis=0)).mean()
    feats['tx_prox_gradient'] = float((gx + gy) / 2)

    # ---- 地形高程统计量 (从 npz 加载) ----
    if npz_path is not None and os.path.exists(npz_path):
        npz = np.load(npz_path)
        if 'terrain_yx' in npz:
            terrain = npz['terrain_yx'].astype(float)
            feats['elev_mean'] = float(terrain.mean())
            feats['elev_std'] = float(terrain.std())
            feats['elev_range'] = float(terrain.max() - terrain.min())
            feats['elev_p25'] = float(np.quantile(terrain, 0.25))
            feats['elev_p75'] = float(np.quantile(terrain, 0.75))
            feats['elev_iqr'] = feats['elev_p75'] - feats['elev_p25']
            # 高程梯度 (地形起伏度)
            gx_t = np.abs(np.diff(terrain, axis=1)).mean()
            gy_t = np.abs(np.diff(terrain, axis=0)).mean()
            feats['elev_gradient'] = float((gx_t + gy_t) / 2)
            # 高程变异系数
            feats['elev_cv'] = float(terrain.std() / (terrain.mean() + 1e-8))
            
            # ---- 高程纹理特征 (隐含材质信息) ----
            # 局部方差 (纹理粗糙度)
            from scipy.ndimage import uniform_filter
            local_mean = uniform_filter(terrain, size=5)
            local_sq_mean = uniform_filter(terrain**2, size=5)
            local_var = local_sq_mean - local_mean**2
            feats['elev_texture_var'] = float(local_var.mean())
            
            # 高程拉普拉斯 (二阶导数, 边缘/角点响应)
            laplacian = ndimage.laplace(terrain)
            feats['elev_laplacian_energy'] = float(np.abs(laplacian).mean())
            
            # 高程方向性 (各向异性)
            gx = ndimage.sobel(terrain, axis=1)
            gy = ndimage.sobel(terrain, axis=0)
            gradient_mag = np.hypot(gx, gy)
            feats['elev_gradient_mag'] = float(gradient_mag.mean())
            # 梯度方向熵 (方向均匀性)
            gradient_dir = np.arctan2(gy, gx + 1e-8)
            hist = np.histogram(gradient_dir, bins=16, range=(-np.pi, np.pi))[0].astype(float)
            hist = hist / max(hist.sum(), 1)
            feats['elev_dir_entropy'] = float(-(hist * np.log10(hist + 1e-9)).sum())
        npz.close()
    else:
        # 无 npz 文件时填充默认值
        for k in ['elev_mean', 'elev_std', 'elev_range', 'elev_p25', 'elev_p75',
                  'elev_iqr', 'elev_gradient', 'elev_cv', 'elev_texture_var',
                  'elev_laplacian_energy', 'elev_gradient_mag', 'elev_dir_entropy']:
            feats[k] = 0.0

    return feats


def main():
    p = argparse.ArgumentParser(description="多物理先验域分类器")
    p.add_argument("--max_per_terrain", type=int, default=150)
    args = p.parse_args()

    src_terrs = ["05.Suburban", "06.DenseUrban", "07.Rural", "08.OrdinaryUrban"]
    ood_terrs = ["01.Grassland", "09.Desert", "10.Mountainous", "11.Forest"]

    all_feats, all_doms = [], []
    for tag, terrs in (("src", src_terrs), ("ood", ood_terrs)):
        for t in terrs:
            ds = SpectrumNetDataset([t], split="all", z_layer="00",
                                    max_per_terrain=args.max_per_terrain)
            for i in range(len(ds)):
                inp, _, _ = ds[i]
                # 获取 npz 路径以加载地形高程
                _, npz_path, _ = ds.items[i]
                feats = extract_features(inp.numpy(), npz_path=npz_path)
                all_feats.append(feats)
                all_doms.append(TERRAIN_ORDER.index(t))
            print(f"[{tag}] {t}: n={len(ds)}")

    # 转为 DataFrame-like 结构
    feat_names = list(all_feats[0].keys())
    feats_arr = np.array([[f[n] for n in feat_names] for f in all_feats])
    doms_arr = np.array(all_doms)

    src_ids = [TERRAIN_ORDER.index(t) for t in src_terrs]
    src_mask = np.isin(doms_arr, src_ids)
    src_feats = feats_arr[src_mask]

    # ---- 打印各特征统计量 ----
    print(f"\n{'feature':20s} {'src_med':>10s}", end="")
    for t in ood_terrs:
        print(f" {t[-6:]:>10s}", end="")
    print()

    ood_by_terrain = {t: feats_arr[doms_arr == TERRAIN_ORDER.index(t)] for t in ood_terrs}
    for j, name in enumerate(feat_names):
        row = [f"{ood_by_terrain[t][:, j].mean():10.3f}" for t in ood_terrs]
        print(f"{name:20s} {np.median(src_feats[:, j]):10.3f} {' '.join(row)}")

    # ---- 各特征 AUROC ----
    def auroc(s, o):
        """P(源域样本得分 > OOD 样本得分), >0.5 表示源域更高"""
        all_v = np.concatenate([s, o])
        ranks = all_v.argsort().argsort().astype(float) + 1
        r_o = ranks[len(s):]
        return ((r_o.sum() - len(o) * (len(o) + 1) / 2) / (len(s) * len(o)))

    print(f"\n{'feature':20s} {'AUROC':>7s} (每测试域, >0.5=源域更高)")
    discriminative = []
    for j, name in enumerate(feat_names):
        aucs = [auroc(src_feats[:, j], ood_by_terrain[t][:, j]) for t in ood_terrs]
        avg_auc = np.mean(aucs)
        auc_str = " ".join(f"{a:6.3f}" for a in aucs)
        print(f"{name:20s} avg={avg_auc:.3f}  {auc_str}")
        # 标记有判别力的特征 (AUROC > 0.7 或 < 0.3)
        if avg_auc > 0.7 or avg_auc < 0.3:
            discriminative.append((name, j, avg_auc))

    print(f"\n有判别力的特征 (|AUROC - 0.5| > 0.2):")
    for name, j, auc in discriminative:
        direction = "源域更高" if auc > 0.5 else "OOD更高"
        print(f"  {name:20s} AUROC={auc:.3f} ({direction})")

    # ---- 方案 1: 规则融合 (建筑 + 高程 独立信号) ----
    print("\n" + "="*60)
    print("方案 1: 规则融合 (建筑 + 高程 独立信号)")
    print("="*60)

    # 设计规则: 建筑 OR 高程 任一强信号即可
    def rule_gate_v5(feats_dict):
        """规则门控 v5: 建筑信号 OR 低高程信号 (独立)"""
        # 建筑信号 (强)
        bldg_strong = feats_dict['bldg_cov'] > 0.01 or feats_dict['bldg_ncomp'] > 5
        # 高程信号: 源域低高程 (<100), OOD 高高程 (450+)
        # 即使建筑为 0, 低高程仍可独立判断为源域
        elev_low = feats_dict['elev_mean'] < 100
        # OR 逻辑: 任一信号触发即源域
        return bldg_strong or elev_low

    # 评估规则门控
    print(f"\n{'domain':16s} {'src_route_to_2nd':>16s} {'ood_route_to_2nd':>18s}")
    for t in src_terrs:
        m = doms_arr == TERRAIN_ORDER.index(t)
        n_src = m.sum()
        n_route = sum(1 for i in np.where(m)[0] if rule_gate_v5(all_feats[i]))
        print(f"{t:16s} {n_route}/{n_src} = {n_route/n_src:.3f}")
    for t in ood_terrs:
        m = doms_arr == TERRAIN_ORDER.index(t)
        n_ood = m.sum()
        n_route = sum(1 for i in np.where(m)[0] if rule_gate_v5(all_feats[i]))
        print(f"{t:16s} {n_route}/{n_ood} = {n_route/n_ood:.3f}")

    # ---- 方案 2: Isolation Forest 一类分类 ----
    print("\n" + "="*60)
    print("方案 2: Isolation Forest 一类分类 (仅源域训练)")
    print("="*60)

    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    # 选择有判别力的特征
    sel_idx = [j for _, j, _ in discriminative]
    if len(sel_idx) == 0:
        print("警告: 无有判别力的特征, 使用全部特征")
        sel_idx = list(range(len(feat_names)))

    sel_names = [feat_names[j] for j in sel_idx]
    print(f"使用特征: {sel_names}")

    X_src = src_feats[:, sel_idx]
    scaler = StandardScaler().fit(X_src)
    X_src_norm = scaler.transform(X_src)

    # 训练 Isolation Forest (contamination=0 表示源域无异常)
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso.fit(X_src_norm)

    # 评估
    print(f"\n{'domain':16s} {'inlier_rate':>12s} (源域应高, OOD应低)")
    for t in src_terrs:
        m = doms_arr == TERRAIN_ORDER.index(t)
        X_t = feats_arr[m][:, sel_idx]
        X_t_norm = scaler.transform(X_t)
        preds = iso.predict(X_t_norm)  # 1=inlier, -1=outlier
        inlier_rate = (preds == 1).mean()
        print(f"{t:16s} {inlier_rate:.3f}")
    for t in ood_terrs:
        m = doms_arr == TERRAIN_ORDER.index(t)
        X_t = feats_arr[m][:, sel_idx]
        X_t_norm = scaler.transform(X_t)
        preds = iso.predict(X_t_norm)
        inlier_rate = (preds == 1).mean()
        print(f"{t:16s} {inlier_rate:.3f}")

    # ---- 综合对比 ----
    print("\n" + "="*60)
    print("综合对比: 路由准确率")
    print("="*60)

    # 规则门控准确率
    n_correct = 0
    n_total = 0
    for i in range(len(all_feats)):
        is_src = src_mask[i]
        pred_src = rule_gate_v5(all_feats[i])
        if pred_src == is_src:
            n_correct += 1
        n_total += 1
    rule_acc = n_correct / n_total
    print(f"规则融合 (建筑+高程 独立) 准确率: {rule_acc:.3f}")

    # Isolation Forest 准确率
    X_all = feats_arr[:, sel_idx]
    X_all_norm = scaler.transform(X_all)
    preds_all = iso.predict(X_all_norm)
    pred_src_all = (preds_all == 1)
    iso_acc = (pred_src_all == src_mask).mean()
    print(f"Isolation Forest 准确率: {iso_acc:.3f}")

    # 单特征 bldg_cov 基线
    bldg_cov_thr = 0.01
    pred_src_bldg = feats_arr[:, feat_names.index('bldg_cov')] > bldg_cov_thr
    bldg_acc = (pred_src_bldg == src_mask).mean()
    print(f"单特征 bldg_cov>0.01 基线: {bldg_acc:.3f}")


if __name__ == "__main__":
    main()
