"""评估: in-domain / cross-domain 的 MAE + MSE + PSNR (firstU / secondU / gated)。

门控推理统一使用多物理先验 v4 规则 (无需训练, 可解释):
  v4 规则: bldg_strong OR (elev_low AND bldg_weak)
  bldg_strong: bldg_cov > 0.01 OR bldg_ncomp > 5
  elev_low:    elev_mean < 100
  bldg_weak:   bldg_cov > 0.001 OR bldg_ncomp > 0
  判定 True → secondU (域内专家), False → firstU (泛化专家)。

evaluate(..., use_gate=False) 只算 firstU/secondU 裸指标 (训练期验证用,
不需要 terrain_yx); evaluate_report() 恒定挂 v4 门控输出 gated 指标。
"""
import math
import statistics

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage


# ------------------------------------------------------------------ #
# 多物理先验门控 v4
# ------------------------------------------------------------------ #

def _extract_gate_features(bldg, terrain_yx):
    """从建筑 mask + 地形高程图提取门控特征。"""
    feats = {}
    feats['bldg_cov'] = float(bldg.mean())
    labeled, ncomp = ndimage.label(bldg > 0.5)
    feats['bldg_ncomp'] = float(ncomp)
    feats['elev_mean'] = float(terrain_yx.mean())
    return feats


def _multi_prior_gate_v4(feats):
    """多物理先验门控 v4: 建筑信号 OR (低高程 AND 弱建筑信号)。

    源域(城市)低海拔 ~31, OOD(自然)高海拔 450-3128。
    高程 < 100 → 大概率源域; 建筑覆盖率 > 1% → 确定源域。
    """
    bldg_strong = feats['bldg_cov'] > 0.01 or feats['bldg_ncomp'] > 5
    elev_low = feats['elev_mean'] < 100
    bldg_weak = feats['bldg_cov'] > 0.001 or feats['bldg_ncomp'] > 0
    return bldg_strong or (elev_low and bldg_weak)


# ------------------------------------------------------------------ #
# 核心评估函数
# ------------------------------------------------------------------ #

def evaluate(model, loader, device, use_gate=False):
    """返回 {MAE_firstU, MSE_firstU, PSNR_firstU, ...} 三指标 × 三输出。

    Args:
        model: RadioWNet (phase="secondU", 已加载权重) 或 DualExpertNet
        loader: DataLoader (训练期验证为 3-tuple; 门控评估必须为 4-tuple 含 terrain_yx)
        device: torch.device
        use_gate: False 仅算 firstU/secondU; True 追加多物理先验 v4 gated 指标
    """
    model.eval()
    stats = {
        'firstU':  {'mae_sum': 0.0, 'mse_sum': 0.0, 'n': 0},
        'secondU': {'mae_sum': 0.0, 'mse_sum': 0.0, 'n': 0},
        'gated':   {'mae_sum': 0.0, 'mse_sum': 0.0, 'n': 0, 'n_second': 0},
    }
    mae_crit = nn.L1Loss()
    mse_crit = nn.MSELoss()
    n = 0
    max_val = 0.0

    with torch.no_grad():
        for batch in loader:
            # 兼容 3-tuple (inp, tgt, domain_id) 和 4-tuple (inp, tgt, domain_id, terrain_yx)
            if len(batch) == 4:
                inputs, targets, _, terrain_yx = batch
                terrain_yx = terrain_yx.to(device)
            else:
                inputs, targets, _ = batch
                terrain_yx = None

            inputs = inputs.to(device)
            targets = targets.to(device)
            B = targets.size(0)
            max_val = max(max_val, targets.max().item())
            o1, o2 = model(inputs)

            for key, out in (("firstU", o1), ("secondU", o2)):
                stats[key]['mae_sum'] += mae_crit(out, targets).item() * B
                stats[key]['mse_sum'] += mse_crit(out, targets).item() * B
                stats[key]['n'] += B

            if use_gate:
                if terrain_yx is None:
                    raise ValueError("use_gate=True 需要数据集返回 terrain_yx "
                                     "(设置 return_terrain=True)")
                # 多物理先验 v4: 逐样本判断
                use_second_list = []
                for i in range(B):
                    bldg = inputs[i, 0].cpu().numpy()
                    t_yx = terrain_yx[i].cpu().numpy()
                    feats = _extract_gate_features(bldg, t_yx)
                    use_second_list.append(_multi_prior_gate_v4(feats))
                use_second = torch.tensor(use_second_list, dtype=torch.bool,
                                          device=device).view(-1, 1, 1, 1)

                pred = torch.where(use_second, o2, o1)
                stats['gated']['mae_sum'] += mae_crit(pred, targets).item() * B
                stats['gated']['mse_sum'] += mse_crit(pred, targets).item() * B
                stats['gated']['n'] += B
                stats['gated']['n_second'] += int(use_second.sum().item())

            n += B

    # 汇总
    result = {}
    max_sq = max_val ** 2 if max_val > 0 else 1.0
    for key in ("firstU", "secondU", "gated"):
        s = stats[key]
        nn_val = s['n']
        if nn_val == 0:
            continue
        mae = s['mae_sum'] / nn_val
        mse = s['mse_sum'] / nn_val
        psnr = 10 * math.log10(max_sq / max(mse, 1e-12))
        result[f'MAE_{key}'] = mae
        result[f'MSE_{key}'] = mse
        result[f'PSNR_{key}'] = psnr

    if use_gate and stats['gated']['n'] > 0:
        result['gate_second_frac'] = stats['gated']['n_second'] / stats['gated']['n']
    return result


def _print_metrics(tag, res, n):
    print(f"  --- {tag} (n={n}) ---")
    for k, v in res.items():
        if 'PSNR' in k:
            print(f"  {k}: {v:.4f} dB")
        elif k == 'gate_second_frac':
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v:.6f}")


def evaluate_report(model, val_dl, test_dls, test_terrs, device):
    """in-domain (val 划分) + cross-domain (多场景) 评估, 恒定挂多物理先验 v4 门控。

    Args:
        model: 已加载权重的 RadioWNet / DualExpertNet
        val_dl: 训练场景 85/15 划分出的验证集 loader (4-tuple, 含 terrain_yx)
        test_dls: {terrain: DataLoader}, 每个跨域测试场景一份
        test_terrs: 测试场景名列表 (test_dls 的键序)
        device: torch.device

    Returns:
        report dict: {"indomain": {...}, "cross_<terrain>": {...}, "cross_avg": {...},
        "gate": {"type": "multi_prior_v4", "rule": ...}}
    """
    gate_info = {"type": "multi_prior_v4",
                 "rule": "bldg_strong OR (elev_low AND bldg_weak)"}
    print("[Gate] 多物理先验门控 v4: 建筑覆盖率 + 地形高程组合规则")

    report = {}
    res = evaluate(model, val_dl, device, use_gate=True)
    _print_metrics("indomain", res, len(val_dl.dataset))
    report["indomain"] = res

    # 跨域累计
    cross_keys = ['MAE_firstU', 'MSE_firstU', 'PSNR_firstU',
                  'MAE_secondU', 'MSE_secondU', 'PSNR_secondU',
                  'MAE_gated', 'MSE_gated', 'PSNR_gated']
    cross_sums = {k: [] for k in cross_keys}

    for tt in test_terrs:
        res = evaluate(model, test_dls[tt], device, use_gate=True)
        key = f"cross_{tt}"
        _print_metrics(key, res, len(test_dls[tt].dataset))
        report[key] = res
        for k in cross_keys:
            if k in res:
                cross_sums[k].append(res[k])

    report["cross_avg"] = {k: statistics.mean(v) for k, v in cross_sums.items() if v}
    _print_metrics("cross_avg", report["cross_avg"], "-")
    report["gate"] = gate_info
    return report
