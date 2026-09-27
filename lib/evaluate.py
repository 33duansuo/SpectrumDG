"""评估: in-domain / cross-domain 的 MAE + MSE + PSNR (firstU / secondU / gated)。

门控推理两种实现 (判定 True → secondU 域内专家, False → firstU 泛化专家):
  v4 规则 (默认对照, 无需训练): bldg_strong OR (elev_low AND bldg_weak)
    bldg_strong: bldg_cov > 0.01 OR bldg_ncomp > 5
    elev_low:    elev_mean < 100
    bldg_weak:   bldg_cov > 0.001 OR bldg_ncomp > 0
  Router 门控 (学习式, 主线): BinaryRouter 的源域 score >= tau → 域内。
    test-time-geo 设定: 目标域频谱不可见、地形公开, Router 用源域真实地形
    作正样本、目标场景真实地形作负样本训练; tau 由源域 score 分位校准。

evaluate(..., use_gate=False) 只算 firstU/secondU 裸指标 (训练期验证用,
不需要 terrain_yx); evaluate_report() 挂门控输出 gated 指标
(router 参数给出时用 Router 门控, 否则 v4 规则)。
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
# Router 门控 (学习式)
# ------------------------------------------------------------------ #

@torch.no_grad()
def calibrate_router_tau(router, loader, device, recall=0.95):
    """源域 score 的 (1-recall) 分位数作为 tau: 保证源域样本召回 ≥ recall。

    loader 必须为 4-tuple (含 terrain_yx)。
    """
    router.eval()
    scores = []
    for batch in loader:
        if len(batch) != 4:
            raise ValueError("tau 校准需要数据集返回 terrain_yx (return_terrain=True)")
        inputs, _, _, terrain_yx = batch
        scores.append(router.gate_score(inputs[:, 0:1].to(device),
                                        terrain_yx.to(device)).cpu())
    scores = torch.cat(scores).numpy()
    return float(np.quantile(scores, 1.0 - recall))


# ------------------------------------------------------------------ #
# 核心评估函数
# ------------------------------------------------------------------ #

def evaluate(model, loader, device, use_gate=False, router=None, tau=0.5):
    """返回 {MAE_firstU, MSE_firstU, PSNR_firstU, ...} 三指标 × 三输出。

    Args:
        model: RadioWNet (phase="secondU", 已加载权重) 或 DualExpertNet
        loader: DataLoader (训练期验证为 3-tuple; 门控评估必须为 4-tuple 含 terrain_yx)
        device: torch.device
        use_gate: False 仅算 firstU/secondU; True 追加 gated 指标
        router: None → 多物理先验 v4 规则门控; 给出 BinaryRouter → score >= tau 门控
        tau: Router 门控阈值 (calibrate_router_tau 校准)
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
                if router is not None:
                    # Router 门控: 源域 score >= tau → 域内专家
                    score = router.gate_score(inputs[:, 0:1], terrain_yx)
                    use_second = (score >= tau).view(-1, 1, 1, 1)
                else:
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


def evaluate_report(model, val_dl, test_dls, test_terrs, device, router=None, tau=None):
    """in-domain (val 划分) + cross-domain (多场景) 评估, 挂门控输出 gated 指标。

    Args:
        model: 已加载权重的 RadioWNet / DualExpertNet
        val_dl: 训练场景 85/15 划分出的验证集 loader (4-tuple, 含 terrain_yx)
        test_dls: {terrain: DataLoader}, 每个跨域测试场景一份
        test_terrs: 测试场景名列表 (test_dls 的键序)
        device: torch.device
        router: None → 多物理先验 v4 规则门控; 给出 BinaryRouter → score 门控
        tau: Router 门控阈值; None 时按 recall=0.95 在源域集自动校准

    Returns:
        report dict: {"indomain": {...}, "cross_<terrain>": {...}, "cross_avg": {...},
        "gate": {"type": ..., ...}}
    """
    if router is not None:
        if tau is None:
            tau = calibrate_router_tau(router, val_dl, device, recall=0.95)
        gate_info = {"type": "router", "tau": tau, "recall_target": 0.95}
        print(f"[Gate] Router 门控: tau={tau:.4f} (源域 95% 召回校准)")
    else:
        gate_info = {"type": "multi_prior_v4",
                     "rule": "bldg_strong OR (elev_low AND bldg_weak)"}
        print("[Gate] 多物理先验门控 v4: 建筑覆盖率 + 地形高程组合规则")

    report = {}
    res = evaluate(model, val_dl, device, use_gate=True, router=router, tau=tau or 0.5)
    _print_metrics("indomain", res, len(val_dl.dataset))
    report["indomain"] = res

    # 跨域累计
    cross_keys = ['MAE_firstU', 'MSE_firstU', 'PSNR_firstU',
                  'MAE_secondU', 'MSE_secondU', 'PSNR_secondU',
                  'MAE_gated', 'MSE_gated', 'PSNR_gated']
    cross_sums = {k: [] for k in cross_keys}

    for tt in test_terrs:
        res = evaluate(model, test_dls[tt], device, use_gate=True,
                       router=router, tau=tau or 0.5)
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
