"""评估: in-domain / cross-domain 的 MSE + NMSE (firstU / secondU / gated)。

gated 门控推理: 输入通道 0 (buildings) 建筑覆盖率 >= threshold → 源域 → secondU,
否则 → 未见域 → firstU。阈值固定 0.01 (1%), 源域中位数 ~5%, OOD ~0-0.1%。
"""
import statistics
from collections import defaultdict

import torch
import torch.nn as nn


def evaluate(model, loader, device, gate=None, gate_threshold=None):
    """返回 {MSE_firstU, NMSE_firstU, MSE_secondU, NMSE_secondU,
    (可选) MSE_gated, NMSE_gated, gate_second_frac}。

    firstU=output1, secondU=output2。NMSE = MSE(pred,tgt)/MSE(tgt,0)。
    gate='simple': 输入通道 0 (buildings) 覆盖率 >= threshold → 源域 → secondU;
                   否则 → 未见域 → firstU。threshold 默认 0.01 (1% 覆盖)。
    """
    model.eval()
    metrics = defaultdict(float)
    crit = nn.MSELoss()
    n = 0
    n_second = 0
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            o1, o2 = model(inputs)
            tgt_mse = crit(targets, torch.zeros_like(targets)) + 1e-12
            for key, out in (("firstU", o1), ("secondU", o2)):
                mse = crit(out, targets)
                metrics[f'MSE_{key}'] += mse.item() * targets.size(0)
                metrics[f'NMSE_{key}'] += (mse / tgt_mse).item() * targets.size(0)

            if gate == 'simple':
                # 简单阈值门控: 建筑覆盖率 >= threshold → 源域
                bldg_cov = inputs[:, 0].mean(dim=(1, 2))  # (B,) 每样本建筑覆盖率
                thr = gate_threshold if gate_threshold is not None else 0.01
                use_second = (bldg_cov >= thr).view(-1, 1, 1, 1)
                pred = torch.where(use_second, o2, o1)
                mse = crit(pred, targets)
                metrics['MSE_gated'] += mse.item() * targets.size(0)
                metrics['NMSE_gated'] += (mse / tgt_mse).item() * targets.size(0)
                n_second += int(use_second.sum().item())
            n += targets.size(0)
    out = {k: v / n for k, v in metrics.items()}
    if 'NMSE_gated' in out:
        out['gate_second_frac'] = n_second / n
    return out


def _print_metrics(tag, res, n):
    print(f"  --- {tag} (n={n}) ---")
    for k, v in res.items():
        print(f"  {k}: {v:.6f}")


def evaluate_report(model, val_dl, test_dls, test_terrs, device,
                    gate='simple', gate_threshold=None):
    """in-domain (val 划分) + cross-domain (多场景) 评估。

    Args:
        model: 已加载权重的 RadioWNet
        val_dl: 训练场景 85/15 划分出的验证集 loader
        test_dls: {terrain: DataLoader}, 每个跨域测试场景一份
        test_terrs: 测试场景名列表 (test_dls 的键序)
        device: torch.device
        gate: 'simple' (建筑覆盖率阈值门控)
        gate_threshold: 门控阈值; 默认 0.01 (1%)

    Returns:
        report dict: {"indomain": {...}, "cross_<terrain>": {...}, "cross_avg": {...},
        (有门控时) "gate": {"threshold", "type", "val_bldg_cov_mean"}}
    """
    # ---- 门控阈值校准: 仅用源域 val (不接触目标域) ----
    gate_info = None
    if gate == 'simple':
        # 从 val 收集建筑覆盖率, 校准阈值
        model.eval()
        covs = []
        with torch.no_grad():
            for inputs, _, _ in val_dl:
                inputs = inputs.to(device)
                covs.append(inputs[:, 0].mean(dim=(1, 2)).cpu())
        val_covs = torch.cat(covs)
        if gate_threshold is None:
            # 固定阈值 0.01 (1%): 源域中位数 ~5%, OOD ~0-0.1%, 安全分离
            gate_threshold = 0.01
        gate_info = {
            "threshold": gate_threshold,
            "type": "simple_bldg_cov",
            "val_bldg_cov_mean": float(val_covs.mean()),
            "val_bldg_cov_min": float(val_covs.min()),
        }
        print(f"[Gate] 简单阈值门控: threshold={gate_threshold:.4f} "
              f"(val 建筑覆盖率 mean={gate_info['val_bldg_cov_mean']:.4f}, "
              f"min={gate_info['val_bldg_cov_min']:.4f})")

    report = {}
    res = evaluate(model, val_dl, device, gate=gate, gate_threshold=gate_threshold)
    _print_metrics("indomain", res, len(val_dl.dataset))
    report["indomain"] = res

    cross_nmse1, cross_nmse2, cross_nmse_g = [], [], []
    for tt in test_terrs:
        res = evaluate(model, test_dls[tt], device, gate=gate, gate_threshold=gate_threshold)
        key = f"cross_{tt}"
        _print_metrics(key, res, len(test_dls[tt].dataset))
        report[key] = res
        cross_nmse1.append(res['NMSE_firstU'])
        cross_nmse2.append(res['NMSE_secondU'])
        if 'NMSE_gated' in res:
            cross_nmse_g.append(res['NMSE_gated'])

    report["cross_avg"] = {
        "NMSE_firstU": statistics.mean(cross_nmse1),
        "NMSE_secondU": statistics.mean(cross_nmse2),
    }
    if cross_nmse_g:
        report["cross_avg"]["NMSE_gated"] = statistics.mean(cross_nmse_g)
    _print_metrics("cross_avg", report["cross_avg"], "-")
    if gate_info is not None:
        report["gate"] = gate_info
    return report
