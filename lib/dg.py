"""域泛化 (DG) 组件: 损失函数与训练策略。

包含两层 DG 干预手段 (可与 ERM 任意组合):
  CIRL 因果分解 (瓶颈层, model/causal.py): 域分类 + GRL 对抗 + 正交解相关
  Fish 梯度对齐 (损失函数层): ERM + λ_fish·Σ_d ||∇_θ L_d - ∇_θ L̄||²
"""
import torch

from data_process.dataset import TERRAIN_ORDER


def build_domain_map(train_terrains):
    """全局 terrain_id → 训练域局部 id (供 CausalDecomposer 域分类器)。"""
    return {TERRAIN_ORDER.index(terr): i for i, terr in enumerate(train_terrains)}


def build_domain_label_tensor(domain_map, device):
    """全局 terrain_id → 训练域局部 id 查表张量 (供 CausalDecomposer 域分类)。"""
    dmap = torch.zeros(len(TERRAIN_ORDER), dtype=torch.long, device=device)
    for g, l in domain_map.items():
        dmap[g] = l
    return dmap


def cirl_aux_loss(decomposer, local_d, ce):
    """CIRL 辅助损失: 返回 (loss_dom, loss_inv, loss_orth) 或 None (decomposer 未启用)。"""
    if decomposer is None or decomposer._last_aux is None:
        return None
    logits_spe, logits_inv, orth = decomposer._last_aux
    return ce(logits_spe, local_d), ce(logits_inv, local_d), orth


def fish_penalty(out, targets, domain_ids, params, crit):
    """Fishr 风格梯度对齐惩罚: Σ_d ||∇_θ L_d - ∇_θ L̄||², 约束跨域梯度方向一致。
    返回 None 当 batch 内有效域 (<2 个样本) 不足 2 个。"""
    unique_d = torch.unique(domain_ids)
    masks = [domain_ids == d for d in unique_d if (domain_ids == d).sum() >= 2]
    if len(masks) < 2:
        return None
    per_domain_grads = []
    for m in masks:
        loss_d = crit(out[m], targets[m])
        g = torch.autograd.grad(loss_d, params, retain_graph=True,
                                create_graph=True, allow_unused=True)
        flat = torch.cat([gi.reshape(-1) for gi in g if gi is not None])
        per_domain_grads.append(flat)
    grad_stack = torch.stack(per_domain_grads)  # (D, P)
    grad_mean = grad_stack.mean(0, keepdim=True)
    return ((grad_stack - grad_mean) ** 2).sum()
