"""源域/未见域门控: z_spe 特征空间马氏距离 OOD 评分 + LODO 阈值校准。

z_spe 由 CIRL 特意训练为域可分 (域特定风格表示), 在其 GAP 特征上拟合
每域高斯 (均值各异 + 共享收缩协方差), 未见域样本偏离所有源域簇 →
到最近中心的马氏距离大 → 路由 firstU; 反之判为源域 → 路由 secondU。

阈值用 leave-one-domain-out (LODO) 校准: 轮流留出一个源域当伪未见域,
在"其余域 vs 留出域"上选门控准确率最高的阈值, 各折平均 —— 全程不接触
真实测试域数据, 无测试集信息泄露。

相比 softmax 置信度 (MSP) 的优势: classifier_spe 在源域近乎完美,
置信度饱和在 1.0 附近 (动态范围 ~1e-3), OOD 样本极易误判高置信;
马氏距离是连续度量, 动态范围大, 对过参数化网络的过自信免疫。
"""
import torch


class MahalanobisGate:
    """基于 z_spe 特征马氏距离的源域/未见域门控。

    fit 后 score(x) = min_d (x-μ_d)^T Σ_d^{-1} (x-μ_d);
    score <= threshold → 判为源域 (路由 secondU 精修), 否则未见域 (路由 firstU)。

    Args:
        shrinkage: 协方差收缩系数, Σ' = (1-α)Σ + α·(trΣ/D)·I,
            保证正定并提升小样本稳健性 (D=500, val 样本 ~1.9k)
        shared_cov: True = 各域共享池化协方差 (组内方差, 域间均值差不计入);
            False = 每域独立协方差。共享协方差会被域间均值差排主导而撑大
            接受域, 逐域协方差对簇形状更敏感 (对未见域更严格)
    """

    def __init__(self, shrinkage=0.1, shared_cov=True):
        self.shrinkage = shrinkage
        self.shared_cov = shared_cov
        self.domains = None    # (K,) 排序后的局部域 id
        self.means = None      # (K, D)
        self.precision = None  # (D, D) 共享协方差逆 (shared_cov=True)
        self.precisions = None  # (K, D, D) 逐域协方差逆 (shared_cov=False)
        self.threshold = None
        self.lodo_acc = None   # LODO 各折平均门控准确率 (诊断用)

    # ------------------------------------------------------------------ #
    # 拟合与评分
    # ------------------------------------------------------------------ #

    def _shrink(self, cov):
        d = cov.shape[0]
        eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
        return (1.0 - self.shrinkage) * cov \
            + self.shrinkage * cov.diagonal().mean() * eye

    def fit(self, feats, dom_ids):
        """在源域 val 特征上拟合每域均值 + 协方差逆。

        Args:
            feats: (N, D) z_spe GAP 特征 (CPU tensor 即可)
            dom_ids: (N,) 域 id (任意整数 id, 如全局 terrain id)
        """
        feats = feats.detach().cpu().float()
        dom_ids = dom_ids.detach().cpu().long()
        self.domains, _ = torch.unique(dom_ids, return_counts=True)
        local = torch.searchsorted(self.domains, dom_ids)
        groups = [feats[local == k] for k in range(len(self.domains))]
        self.means = torch.stack([g.mean(0) for g in groups])
        centered = [g - self.means[k] for k, g in enumerate(groups)]
        if self.shared_cov:
            # 池化组内协方差: Σ (N_d-1) 加权, 除 (N-K)
            scatter = sum(c.T @ c for c in centered)
            dof = max(sum(len(c) for c in centered) - len(centered), 1)
            self.precision = torch.linalg.inv(self._shrink(scatter / dof))
            self.precisions = None
        else:
            self.precisions = torch.stack([
                torch.linalg.inv(self._shrink(c.T @ c / max(len(c) - 1, 1)))
                for c in centered])
            self.precision = None
        return self

    def score(self, feats):
        """每样本到最近源域中心的马氏距离 (B,), 越大越 OOD。"""
        feats = feats.detach().cpu().float()
        diffs = feats.unsqueeze(1) - self.means.unsqueeze(0)    # (B, K, D)
        if self.shared_cov:
            proj = diffs @ self.precision                        # (B, K, D)
        else:
            # 逐域协方差: (B,K,1,D) @ (K,D,D) → (B,K,D)
            proj = torch.einsum('bkd,kde->bke', diffs, self.precisions)
        dist = torch.einsum('bkd,bkd->bk', proj, diffs)          # (B, K)
        return dist.min(dim=1).values

    def use_second(self, feats):
        """门控决策 (B, bool): True → 源域 (secondU), False → 未见域 (firstU)。"""
        return self.score(feats) <= self.threshold

    # ------------------------------------------------------------------ #
    # LODO 阈值校准
    # ------------------------------------------------------------------ #

    def _fold_best_threshold(self, src_scores, ood_scores):
        """单折: 在候选分位阈值中选门控准确率最高的。"""
        qs = torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99], dtype=torch.float)
        cand = torch.quantile(src_scores, qs)
        cand = torch.cat([cand, src_scores.max().reshape(1)])
        best_t, best_acc = None, -1.0
        for t in cand.tolist():
            acc = ((src_scores <= t).float().mean().item()
                   + (ood_scores > t).float().mean().item()) / 2
            if acc > best_acc:
                best_acc, best_t = acc, t
        return best_t, best_acc

    def calibrate_lodo(self, feats, dom_ids):
        """LODO 阈值校准: 轮流留出一个源域当伪未见域, 各折最优阈值取平均。

        同时记录各折平均门控准确率到 self.lodo_acc (衡量门控信号本身的可分性)。
        """
        feats = feats.detach().cpu().float()
        dom_ids = dom_ids.detach().cpu().long()
        thresholds, accs = [], []
        for held in self._unique_domains(dom_ids):
            m_src, m_ood = dom_ids != held, dom_ids == held
            if m_src.sum() < 2 or m_ood.sum() < 1:
                continue
            sub = MahalanobisGate(self.shrinkage)
            sub.fit(feats[m_src], dom_ids[m_src])
            t, acc = sub._fold_best_threshold(
                sub.score(feats[m_src]), sub.score(feats[m_ood]))
            thresholds.append(t)
            accs.append(acc)
        self.threshold = sum(thresholds) / len(thresholds)
        self.lodo_acc = sum(accs) / len(accs)
        return self

    @staticmethod
    def _unique_domains(dom_ids):
        return torch.unique(dom_ids).tolist()
