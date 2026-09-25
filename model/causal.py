"""CIRL 因果分解模块 (瓶颈层)。

参考 CIRL (Causal Invariant Representation Learning): 将 bottleneck 表示分解为
  z_inv (域不变/传播物理) → 解码器预测,
  z_spe (域特定/城市风格)  → 域分类器。
通过 GRL 对抗使 z_inv 域不可分, 正交约束解相关 z_inv/z_spe。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """GRL 自定义算子: 前向恒等, 反向梯度乘 -lambda。"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GRL(nn.Module):
    """Gradient Reversal Layer (Ganin & Lempitsky 2015)。
    用于对抗学习: 最小化下游 CE 损失时, 上游特征编码器被推向使域分类器
    损失最大化 (域不可分)。"""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class CausalDecomposer(nn.Module):
    """瓶颈层因果分解。

    注意: proj_inv/proj_spe/分类器 均有学习参数, 会进入 state_dict,
    故 use_causal=True 时加载旧 checkpoint 需 strict=False。测试期
    z_inv=proj_inv(x) 仍作用以保持分布一致, 但跳过分类器, aux=None。
    """

    def __init__(self, channels=500, num_domains=3, lambda_grl=1.0):
        super().__init__()
        # 1x1 conv 投影, 保持通道数方便接解码器
        self.proj_inv = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj_spe = nn.Conv2d(channels, channels, 1, bias=False)
        # proj_inv 初始化为恒等, 训练初期 z_inv ≈ x, 不破坏解码器
        with torch.no_grad():
            self.proj_inv.weight.copy_(
                torch.eye(channels).view(channels, channels, 1, 1))
        # 域分类器: GAP → FC → D
        self.classifier_spe = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, num_domains))
        self.classifier_inv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, num_domains))
        self.grl = GRL(lambda_grl)
        self._last_aux = None  # 训练时暂存 (logits_spe, logits_inv, orth)
        self._last_spe_feat = None  # eval 时暂存 z_spe GAP 特征 (供源域/未见域门控)
        self._last_inv_feat = None  # eval 时暂存 z_inv GAP 特征 (门控诊断/备选信号)

    def forward(self, x):
        z_inv = self.proj_inv(x)   # (B, C, H, W) → 解码器
        z_spe = self.proj_spe(x)
        if self.training:
            logits_spe = self.classifier_spe(z_spe)
            z_inv_rev = self.grl(z_inv)              # GRL: 反向翻转梯度
            logits_inv = self.classifier_inv(z_inv_rev)
            # 通道解相关: GAP 后逐样本内积 → 0 (鼓励 z_inv ⊥ z_spe)
            zi = F.adaptive_avg_pool2d(z_inv, 1).flatten(1)  # (B, C)
            zs = F.adaptive_avg_pool2d(z_spe, 1).flatten(1)  # (B, C)
            orth = (zi * zs).sum(dim=1).abs().mean()
            self._last_aux = (logits_spe, logits_inv, orth)
            self._last_spe_feat = None
            self._last_inv_feat = None
        else:
            # 推理时缓存 z_spe/z_inv GAP 特征 (供门控马氏距离评分, 优于饱和的 softmax 置信度)
            with torch.no_grad():
                self._last_spe_feat = F.adaptive_avg_pool2d(z_spe, 1).flatten(1)  # (B, C)
                self._last_inv_feat = F.adaptive_avg_pool2d(z_inv, 1).flatten(1)  # (B, C)
            self._last_aux = None
        return z_inv
