"""双专家并行架构 (Dual-Expert Parallel Architecture)。

域外泛化专家 (expert_dg): CBAMUNet + CIRL + Fish, 跨域最优
域内精修专家 (expert_in): RadioWNet (firstU+secondU 级联, 纯 ERM), 域内最优

训练: 两个网络同批联合训练 (joint 相位, 独立早停选优)
推理: 多物理先验门控逐样本路由 — 域内样本走 expert_in, 域外样本走 expert_dg
"""
import torch
import torch.nn as nn

from .attention import CBAM  # noqa: F401 (结构一致性)
from .radiownet import CBAMUNet, RadioWNet


class DualExpertNet(nn.Module):
    """并行双专家组合网络。

    forward 返回 [out_in, out_dg]:
        out_in: 域内专家 (RadioWNet) 的 secondU 输出
        out_dg: 泛化专家 (CBAMUNet) 的输出
    评估门控 (lib/evaluate.py) 依据物理先验选择其一。

    Args:
        inputs: 输入通道数
        num_domains: 训练域个数 (CIRL 域分类器输出维)
        causal_lambda_grl: GRL 梯度反转系数
    """

    def __init__(self, inputs=2, num_domains=0, causal_lambda_grl=1.0):
        super().__init__()
        # 域内专家: 级联双 U-Net, 纯 ERM (joint 相位同批训练, 梯度隔离)
        self.expert_in = RadioWNet(inputs=inputs, phase="joint",
                                   use_causal=False, num_domains=0,
                                   use_causal_2nd=False)
        # 泛化专家: 单 U-Net + CIRL (+ Fish 在损失层施加)
        self.expert_dg = CBAMUNet(inputs=inputs,
                                  use_causal=True, num_domains=num_domains,
                                  causal_lambda_grl=causal_lambda_grl)

    def forward(self, x):
        o1, o2 = self.expert_in(x)          # o1: firstU 粗预测, o2: secondU 精修
        out_dg = self.expert_dg(x)           # 泛化专家输出
        # 返回顺序与 RadioWNet 约定一致: [泛化分支, 域内分支]
        # 评估端 (lib/evaluate.py) 的 firstU 键 = 泛化专家, secondU 键 = 域内专家
        return [out_dg, o2]

    @property
    def phase(self):
        return "joint"

    @phase.setter
    def phase(self, value):
        # 兼容外部 phase 赋值 (对双专家无实际作用, 训练始终联合)
        pass
