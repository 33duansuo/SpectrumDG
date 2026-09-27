"""门控 Router (test-time-geo 设定)。

源域/目标域二分类 CNN: 输入 (building, terrain_yx) 两通道,
输出 1 个 logit; sigmoid score ≥ tau → 域内 → expert_in (RadioWNet 纯ERM),
否则 → expert_dg (CBAMUNet+CIRL+Fish)。

训练协议 (train_router_bin.py): 目标域频谱测量不可见, 地形/建筑为公开先验;
正样本 = 源域城市真实地形, 负样本 = 目标场景真实地形, 不读取任何 radiomap。
tau 由源域正样本 score 的 5% 分位校准 (95% 召回)。
"""
import math

import torch
import torch.nn as nn

_LOG_ELEV_MAX = math.log1p(4096.0)  # SpectrumNet 高程上界 ~3885, log 压缩动态范围


class BinaryRouter(nn.Module):
    """源域/目标域二分类 Router: 4 层 stride-2 CNN + GAP + fc(1)。

    Args:
        width: 首层通道数 (32 → 64 → 128 → 256)
    """

    def __init__(self, width=32):
        super().__init__()
        chs = [width, width * 2, width * 4, width * 8]
        layers = []
        in_c = 2
        for c in chs:
            layers += [nn.Conv2d(in_c, c, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            in_c = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(chs[-1], 1)

    def forward(self, bldg, terrain_yx):
        """bldg: (B,1,H,W)/(B,H,W) 二值建筑; terrain_yx: 原始高程, 内部 log 归一化。"""
        if bldg.dim() == 3:
            bldg = bldg.unsqueeze(1)
        if terrain_yx.dim() == 3:
            terrain_yx = terrain_yx.unsqueeze(1)
        # 高程含负值 (海平面以下, 最低 ~-41), clamp 到 >=0 再 log1p 避免 NaN
        t = torch.log1p(terrain_yx.float().clamp(min=0)) / _LOG_ELEV_MAX
        x = torch.cat([bldg.float(), t], dim=1)
        x = self.features(x)          # (B,256,8,8) @128 输入
        x = x.mean(dim=(2, 3))        # GAP
        return self.head(x).squeeze(1)  # (B,) logit

    @torch.no_grad()
    def score(self, bldg, terrain_yx):
        return torch.sigmoid(self.forward(bldg, terrain_yx))

    @torch.no_grad()
    def gate_score(self, bldg, terrain_yx):
        """门控统一接口: 越大越像源域 (evaluate.py 调用)。"""
        return self.score(bldg, terrain_yx)
