"""CBAM 注意力模块 (Woo et al. 2018): 通道注意力 × 空间注意力。
用于 RadioWNet 的 skip 连接, 编码器特征经 CBAM 精炼后再与解码器特征拼接。"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """GAP+GMP → 共享 MLP → sigmoid, 突出重要通道。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx = self.mlp(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """通道维 GAP+GMP → concat → conv → sigmoid, 突出重要空间区域。"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: 通道注意力 × 空间注意力。
    自注意力模块 (无需门控信号), 顺序施加于特征图。"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)
