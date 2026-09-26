"""RadioWNet 骨干网络: 双 U-Net 级联 (firstU 粗预测 → secondU 精修)。

骨干来源于 RadioUNet 项目 (/data/userSZ/duanrj/RadioUNet/lib/modules.py),
skip 连接升级为 CBAM 注意力精炼, 瓶颈层可插入 CausalDecomposer。

两阶段训练通过 phase 属性切换, .detach() 保证梯度隔离 (设计如此, 非 bug):
  phase="firstU":  firstU 有梯度, secondU 全 detach (Stage 1 只训练 firstU)
  phase="secondU": firstU 全 detach, secondU 有梯度 (Stage 2 只训练 W-Net)
"""
import torch
import torch.nn as nn

from .attention import CBAM
from .causal import CausalDecomposer


def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool),
    )


def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True),
    )


class RadioWNet(nn.Module):
    """双 U-Net 级联骨干。

    Args:
        inputs: 输入通道数 (SpectrumDG 场景为 2: buildings + Tx 邻近图)
        phase: "firstU" / "secondU", 两阶段训练的梯度开关
        use_causal: firstU 瓶颈层是否插入 CausalDecomposer
        num_domains: 训练域个数 (仅 use_causal 时用于域分类器输出维)
        causal_lambda_grl: GRL 梯度反转系数
        use_causal_2nd: secondU 瓶颈层是否插入 CausalDecomposer
            (dg_second=False 时为 False → Stage2 纯 ERM 域内精修)
    """

    def __init__(self, inputs=2, phase="firstU",
                 use_causal=False, num_domains=0, causal_lambda_grl=1.0,
                 use_causal_2nd=True):
        super().__init__()

        self.inputs = inputs
        self.phase = phase

        # CIRL 因果分解 (firstU 瓶颈 layer5, 500 通道)。
        # 有学习参数 → 旧 checkpoint 需 strict=False 加载; num_domains = 训练域数。
        self.causal = None
        if use_causal:
            self.causal = CausalDecomposer(channels=500, num_domains=num_domains,
                                           lambda_grl=causal_lambda_grl)

        if inputs <= 3:
            self.layer00 = convrelu(inputs, 6, 3, 1, 1)
            self.layer0 = convrelu(6, 40, 5, 2, 2)
        else:
            self.layer00 = convrelu(inputs, 10, 3, 1, 1)
            self.layer0 = convrelu(10, 40, 5, 2, 2)

        self.layer1 = convrelu(40, 50, 5, 2, 2)
        self.layer10 = convrelu(50, 60, 5, 2, 1)
        self.layer2 = convrelu(60, 100, 5, 2, 2)
        self.layer20 = convrelu(100, 100, 3, 1, 1)
        self.layer3 = convrelu(100, 150, 5, 2, 2)
        self.layer4 = convrelu(150, 300, 5, 2, 2)
        self.layer5 = convrelu(300, 500, 5, 2, 2)

        self.conv_up5 = convreluT(500, 300, 4, 1)
        self.conv_up4 = convreluT(300 + 300, 150, 4, 1)
        self.conv_up3 = convreluT(150 + 150, 100, 4, 1)
        self.conv_up20 = convrelu(100 + 100, 100, 3, 1, 1)
        self.conv_up2 = convreluT(100 + 100, 60, 6, 2)
        self.conv_up10 = convrelu(60 + 60, 50, 5, 2, 1)
        self.conv_up1 = convreluT(50 + 50, 40, 6, 2)
        self.conv_up0 = convreluT(40 + 40, 20, 6, 2)
        if inputs <= 3:
            self.conv_up00 = convrelu(20 + 6 + inputs, 20, 5, 2, 1)
        else:
            self.conv_up00 = convrelu(20 + 10 + inputs, 20, 5, 2, 1)
        self.conv_up000 = convrelu(20 + inputs, 1, 5, 2, 1)

        # 特征桥接: 将 firstU 编码器特征 (CIRL 前) 传递给 secondU
        # 500ch → 50ch (1x1 卷积压缩), 解决 secondU 无法访问域特定信息的问题
        # 上采样到输入尺寸 (128x128) 以匹配 output1 和 x
        self.feat_bridge = nn.Sequential(
            nn.Conv2d(500, 50, kernel_size=1),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
        )

        self.Wlayer00 = convrelu(inputs + 1 + 50, 20, 3, 1, 1)  # 输入: output1(1) + x(2) + bridge(50)
        self.Wlayer0 = convrelu(20, 30, 5, 2, 2)
        self.Wlayer1 = convrelu(30, 40, 5, 2, 2)
        self.Wlayer10 = convrelu(40, 50, 5, 2, 1)
        self.Wlayer2 = convrelu(50, 60, 5, 2, 2)
        self.Wlayer20 = convrelu(60, 70, 3, 1, 1)
        self.Wlayer3 = convrelu(70, 90, 5, 2, 2)
        self.Wlayer4 = convrelu(90, 110, 5, 2, 2)
        self.Wlayer5 = convrelu(110, 150, 5, 2, 2)

        self.Wconv_up5 = convreluT(150, 110, 4, 1)
        self.Wconv_up4 = convreluT(110 + 110, 90, 4, 1)
        self.Wconv_up3 = convreluT(90 + 90, 70, 4, 1)
        self.Wconv_up20 = convrelu(70 + 70, 60, 3, 1, 1)
        self.Wconv_up2 = convreluT(60 + 60, 50, 6, 2)
        self.Wconv_up10 = convrelu(50 + 50, 40, 5, 2, 1)
        self.Wconv_up1 = convreluT(40 + 40, 30, 6, 2)
        self.Wconv_up0 = convreluT(30 + 30, 20, 6, 2)
        # secondU 最终 skip: decoder(20) + wlayer00(20) + winput(1+2+50=53)
        self.Wconv_up00 = convrelu(20 + 20 + (inputs + 1 + 50), 20, 5, 2, 1)
        self.Wconv_up000 = convrelu(20 + (inputs + 1 + 50), 1, 5, 2, 1)

        # CBAM 注意力: firstU skip 连接 (7 个), 编码器特征 → CBAM → cat(decoder, refined)
        self.cbam4 = CBAM(300)
        self.cbam3 = CBAM(150)
        self.cbam20 = CBAM(100)
        self.cbam2 = CBAM(100)
        self.cbam10 = CBAM(60)
        self.cbam1 = CBAM(50)
        self.cbam0 = CBAM(40)
        # secondU skip 连接 (7 个)
        self.Wcbam4 = CBAM(110)
        self.Wcbam3 = CBAM(90)
        self.Wcbam20 = CBAM(70)
        self.Wcbam2 = CBAM(60)
        self.Wcbam10 = CBAM(50)
        self.Wcbam1 = CBAM(40)
        self.Wcbam0 = CBAM(30)

        # CIRL 因果分解 (secondU 瓶颈 Wlayer5, 150 通道);
        # dg_second=False 时不建 → secondU 纯 ERM 域内精修, 旧 checkpoint 加载需 strict=False
        self.causal_2nd = None
        if use_causal and use_causal_2nd:
            self.causal_2nd = CausalDecomposer(channels=150, num_domains=num_domains,
                                               lambda_grl=causal_lambda_grl)

    def forward(self, x):
        x0 = x[:, 0:self.inputs, :, :]

        if self.phase == "firstU":
            output1, feat_bridged = self._forward_firstU(x, x0)
            output2 = self._forward_secondU(x, output1, feat_bridged, detach=True)
        else:
            output1, feat_bridged = self._forward_firstU(x, x0, detach=True)
            output2 = self._forward_secondU(x, output1, feat_bridged, detach=False)

        return [output1, output2]

    def _forward_firstU(self, x, x0, detach=False):
        layer00 = self.layer00(x0)
        layer0 = self.layer0(layer00)
        layer1 = self.layer1(layer0)
        layer10 = self.layer10(layer1)
        layer2 = self.layer2(layer10)
        layer20 = self.layer20(layer2)
        layer3 = self.layer3(layer20)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        
        # 保存 CIRL 前的特征，通过桥接传递给 secondU
        feat_bridged = self.feat_bridge(layer5)
        
        if self.causal is not None:
            # z_inv(域不变) → 解码器; aux 暂存于 self.causal._last_aux
            layer5 = self.causal(layer5)

        layer4u = self.conv_up5(layer5)
        layer4u = torch.cat([layer4u, self.cbam4(layer4)], dim=1)
        layer3u = self.conv_up4(layer4u)
        layer3u = torch.cat([layer3u, self.cbam3(layer3)], dim=1)
        layer20u = self.conv_up3(layer3u)
        layer20u = torch.cat([layer20u, self.cbam20(layer20)], dim=1)
        layer2u = self.conv_up20(layer20u)
        layer2u = torch.cat([layer2u, self.cbam2(layer2)], dim=1)
        layer10u = self.conv_up2(layer2u)
        layer10u = torch.cat([layer10u, self.cbam10(layer10)], dim=1)
        layer1u = self.conv_up10(layer10u)
        layer1u = torch.cat([layer1u, self.cbam1(layer1)], dim=1)
        layer0u = self.conv_up1(layer1u)
        layer0u = torch.cat([layer0u, self.cbam0(layer0)], dim=1)
        layer00u = self.conv_up0(layer0u)
        layer00u = torch.cat([layer00u, layer00], dim=1)
        layer00u = torch.cat([layer00u, x0], dim=1)
        layer000u = self.conv_up00(layer00u)
        layer000u = torch.cat([layer000u, x0], dim=1)
        output1 = self.conv_up000(layer000u)

        if detach:
            output1 = output1.detach()
            feat_bridged = feat_bridged.detach()
        return output1, feat_bridged

    def _forward_secondU(self, x, output1, feat_bridged, detach=False):
        # secondU 输入: output1(1ch) + x(2ch) + feat_bridged(50ch)
        # feat_bridged 包含 firstU 编码器的域特定信息 (CIRL 前), 解决信息丢失问题
        winput = torch.cat([output1, x, feat_bridged], dim=1)
        if detach:
            winput = winput.detach()

        wlayer00 = self.Wlayer00(winput)
        wlayer0 = self.Wlayer0(wlayer00)
        wlayer1 = self.Wlayer1(wlayer0)
        wlayer10 = self.Wlayer10(wlayer1)
        wlayer2 = self.Wlayer2(wlayer10)
        wlayer20 = self.Wlayer20(wlayer2)
        wlayer3 = self.Wlayer3(wlayer20)
        wlayer4 = self.Wlayer4(wlayer3)
        wlayer5 = self.Wlayer5(wlayer4)
        if self.causal_2nd is not None:
            wlayer5 = self.causal_2nd(wlayer5)

        wlayer4u = self.Wconv_up5(wlayer5)
        wlayer4u = torch.cat([wlayer4u, self.Wcbam4(wlayer4)], dim=1)
        wlayer3u = self.Wconv_up4(wlayer4u)
        wlayer3u = torch.cat([wlayer3u, self.Wcbam3(wlayer3)], dim=1)
        wlayer20u = self.Wconv_up3(wlayer3u)
        wlayer20u = torch.cat([wlayer20u, self.Wcbam20(wlayer20)], dim=1)
        wlayer2u = self.Wconv_up20(wlayer20u)
        wlayer2u = torch.cat([wlayer2u, self.Wcbam2(wlayer2)], dim=1)
        wlayer10u = self.Wconv_up2(wlayer2u)
        wlayer10u = torch.cat([wlayer10u, self.Wcbam10(wlayer10)], dim=1)
        wlayer1u = self.Wconv_up10(wlayer10u)
        wlayer1u = torch.cat([wlayer1u, self.Wcbam1(wlayer1)], dim=1)
        wlayer0u = self.Wconv_up1(wlayer1u)
        wlayer0u = torch.cat([wlayer0u, self.Wcbam0(wlayer0)], dim=1)
        wlayer00u = self.Wconv_up0(wlayer0u)
        wlayer00u = torch.cat([wlayer00u, wlayer00], dim=1)
        wlayer00u = torch.cat([wlayer00u, winput], dim=1)
        wlayer000u = self.Wconv_up00(wlayer00u)
        wlayer000u = torch.cat([wlayer000u, winput], dim=1)
        output2 = self.Wconv_up000(wlayer000u)

        if detach:
            output2 = output2.detach()
        return output2
