# CBAM 双 U-Net 两阶段训练架构

## 1. 概述

SpectrumDG 的骨干网络为 **RadioWNet**——双 U-Net 级联架构（firstU 粗预测 → secondU 精修）。
skip 连接升级为 **CBAM 注意力精炼**，瓶颈层可插入 **CIRL 因果分解模块**。

两阶段训练通过 `phase` 属性 + `.detach()` 实现梯度隔离：

| 阶段 | firstU 路径 | secondU 路径 | 训练目标 |
|---|---|---|---|
| Stage 1 (firstU) | 有梯度 | 全 detach | 监督 output1 |
| Stage 2 (secondU) | 全 detach | 有梯度 | 监督 output2 |

## 2. 网络结构

### 2.1 firstU（编码器 layer0→layer5 + 解码器 conv_up5→conv_up000）

**编码器**（输入 2 通道: buildings + Tx 邻近图）:

| 层 | 输入通道 | 输出通道 | 卷积核 | 池化 |
|---|---|---|---|---|
| layer00 | 2 | 6 | 3×3, pad=1 | 1×1 |
| layer0 | 6 | 40 | 5×5, pad=2 | 2×2 |
| layer1 | 40 | 50 | 5×5, pad=2 | 2×2 |
| layer10 | 50 | 60 | 5×5, pad=2 | 1×1 |
| layer2 | 60 | 100 | 5×5, pad=2 | 2×2 |
| layer20 | 100 | 100 | 3×3, pad=1 | 1×1 |
| layer3 | 100 | 150 | 5×5, pad=2 | 2×2 |
| layer4 | 150 | 300 | 5×5, pad=2 | 2×2 |
| layer5 | 300 | **500** | 5×5, pad=2 | 2×2 |

**瓶颈层**: layer5 输出 500 通道特征图，可选插入 `CausalDecomposer(channels=500)`。

**解码器** + **CBAM skip 连接**（7 个 CBAM 模块）:

| Skip | CBAM 通道 | 解码器操作 | 拼接方式 |
|---|---|---|---|
| layer4 | CBAM(300) | conv_up5: 500→300 | cat(decoder, cbam(encoder)) |
| layer3 | CBAM(150) | conv_up4: 600→150 | cat(decoder, cbam(encoder)) |
| layer20 | CBAM(100) | conv_up3: 300→100 | cat(decoder, cbam(encoder)) |
| layer2 | CBAM(100) | conv_up20: 200→100 | cat(decoder, cbam(encoder)) |
| layer10 | CBAM(60) | conv_up2: 200→60 | cat(decoder, cbam(encoder)) |
| layer1 | CBAM(50) | conv_up10: 120→50 | cat(decoder, cbam(encoder)) |
| layer0 | CBAM(40) | conv_up1: 100→40 | cat(decoder, cbam(encoder)) |

最终输出: conv_up0(80→20) → cat(layer00, x0) → conv_up00(40+2→20) → cat(x0) → conv_up000(20+2→1) → **output1** (1 通道)

### 2.2 secondU（W-编码器 Wlayer0→Wlayer5 + W-解码器 Wconv_up5→Wconv_up000）

**输入**: cat(output1.detach(), x) = 3 通道（firstU 预测 + buildings + Tx 邻近图）

**W-编码器**:

| 层 | 输入通道 | 输出通道 |
|---|---|---|
| Wlayer00 | 3 | 20 |
| Wlayer0 | 20 | 30 |
| Wlayer1 | 30 | 40 |
| Wlayer10 | 40 | 50 |
| Wlayer2 | 50 | 60 |
| Wlayer20 | 60 | 70 |
| Wlayer3 | 70 | 90 |
| Wlayer4 | 90 | 110 |
| Wlayer5 | 110 | **150** |

**瓶颈层**: Wlayer5 输出 150 通道，可选插入 `CausalDecomposer(channels=150)`。

**W-解码器** + **CBAM skip 连接**（7 个 W-CBAM 模块）: 结构对称，模式同 firstU。

最终输出: **output2** (1 通道)

## 3. CBAM 注意力模块

CBAM（Convolutional Block Attention Module, Woo et al. 2018）= 通道注意力 × 空间注意力。

### 3.1 通道注意力 (ChannelAttention)

```
GAP(x) → MLP → avg
                    } → sigmoid → 通道权重
GMP(x) → MLP → max
```

- GAP + GMP 双路径 pooling → 共享 MLP（降维 reduction=16）→ sigmoid
- 突出对当前任务重要的特征通道

### 3.2 空间注意力 (SpatialAttention)

```
GAP(x) → avg → concat → Conv(2→1, 7×7) → sigmoid → 空间权重
GMP(x) → max ↗
```

- 通道维 GAP + GMP → concat → 7×7 卷积 → sigmoid
- 突出对当前任务重要的空间区域

### 3.3 组合

```python
CBAM(x) = x * ChannelAttention(x) * SpatialAttention(x)
```

相比 AttentionGate（Oktay et al. 2018），CBAM 同时建模通道和空间重要性，且为自注意力模块（无需门控信号），参数更少。

## 4. CIRL 因果分解模块（瓶颈层）

参考 CIRL（Causal Invariant Representation Learning），将 bottleneck 表示分解为:

- **z_inv（域不变/传播物理）** → 解码器预测
- **z_spe（域特定/城市风格）** → 域分类器

### 4.1 组件

| 组件 | 功能 |
|---|---|
| proj_inv (1×1 Conv) | 投影 z_inv，初始化为恒等（训练初期不破坏解码器） |
| proj_spe (1×1 Conv) | 投影 z_spe |
| classifier_spe | GAP → FC → D，域分类（CE 损失，应可分） |
| classifier_inv | GAP → FC → D，经 GRL 的对抗域分类 |
| GRL | 前向恒等，反向梯度乘 -λ（DANN 风格） |

### 4.2 损失

```
L_CIRL = λ_dom · CE(z_spe, domain) + λ_inv · CE(z_inv, domain; GRL) + λ_orth · |⟨z_inv, z_spe⟩|
```

- `λ_dom=1.0`: z_spe 域分类权重（应可分）
- `λ_inv=0.1`: z_inv 对抗域分类权重（经 GRL，使 z_inv 域不可分）
- `λ_orth=0.1`: 正交解相关权重（鼓励 z_inv ⊥ z_spe）
- `λ_grl=1.0`: GRL 梯度反转系数

### 4.3 双瓶颈部署

| U-Net | 瓶颈层 | 通道数 | CausalDecomposer |
|---|---|---|---|
| firstU | layer5 | 500 | `self.causal` |
| secondU | Wlayer5 | 150 | `self.causal_2nd` |

## 5. 两阶段训练流程

```
Stage 1: model.phase = "firstU"
  → firstU 有梯度, secondU 全 detach
  → 监督 output1, 验证选优 NMSE_firstU
  → 保存 Trained_Model_FirstU.pt

Stage 2: model.phase = "secondU"
  → firstU 全 detach, secondU 有梯度
  → 监督 output2, 验证选优 NMSE_secondU
  → 保存 Trained_Model_SecondU.pt (含两阶段全部权重)
```

每阶段均支持:
- **早停**: patience 轮验证集无改善则停止
- **学习率调度**: StepLR, step_size = num_epochs/3, gamma=0.5
- **SWAD 权重平均**（可选）: 自 swad_start_ratio 起累积权重
- **DG 模块组合**: CIRL / Fish / SWAD 可任意组合

## 6. 实验预设系统

通过 `configs.py` 的 `EXPERIMENTS` 字典定义预设，命令行 `--exp` 选择:

| 预设 | CIRL | Fish | SWAD | 说明 |
|---|---|---|---|---|
| baseline | - | - | - | ERM 基线 |
| causal | ✓ | - | - | 瓶颈层因果分解 |
| fish | - | ✓ | - | 梯度对齐 |
| fish_causal | ✓ | ✓ | - | 最优组合 |
| fish_causal_swad | ✓ | ✓ | ✓ | 含权重平均 |

命令行开关（`--use_causal`, `--use_fish`, `--use_swad`）可覆盖预设。

## 7. 文件结构

```
model/
├── __init__.py          # 包导出
├── radiownet.py         # RadioWNet 双 U-Net 骨干
├── attention.py         # CBAM 注意力模块
├── causal.py            # CIRL 因果分解 + GRL
└── physics.py           # 物理先验 (SignalMutationAttention, PhysicsElection)
```
