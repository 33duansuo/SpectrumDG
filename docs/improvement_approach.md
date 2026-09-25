# RadioMap 跨域泛化改进思路说明

## 1. 研究背景

无线电地图预测（Radio Map Prediction）旨在从环境特征（建筑分布、发射机位置）预测空间路径损耗增益分布。现有方法（如 RadioUNet）在同分布场景下表现良好，但跨地形场景（如城市→草地、郊区→密集城区）泛化能力不足。

本工作以 SpectrumNet 数据集（11 种地形场景，鹏城实验室 IEEE TCCN 2024）为基准，在 RadioWNet（双 U-Net 级联架构）上系统性探索多层域泛化改进策略。

## 2. 多层改进策略

### 2.1 编码器层：MixStyle（已移除）

**思路**：在编码器卷积块之间插入 MixStyle 模块，对 instance 级特征统计量（均值、标准差）做批次内随机置换与凸组合，混合不同地形的"风格"以增强域泛化。

**结果**：跨域 NMSE 改善 2-3%，增益有限。MixStyle 仅靠特征统计量混合难以应对 SpectrumNet 的地形域偏移（城市→草地的建筑分布差异远超风格统计量能捕捉的范围）。

**决策**：移除 MixStyle 模块。

### 2.2 瓶颈层：CIRL 因果分解（保留）

**思路**：参考 CIRL（Causal Invariant Representation Learning），在 U-Net 瓶颈层插入因果分解模块 `CausalDecomposer`，将表示分解为：

- **z_inv（域不变/传播物理）**：通过 GRL（Gradient Reversal Layer）对抗训练使域分类器无法区分域标签，编码与地形无关的电磁波传播物理规律。z_inv 连接解码器，用于预测路径损耗。
- **z_spe（域特定/城市风格）**：通过域分类器 CE 损失显式分类地形类型，编码城市建筑风格等域特定信息。

**正交解相关**：通过最小化 z_inv 与 z_spe 的通道级内积，鼓励两组表示解相关，避免信息冗余。

**总损失**：`L = L_ERM + λ_dom · CE(z_spe) + λ_inv · CE(z_inv, GRL) + λ_orth · |⟨z_inv, z_spe⟩|`

**双瓶颈部署**：CIRL 同时应用于 firstU（layer5, 500 通道）和 secondU（Wlayer5, 150 通道）的瓶颈层。

**结果**：3urban→Grassland 跨域 NMSE 改善 20.6%，urban2urban→DenseUrban 改善 7.3%，显著优于 MixStyle。

### 2.3 损失函数层：Fish 梯度对齐（保留）

**思路**：参考 Fishr（Rame et al. 2022），在 ERM 损失上增加梯度对齐惩罚，约束不同域的梯度方向一致：

```
penalty = Σ_d ||∇_θ L_d - ∇_θ L̄||²
```

其中 `L_d` 为域 d 的 ERM 损失，`L̄` 为各域平均损失。通过 `torch.autograd.grad(create_graph=True)` 计算二阶梯度，使模型学习跨域不变的表示。

**与 CIRL 的正交性**：CIRL 在前向瓶颈层做因果分解，Fish 在反向梯度层做跨域对齐，两者机制正交可叠加。Fish 仅对 ERM loss 计算逐域梯度惩罚，不干扰 CIRL 辅助损失的梯度动态。

**结果**：单独使用 Fish 跨域改善 19.6%（3urban→Grassland），与 CIRL 组合后达 21.3%。

### 2.4 训练策略层：SWAD 权重平均（保留为可选）

**思路**：Seek Weights with Flat loss lAndscape for Domain generalization（Cha et al. 2021），在训练后期对权重做等权平均，寻找平坦极小值。

**结果**：Fish+CIRL+SWAD 跨域 NMSE 0.3398，未优于 Fish+CIRL（0.3387）。双重正则化已将模型推向平坦区域，SWAD 边际收益递减。

**决策**：保留为可选模块（`--use_swad`），默认不启用。

## 3. 骨干网络改进：CBAM 注意力 U-Net

将原始 RadioWNet 的 skip connection 升级为 **CBAM**（Convolutional Block Attention Module, Woo et al. 2018）精炼后的特征：

- **通道注意力**：GAP + GMP → 共享 MLP → sigmoid，突出重要通道
- **空间注意力**：通道维 GAP + GMP → Conv(7×7) → sigmoid，突出重要空间区域

CBAM 为自注意力模块（无需门控信号），在 skip 连接处精炼编码器特征后再与解码器特征拼接。

**部署**：firstU 7 个 CBAM + secondU 7 个 CBAM = 共 14 个 CBAM 模块。

相比 AttentionGate（Oktay et al. 2018），CBAM 同时建模通道和空间重要性，参数更少，且无需额外的门控信号输入。

## 4. 两阶段训练流程

```
Stage 1 (firstU):  model.phase = "firstU"
  → firstU 有梯度, secondU 全 detach (梯度隔离)
  → 监督 output1, 验证集选优 NMSE_firstU
  → 保存 Trained_Model_FirstU.pt

Stage 2 (secondU): model.phase = "secondU"
  → firstU 全 detach, secondU 有梯度 (只训练 W-Net)
  → 监督 output2, 验证集选优 NMSE_secondU
  → 保存 Trained_Model_SecondU.pt (含两阶段全部权重)
```

每阶段均支持早停（patience 轮验证集无改善则停止）、学习率调度（StepLR）、以及 DG 模块的任意组合。

## 5. 实验预设系统

通过 `configs.py` 的 `EXPERIMENTS` 字典定义预设，命令行 `--exp` 选择，命令行开关可覆盖预设：

| 预设 | CIRL | Fish | SWAD | 说明 |
|---|---|---|---|---|
| baseline | - | - | - | ERM 基线 |
| causal | ✓ | - | - | 瓶颈层因果分解 |
| fish | - | ✓ | - | 梯度对齐 |
| fish_causal | ✓ | ✓ | - | 最优组合（推荐） |
| fish_causal_swad | ✓ | ✓ | ✓ | 含权重平均 |

用法示例：
```bash
# 实验预设
python train.py --exp fish_causal --save_dir run/fish_causal_3urban

# 命令行开关覆盖
python train.py --exp baseline --use_causal --causal_lambda_inv 0.05 --save_dir run/custom
```

## 6. 最终架构

**骨干**：CBAM U-Net（双级联 U-Net，14 个 CBAM 模块 + 2 个 CausalDecomposer）

**域泛化模块**：
- 瓶颈层：CIRL 因果分解（firstU layer5 + secondU Wlayer5）
- 损失函数：Fish 梯度对齐（仅对 ERM loss 计算逐域梯度惩罚）

**训练策略**：
- 两阶段训练：Stage 1 监督 firstU（output1），Stage 2 监督 secondU（output2）
- 验证集从训练集 85/15 划分（防止测试集泄露）
- 早停机制：patience=15 轮无改善则停止

**数据划分**（默认配置）：
- 训练域：05.Suburban, 06.DenseUrban, 08.OrdinaryUrban（3 城市场景）
- 测试域：01.Grassland（自然场景）
- 4to4 实验：训练 4 城市 → 测试 4 自然（Grassland, Desert, Mountainous, Forest）

**域标签**：terrain_id（0-10，按 TERRAIN_ORDER 全序），用于 CIRL 域分类与 Fish 逐域梯度计算。
