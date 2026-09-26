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

### 2.4 训练策略层：SWAD 权重平均（已移除）

**思路**：Seek Weights with Flat loss lAndscape for Domain generalization（Cha et al. 2021），在训练后期对权重做等权平均，寻找平坦极小值。

**结果**：Fish+CIRL+SWAD 跨域 NMSE 0.3398，未优于 Fish+CIRL（0.3387）。双重正则化已将模型推向平坦区域，SWAD 边际收益递减。

**决策**：已移除。实验证明 CIRL+Fish 双重正则化已足够，SWAD 无额外增益。

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
  → 监督 output1, 验证集选优 MSE_firstU
  → 保存 Trained_Model_FirstU.pt

Stage 2 (secondU): model.phase = "secondU"
  → firstU 全 detach, secondU 有梯度 (只训练 W-Net)
  → 监督 output2, 验证集选优 MSE_secondU
  → 保存 Trained_Model_SecondU.pt (含两阶段全部权重)
```

每阶段均支持早停（patience 轮验证集无改善则停止）、学习率调度（StepLR）、以及 DG 模块的任意组合。

## 5. 实验预设系统

通过 `configs.py` 的 `EXPERIMENTS` 字典定义预设，命令行 `--exp` 选择，命令行开关可覆盖预设。重构后只保留主线三个预设（历史消融预设 causal/fish/fish_causal 已删除，实验记录见 [validation_results.md](validation_results.md)）：

| 预设 | CIRL | Fish | 说明 |
|---|---|---|---|
| baseline | - | - | ERM 基线 |
| dg_gate | ✓ | ✓ | 级联门控 (firstU=DG, Stage2 纯 ERM) |
| dual_expert | ✓（仅 dg 专家） | ✓（仅 dg 专家） | 双专家并行（当前主线） |

用法示例：
```bash
# 实验预设
python train.py --exp dg_gate --save_dir run/dg_gate_4to7

# 命令行开关覆盖
python train.py --exp baseline --use_causal --causal_lambda_inv 0.05 --save_dir run/custom
```

## 6. 最终架构

> 2026-09-26 更新：消融实验（详见 [validation_results.md](validation_results.md) §8）确认 Fish 必选、SWAD 与特征桥接均无增益；级联两阶段还暴露出"强 DG firstU 使 secondU 退化"的结构性矛盾。主线已转向**双专家并行架构**（[dual_expert.md](dual_expert.md)），本节保留级联方案的最终配置作为对照。

**级联门控方案（已完成验证，跨域 NMSE 0.5678）**：

**骨干**：CBAM U-Net（双级联 U-Net，14 个 CBAM 模块 + 1 个 CausalDecomposer）

**域泛化模块**：
- 瓶颈层：CIRL 因果分解（仅 firstU layer5，secondU 纯 ERM）
- 损失函数：Fish 梯度对齐（仅对 ERM loss 计算逐域梯度惩罚）

**特征桥接**：~~firstU 编码器 layer5 (CIRL 前) → 1×1 卷积压缩 (500→50ch) + 上采样 → 传入 secondU~~ —— **消融证明无收益（bridge 在域内/跨域全部略差于 no-bridge），重构时代码已物理删除（`use_bridge`/`feat_bridge` 不再存在）**。

**门控推理**：
- 多物理先验 v4 规则：建筑覆盖率 + 地形高程组合
- 推理时自动判断域内/跨域，选择 secondU/firstU 输出

**训练策略**：
- 两阶段训练：Stage 1 监督 firstU（output1），Stage 2 监督 secondU（output2）
- 验证集从训练集 85/15 划分（防止测试集泄露）
- 早停机制：patience=15 轮无改善则停止
- 100 epoch/阶段（原 200 轮 + SWAD 已证明多余）

**双专家并行方案（当前主线，`--exp dual_expert`）**：
- expert_in：纯 ERM CBAM RadioWNet（无桥接），专职域内
- expert_dg：CBAM 单 U-Net + CIRL + Fish，专职跨域
- 单阶段联合训练，多物理先验门控推理择优

**数据划分**（默认配置，11 场景全利用）：
- 训练域（4 城市）：05.Suburban, 06.DenseUrban, 07.Rural, 08.OrdinaryUrban
- 跨域测试域（7 未见场景）：01.Grassland, 02.Island, 03.Ocean, 04.Lake, 09.Desert, 10.Mountainous, 11.Forest（4 陆地自然 + 3 水域）
- 历史 4to4 实验仅测 4 陆地自然（Grassland, Desert, Mountainous, Forest），结果存档不与 4to7 混比

**域标签**：terrain_id（0-10，按 TERRAIN_ORDER 全序），用于 CIRL 域分类与 Fish 逐域梯度计算。
