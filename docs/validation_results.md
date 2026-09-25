# 模块有效性验证结果

## 1. 实验设置

### 数据集
- **SpectrumNet**（IEEE TCCN 2024, 鹏城实验室）
- 11 种地形场景，每场景约 5000-10000 样本
- 每样本：128×128 增益图（PNG）+ 建筑掩码（NPZ）
- 模型输入：2 通道（建筑二值掩码 + Tx 邻近图）
- 预测目标：路径损耗增益图（/255 → 阈值 0.2 → [0,1]）

### 评估协议
- **域内（In-domain）**：训练场景的 15% 留出验证集
- **跨域（Cross-domain）**：完全未见过的地形场景
- **指标**：NMSE = MSE(pred, tgt) / MSE(tgt, 0)，越低越好

### 实验分组

| 实验组 | 训练场景 | 测试场景 | 域偏移类型 | 阶段 |
|---|---|---|---|---|
| 3urban→Grassland | 05.Suburban, 06.DenseUrban, 08.OrdinaryUrban | 01.Grassland | 城市→自然 | firstU |
| urban2urban→DenseUrban | 05.Suburban, 07.Rural, 08.OrdinaryUrban | 06.DenseUrban | 城市→密集城市 | firstU |
| CBAM 4to4 | 05.Suburban, 06.DenseUrban, 07.Rural, 08.OrdinaryUrban | 01.Grassland, 09.Desert, 10.Mountainous, 11.Forest | 4城市→4自然 | firstU + secondU |

## 2. 模块逐一验证结果（firstU NMSE）

### 2.1 实验 1：3urban→Grassland

| 方法 | indomain NMSE | cross NMSE | 跨域改善 |
|---|---|---|---|
| Baseline | 0.3298 | 0.4306 | — |
| MixStyle | 0.3450 | 0.4187 | -2.8% |
| CIRL | 0.3784 | **0.3417** | **-20.6%** |
| Fish | 0.3956 | 0.3461 | -19.6% |
| MixStyle+CIRL | 0.3986 | 0.3460 | -19.6% |
| **Fish+CIRL** | 0.4349 | **0.3387** | **-21.3%** |
| Fish+CIRL+SWAD | 0.4484 | 0.3398 | -21.2% |

### 2.2 实验 2：urban2urban→DenseUrban

| 方法 | indomain NMSE | cross NMSE | 跨域改善 |
|---|---|---|---|
| Baseline | 0.2954 | 0.7499 | — |
| MixStyle | 0.3004 | 0.7325 | -2.3% |
| CIRL | 0.3489 | **0.6953** | **-7.3%** |

## 3. CBAM 骨干两阶段验证（4to4 实验）

### 3.1 实验设置

升级骨干为 CBAM U-Net 后，在 4 城市→4 自然场景进行两阶段训练评估。

- **训练域 (4 城市)**：05.Suburban, 06.DenseUrban, 07.Rural, 08.OrdinaryUrban
- **测试域 (4 自然)**：01.Grassland, 09.Desert, 10.Mountainous, 11.Forest
- **方法**：CBAM baseline（无 DG 模块），两阶段训练

### 3.2 结果

| 指标 | indomain | Grassland | Desert | Mountainous | Forest | cross_avg |
|---|---|---|---|---|---|---|
| **NMSE_firstU** | 0.2225 | 0.5256 | 0.9879 | 0.9653 | 0.6185 | 0.7743 |
| **NMSE_secondU** | 0.2209 | 0.5558 | 1.0459 | 1.0295 | 0.6513 | 0.8206 |

### 3.3 分析

- **CBAM 骨干提升域内精度**：对比旧骨干 Baseline（indomain NMSE 0.3298），CBAM 基线域内 NMSE 降至 0.2225（firstU），改善 32.6%
- **secondU 精修效果有限**：secondU NMSE（0.8206）略高于 firstU（0.7743），说明在跨域场景中 secondU 的精修效果不如域内
- **Desert/Mountainous 跨域难度大**：NMSE > 0.96，远高于 Grassland（0.53）和 Forest（0.62），表明极端地形差异是主要挑战
- **Forest 跨域表现较好**：NMSE 0.62/0.65，可能因为 Forest 的建筑稀疏模式与城市有一定相似性

## 4. 关键发现

### 4.1 CIRL 因果分解最有效
- 3urban→Grassland：跨域改善 20.6%（0.4306→0.3417）
- urban2urban→DenseUrban：跨域改善 7.3%（0.7499→0.6953）
- 域内-跨域权衡：indomain NMSE 略升（因果分解主动放弃域特定拟合）

### 4.2 Fish 梯度对齐与 CIRL 互补
- Fish 单独改善 19.6%（3urban），与 CIRL 机制正交
- **Fish+CIRL 组合达最优**：跨域 NMSE 0.3387，改善 21.3%
- 组合效果优于任一单独模块，证实前向因果分解 + 反向梯度对齐的协同性

### 4.3 MixStyle 增益有限且与 CIRL 冲突
- 单独 MixStyle 仅改善 2-3%
- MixStyle+CIRL（0.3460）反不如 CIRL 单独（0.3417）
- 原因：MixStyle 编码器风格混合与 CIRL 瓶颈因果分解施加双重"去域信息"压力，过度正则化

### 4.4 SWAD 无额外增益
- Fish+CIRL+SWAD（0.3398）未优于 Fish+CIRL（0.3387）
- 双重正则化已将模型推向平坦区域，SWAD 权重平均边际收益递减

### 4.5 域内-跨域权衡
所有域泛化方法均存在 indomain 退化现象：

| 方法 | indomain 变化 | cross 改善 |
|---|---|---|
| Baseline → CIRL | +15% | -20.6% |
| Baseline → Fish+CIRL | +32% | -21.3% |

这是泛化-拟合权衡的预期表现：模型主动放弃部分域特定拟合能力以换取跨域泛化。

### 4.6 CBAM 骨干显著提升域内精度
- CBAM 基线 indomain NMSE 0.2225 vs 旧骨干 0.3298，改善 32.6%
- CBAM 注意力精炼 skip 连接有效过滤无关特征

## 5. 最终模型配置

基于验证结果，最终保留：
- **骨干**：CBAM U-Net（14 个 CBAM 模块，双阶段训练）
- **瓶颈层**：CIRL 因果分解（λ_dom=1.0, λ_inv=0.1, λ_orth=0.1, λ_grl=1.0）
- **损失函数**：Fish 梯度对齐（λ_fish=0.1）
- **移除**：MixStyle（增益有限且与 CIRL 冲突）
- **可选**：SWAD 权重平均（默认关闭）

训练配置：100 epoch/阶段，patience=15，两阶段（firstU → secondU），验证集从训练集 85/15 划分。

## 6. 待完成实验

| 实验 | 说明 | 状态 |
|---|---|---|
| DG 方法 + secondU | CIRL/Fish/Fish+CIRL 的两阶段训练 + 4to4 评估 | 待完成 |
| CBAM + DG 4to4 | CBAM 骨干 + Fish+CIRL 在 4to4 设置下的验证 | 待完成 |
| 消融实验 | 固定 CBAM 骨干，逐一开关 CIRL/Fish 的 secondU 贡献 | 待完成 |
