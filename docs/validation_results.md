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
- **移除**：MixStyle（增益有限且与 CIRL 冲突）、SWAD（无额外增益）

训练配置：100 epoch/阶段，patience=15，两阶段（firstU → secondU），验证集从训练集 85/15 划分。

## 6. DG + 门控最终验证（4to4 实验）

### 6.1 实验设置

在 CBAM 骨干 + DG (Fish+CIRL) 基础上，增加双分支架构 + 多物理先验门控：
- **firstU**：DG 正则化网络（CIRL + Fish），负责跨域泛化
- **secondU**：ERM 精修网络，负责域内精度
- **门控**：建筑覆盖率 + 地形高程组合规则，实现域感知路由

### 6.2 三版本性能对比

| 版本 | 架构 | 域内 NMSE ↓ | 跨域 NMSE ↓ |
|------|------|-------------|-------------|
| **基线** (无DG无门控) | CBAM U-Net, ERM | 0.2209 | 0.8206 |
| **DG 无门控** | CBAM U-Net + DG | 0.2482 | 0.6787 |
| **DG + 单先验门控** | CBAM U-Net + DG + Gate (仅建筑) | 0.2505 | 0.6270 |
| **DG + 多先验门控** | CBAM U-Net + DG + Gate (建筑+高程) | **0.2477** | **0.6122** |

### 6.3 分场景详细对比

| 场景 | 基线 (secondU) | DG (secondU) | DG+单先验门控 (gated) | DG+多先验门控 (gated) | vs 基线 |
|------|----------------|--------------|----------------------|----------------------|---------|
| **域内** | 0.2209 | 0.2482 | 0.2505 | **0.2477** | +12.1% ⬇️ |
| Grassland | 0.5558 | 0.4397 | 0.3754 | **0.3839** | **-31.0%** ⬆️ |
| Desert | 1.0459 | 0.8854 | 0.8471 | **0.7911** | **-24.4%** ⬆️ |
| Mountainous | 1.0295 | 0.8573 | 0.8116 | **0.8395** | **-18.5%** ⬆️ |
| Forest | 0.6513 | 0.5322 | 0.4738 | **0.4345** | **-33.3%** ⬆️ |
| **跨域平均** | 0.8206 | 0.6787 | 0.6270 | **0.6122** | **-25.4%** ⬆️ |

### 6.4 门控路由行为分析

**单先验门控** (仅建筑覆盖率 > 0.01)：
- 域内 gate_second_frac: 0.791（79.1% 路由到 secondU）
- 跨域 gate_second_frac: Grassland 0.5%, Desert 2.2%, Mountainous 0%, Forest 0%

**多先验门控** (建筑 + 高程组合规则)：
- 域内 gate_second_frac: 0.894（89.4% 路由到 secondU）
- 跨域 gate_second_frac: Grassland 2.6%, Desert 3.2%, Mountainous 0%, Forest 0%

**分析**：
- 多先验门控更准确地识别域内样本（+10.3% 路由到 secondU）
- 跨域误路由率仍很低（< 3.2%），泛化性能得到保护
- 高程信号帮助识别"低建筑覆盖但低海拔"的源域样本

### 6.5 改进幅度分析

| 对比 | 域内变化 | 跨域变化 | 说明 |
|------|----------|----------|------|
| 基线 → DG 无门控 | +12.4% ⬇️ | **-17.3%** ⬆️ | DG 正则牺牲域内，换取跨域泛化 |
| DG 无门控 → DG+多先验门控 | -0.2% ≈ | **-9.8%** ⬆️ | 门控进一步提升跨域，域内几乎无损 |
| **基线 → DG+多先验门控** | +12.1% ⬇️ | **-25.4%** ⬆️ | **总体：跨域泛化提升 25%** |

### 6.6 结论

1. **DG 正则**：跨域泛化提升 17%，域内损失 12%——trade-off 合理
2. **多物理先验门控**：在 DG 基础上跨域再提升 10%，域内几乎无损
3. **最终方案**：跨域泛化总计提升 25%，域内损失 12%——核心价值在跨域

## 7. 待完成实验

| 实验 | 说明 | 状态 |
|---|---|---|
| 消融实验 | 固定 CBAM 骨干，逐一开关 CIRL/Fish/桥接/SWAD 的贡献 | 完成（见 §8） |
| 双专家并行架构 | 独立域内专家 + 独立泛化专家联合训练，门控择优 | 完成，4to7 跨域 −28.4%（§8.6） |
| 学习式 Router 门控 | test-time-geo 协议，真实目标地形负样本二分类 | 完成定稿（§9，[gate_mechanism.md](gate_mechanism.md)） |
| 软路由探索 | 按 score 线性混合两专家输出 | 已否决（目标域反退至 0.03261，§9.4） |
| 更多 DG 方法 | Group DRO 等与门控的组合 | 待探索 |

## 8. 2026-09-26 补充消融与分析

### 8.1 口径说明

- 旧报告（§2-§6）的 NMSE 为**逐 batch** `MSE(pred,tgt)/MSE(tgt,0)` 加权平均，训练结束时内嵌评估产生；9 月 26 日起评估流程统一输出 MAE/MSE/PSNR（[lib/evaluate.py](../lib/evaluate.py)），`test.py` 重评口径与旧内嵌评估有几个百分点差异。
- 同日实验之间严格可比，以 **MSE** 为准；括号内给出按各场景固定分母换算的近似 NMSE，仅用于与 §2-§6 表格衔接。
- 全部实验同协议：seed=42，4 城市训练，域内 val n=1856；Grassland n=1561、Desert n=5443、Mountainous n=848、Forest n=258。

### 8.2 SWAD 移除后取得新最优

在 dg_gate 配置上去掉 SWAD、训练预算由 200 轮收敛为 100 轮（patience=15），多物理先验门控重评：

| 配置 | 域内 gated NMSE | 跨域 gated NMSE | Grassland | Desert | Mountainous | Forest |
|---|---|---|---|---|---|---|
| DG+多先验门控（含 SWAD，§6） | 0.2477 | 0.6122 | 0.3839 | 0.7911 | 0.8395 | 0.4345 |
| **DG+多先验门控（去 SWAD）** | 0.2633 | **0.5678** | **0.3674** | **0.7300** | **0.7650** | **0.4086** |

- 去 SWAD 后跨域再降 7.3%，四个自然场景全面改善；域内小幅退化（+6.3%）。
- 与 §4.4 的早期判断一致：CIRL+Fish 双重正则已把解推向平坦区域，SWAD 权重平均无增益，反而因平均到后期过拟合点而略损泛化。
- 产物：[run/cbam_dg_gate_nswad_4to4](../run/cbam_dg_gate_nswad_4to4/dg_report_multi_prior.json)。**该结果目前是级联方案的实际最好成绩**，README/§6 记载的 0.6122 已被刷新。

### 8.3 特征桥接消融：桥接未带来收益（仅 CIRL 对照）

同配置（CIRL only、Stage2 纯 ERM、100 轮）下对比有无 feat_bridge：

| 配置 | 域内 secondU MSE | 域内 gated MSE | 跨域 firstU MSE | 跨域 gated MSE | 跨域 gated≈NMSE |
|---|---|---|---|---|---|
| causal + bridge | 0.01244 | 0.01283 | 0.03939 | 0.03944 | 0.674 |
| **causal + no bridge** | **0.01155** | **0.01218** | **0.03687** | **0.03693** | **0.635** |

- 无桥接在域内 secondU、域内 gated、跨域 firstU/gated 上**全部更优**。"把 CIRL 前特征传给 secondU 以弥补域信息丢失"的假设未被实验支持，桥接特征对纯 ERM 精修网络更像噪声。
- 结论：桥接对最终性能无贡献，后续默认架构（含双专家）均不再使用；代码重构时 `use_bridge`/`feat_bridge` 已物理删除（旧带桥接 checkpoint 不再兼容加载）。

### 8.4 Fish 消融：泛化增益确认，但 secondU 精修分支退化

在 bridge 配置上加入 Fish（CIRL+Fish），与仅 CIRL 对比 firstU（泛化分支）：

| 配置 | 跨域 firstU MSE | ≈NMSE | Grassland | Desert | Mountainous | Forest |
|---|---|---|---|---|---|---|
| causal + bridge | 0.03939 | 0.673 | 0.03539 | 0.04269 | 0.04000 | 0.03950 |
| causal + no bridge | 0.03687 | 0.634 | 0.03118 | 0.04177 | 0.03825 | 0.03627 |
| **Fish+CIRL + bridge** | **0.03355** | **0.580** | **0.02725** | **0.03844** | **0.03609** | **0.03243** |

- Fish 在带桥接配置下仍把跨域 MSE 降低 9%-15%（Grassland -23%），与 §4.2 "前向 CIRL + 反向 Fish 互补"的结论一致，**Fish 确认为必选模块**。
- **但同实验的 secondU 分支退化**：Stage 2 仅 15 轮触顶、epoch 30 早停；域内 secondU MSE=0.01455，反而差于 firstU 的 0.01451，跨域二者也基本持平（0.03388 vs 0.03355）。强 DG firstU 的输出已足够好，secondU 精修无空间，双分支退化为单分支。

### 8.5 阶段性结论与架构转向

1. **模块取舍**：保留 CBAM、CIRL、Fish、多物理先验门控；移除 MixStyle、SWAD；特征桥接消融未见增益，弃用。
2. **级联方案的结构性矛盾**：DG 正则与域内精修被绑在同一条级联链上——firstU 越强（加 Fish），secondU 越没有存在价值；firstU 越弱，域内又保不住。两阶段训练无法解耦这两个目标。
3. **架构转向**：改为**双专家并行**——域内专家（纯 ERM RadioWNet）与泛化专家（CBAM 单 U-Net + CIRL + Fish）各自独立训练，门控只负责择优。设计、训练与推理细节见 [dual_expert.md](dual_expert.md)，代码为 `--exp dual_expert`（[model/dual_expert.py](../model/dual_expert.py)）。

### 8.6 双专家 4to7 实验

测试集扩展为 **7 个未见场景（11 场景全利用）**：在原 4 陆地自然场景（01/09/10/11）基础上新增 3 个水域场景 02.Island(n=3188)、03.Ocean(n=2754)、04.Lake(n=5062)；训练域不变（4 城市，n_train=10516 / n_val=1856）。MSE 按场景分别报告，4to7 总体值与历史 4to4 口径（0.5678）不直接可比。

gated 列与 gate→in 比例为 **BinaryRouter 门控**（2026-09-27 定稿，τ=0.4055；详见 §9 与 [gate_mechanism.md](gate_mechanism.md)）；括号内给出 v4 规则门控值作对照。

| 场景 (n) | expert_in（域内）裸 | expert_dg（泛化）裸 | gated MSE | gate→in 比例 |
|---|---|---|---|---|
| 域内 val (1856) | **0.01097** | 0.01406 | **0.01155**（v4: 0.01195） | 95.1%（v4: 89.4%） |
| Grassland (1561) | 0.04096 | **0.02668** | **0.02668**（v4: 0.02684） | 0%（v4: 2.3%） |
| Island (3188) | 0.03928 | **0.03058** | **0.03059**（v4: 0.03293） | 0.5%（v4: 52.7%） |
| Ocean (2754) | 0.05115 | **0.03238** | **0.03238** | 0% |
| Lake (5062) | 0.04240 | 0.02704 | **0.02749**（v4: 0.02713） | 3.2%（v4: 3.9%） |
| Desert (5443) | 0.04979 | 0.03804 | **0.03828**（v4: 0.03808） | 2.2%（v4: 2.9%） |
| Mountainous (848) | 0.04751 | **0.03554** | **0.03554** | 0% |
| Forest (258) | 0.04539 | **0.03166** | **0.03166**（v4: 0.03191） | 0%（v4: 1.6%） |
| **跨域平均（7 场景）** | 0.04521 | **0.03170** | **0.03180**（v4: 0.03212） | — |
| 陆地 4 场景均值（对齐 4to4 口径） | 0.04591 | **0.03298** | **0.03304** | — |

- 训练已完成，产物 `run/dual_expert_4to7/`：`Trained_Model_DualExpert.pt`（95MB 两专家权重）、`train_loss_dual.log`（4 列 `train_avg_in/train_avg_dg/val_avg_in/val_avg_dg`）、`dg_report.json`（v4 门控）、`dg_report_router.json`（Router 门控）。
- **vs 纯 ERM 基线（同 4to7 协议，`run/cbam_baseline_4to4/dg_report_4to7.json`）**：基线跨域 gated MSE 0.04443、域内 0.01073。双专家+Router 跨域 0.03180（**−28.4%**，7 场景全面改善 −21%~−35%），域内 0.01155（+7.6%，代价主要来自 Suburban 无建筑 patch 的 4.9% 误路由，见 [gate_mechanism.md](gate_mechanism.md) §8）。
- 门控几乎贴着理论下界：跨域 gated 0.03180 vs expert_dg 裸跑 0.03170；域内 0.01155 vs expert_in 裸跑 0.01097。
- 注：2026-09-26 曾启动 4to4 版本（仅 4 陆地测试场景、7 列日志），应用户要求在 epoch 3 后停止并删除产物，统一以 4to7 协议为准。

## 9. 2026-09-27 学习式 Router 门控

### 9.1 动机

v4 规则门控在 4to7 上暴露结构性缺陷：Island 误路由 **52.7%**（稀疏建筑+低高程触发城市规则）、Suburban 约 50% 被误判域外；离线阈值模拟确认手工规则天花板约 86%。详见 [gate_mechanism.md](gate_mechanism.md) §2-§3。

### 9.2 协议与方案

- **test-time-geo 协议**：目标域频谱测量不可见、地形/建筑公开（RadioMap 任务标准设定）。双专家对目标域频谱仍零样本；Router 只用地理通道（bldg, terrain_yx）学习"地形是否落在源域城市分布内"，不读取任何目标域 radiomap。
- **BinaryRouter**（[model/router.py](../model/router.py)）：4 层 stride-2 轻量 CNN（0.39M 参数），源域/目标域二分类。
- **训练**（[train_router_bin.py](../train_router_bin.py)）：正样本 4 城市全量真实地形 n=12372，负样本 7 目标场景全量真实地形 n=19114；BCE + Adam，50 epochs 固定（全量地形公开，无需留出）；train AUC **0.991**。
- **τ 校准**：全量源域 score 5% 分位 = **0.4055**（95% 召回）。

### 9.3 路由与端到端结果

| 指标 | v4 规则 | BinaryRouter |
|---|---|---|
| 源域误路由率 | 10.6% | **4.9%** |
| 目标域误路由率 | Island 52.7% / Lake 3.9% / Desert 2.9% | Island 0.5% / Lake 3.2% / Desert 2.2%，其余 4 场景 0% |
| 域内 gated MSE | 0.01195 | **0.01155（−3.3%）** |
| 跨域 gated MSE 均值 | 0.03212 | **0.03180（−1.0%）** |
| Island gated MSE | 0.03293 | **0.03059（−7.1%）** |

### 9.4 误路由分析与定稿结论

- 全部 388 个误路由样本 bldg_cov=0：城市空地与自然地形在 (bldg, terrain) 输入空间重合，属信息不可分。
- oracle τ 扫描增益仅 0.00002；软门控（线性混合）经验证反损目标域（0.03261），否决。
- 剩余理论空间 < 0.0007 MSE，门控定稿；后续改进点在两专家"无建筑"子分布能力差（专家侧），不在门控。

### 9.5 被否决的中间路线（实验记录）

| 路线 | 结论 |
|---|---|
| 4 城市 softmax 分类器 + MSP 置信度 | 未见域过自信：Mountainous 93.6%、Lake 78.9%、Forest 75.6% 误判域内；仅 Ocean 0% |
| 合成负样本（建筑清零 / terrain 平移置换） | 留出 AUC 0.985 但真实 OOD 误路由 >10%，合成分布失配；代码与产物已按用户决定删除 |
| 软门控线性融合 | 目标域退化至 0.03261，否决 |

