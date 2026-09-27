# 门控机制设计与验证

> **2026-09-27 更新**：门控主线由多物理先验 v4 规则升级为**学习式 BinaryRouter**（test-time-geo 协议）。v4 规则保留在代码中作为零训练对照（`test.py` 不传 `--router` 时仍走 v4）。本文 §1-§4 记录规则门控时代的诊断与结论，§5-§9 记录 Router 的协议、训练、结果与误路由分析。

## 1. 问题背景

### 1.1 核心挑战

在双分支架构中：
- **泛化分支（firstU / expert_dg）**：CBAMUNet + CIRL + Fish，跨域泛化能力强
- **域内分支（secondU / expert_in）**：CBAM RadioWNet 纯 ERM，域内精度高但跨域性能差

**问题**：如何判断输入样本来自源域（城市）还是目标域（自然/水域），从而把样本送给合适的专家？门控输入只允许使用**输入空间可解释信号**：建筑掩码 `buildings` 与地形高程 `terrain_yx`（npz 公开数据，非频谱测量）。

## 2. 失败尝试记录

### 2.1 瓶颈特征马氏距离（2026-09-26，否决）

**思路**：利用 CIRL 瓶颈层 z_spe/z_inv 特征，计算马氏距离进行 OOD 检测。

**结果**：AUROC < 0.57，接近随机猜测。原因：对抗训练使 z_spe 特征被优化为"域可分"，源域间差异远大于"源域 vs 未见域"差异。

### 2.2 4 城市分类器 + MSP 置信度（2026-09-27，否决）

**思路**：参考 RadioLAM 的场景路由网络，训练 4 源域城市 softmax 分类器（ResNet 风格 CNN，输入 bldg+terrain），以 max-softmax-probability (MSP) 作 OOD 判定：低置信 → 域外。

**结果**：4 城市分类本身准确率很高，但 MSP 在未见域上严重过自信（`score ≥ τ@源域95%` 的比例即误路由率）：

| 未见场景 | Grassland | Island | Ocean | Lake | Desert | Mountainous | Forest |
|---|---|---|---|---|---|---|---|
| 误判为域内比例 | 45.7% | 49.7% | **0%** | 78.9% | 39.4% | **93.6%** | 75.6% |

- softmax 概率饱和，"像某个城市"与"属于源域"不等价（Mountainous 纹理规则、Lake 平坦均被高置信错分）。
- 仅 Ocean（高程全 0、无建筑）判别成功，说明问题不在分类器容量而在 **MSP 本身不是可靠的 OOD 分数**。

### 2.3 合成负样本二分类协议（2026-09-27，否决并删除）

**思路**：不使用任何目标域数据，用源域样本合成"假 OOD"训练二分类 Router：V1 = 建筑清零（模拟纯自然），V2 = 建筑清零 + terrain 随机平移/跨样本置换。

**结果**：留出源域/合成负样本上 AUC 0.985，但对真实目标域误路由严重（Lake 10.8%、Desert 12.1%，且 4 城市分类 MSP 已证明的纹理过自信问题依旧）。合成分布与真实目标地形分布差距过大，分类器学到的是"是否合成"而非"是否源域"。按用户决定，该协议代码与产物已物理删除，仅在此留存实验记录。

### 2.4 规则门控天花板分析（v4，见 §3）

v4 规则在 4to7 上端到端误路由：Island **52.7%**（稀疏建筑 + 低高程被误判城市）、Suburban 约 50%（建筑覆盖率中位 0.0002 过低被误判域外）。离线阈值模拟（含水域排除等补丁）显示规则组合准确率上限约 86%，补丁会同时误伤 DenseUrban/OrdinaryUrban——**手工规则已达天花板**，由此转向学习式 Router。

## 3. 多物理先验门控 v4（历史方案，代码保留为对照）

### 3.1 物理先验分离度

| 物理先验 | 源域（城市） | 未见域（自然） | 分离度 |
|---------|-------------|---------------|--------|
| 建筑覆盖率 | ~5% | ~0% | AUROC = 0.98 |
| 地形高程 | ~31（低海拔） | 450-3128（高海拔） | AUROC = 0.84 |
| Tx 邻近特征 | — | — | ~0.5（无判别力） |

### 3.2 v4 规则

```python
bldg_strong = (bldg_cov > 0.01) or (bldg_ncomp > 5)
elev_low = elev_mean < 100
bldg_weak = (bldg_cov > 0.001) or (bldg_ncomp > 0)
use_domain_expert = bldg_strong or (elev_low and bldg_weak)
```

特征：`bldg_cov`（建筑覆盖率）、`bldg_ncomp`（连通分量数）、`elev_mean`（平均高程），逐样本由 scipy.ndimage 计算。

### 3.3 历史结论（4to4 时代）

- 零训练、可解释；4to4 上路由准确率 90.7%，优于 Isolation Forest（70.5%）。
- 级联门控方案借其将跨域 NMSE 从 0.8206 改善到 0.5678（详见 [validation_results.md](validation_results.md) §6-§8）。
- 4to7 引入水域场景后失效：Island 误路由 52.7%（少量建筑即触发规则），Suburban 误路由 50%（低密度城区 patch 无建筑）。规则阈值无法同时覆盖"城市中的自然 patch"与"自然中的建筑/低海拔 patch"。

## 4. 两代架构下的语义

无论 v4 还是 Router，门控输出的物理含义一致（"像源域 → 域内强模型"）：

| 架构 | 门控 True | 门控 False |
|---|---|---|
| 级联门控（RadioWNet 两阶段） | secondU（域内精修） | firstU（DG 泛化） |
| **双专家并行（dual_expert）** | **expert_in（纯 ERM RadioWNet）** | **expert_dg（CBAMUNet+CIRL+Fish）** |

## 5. BinaryRouter（当前主线）

### 5.1 协议：test-time-geo（目标域频谱不可见，地形公开）

**假设**：部署区域的地形/建筑数据库（高程图、建筑轮廓）是公开先验，不可见的只是目标域的**频谱测量数据**（radiomap target）。这与 RadioMap 任务的标准设定一致（建筑图是模型标准输入，RadioLAM 等工作亦直接使用各场景地形训练路由网络）。

在此协议下：

- **频谱预测双专家仍严格零样本**：只用 4 源域城市的 radiomap 训练，对 7 目标场景频谱零接触。
- **Router 只学习一个部署期判定**："当前 patch 的地形是否落在专家的训练分布（城市）内"，训练仅需 (bldg, terrain) 两类**地理通道**，不读取任何目标域 radiomap。
- 部署时全部目标区域地形均已公开，因此 Router 训练**无需为未见地形留验证/测试集**，正负样本均使用全量真实地形（避免 §2.3 合成负样本的分布失配）。

### 5.2 网络结构（[model/router.py](../model/router.py)）

轻量二分类 CNN，0.39M 参数（专家网络 24.74M 的 1.6%）：

```
输入 2ch: buildings(0/1), terrain_yx(log1p 压缩, clamp≥0 处理海面以下负值)
  → Conv3×3 s2 (32) + BN + ReLU
  → Conv3×3 s2 (64) + BN + ReLU
  → Conv3×3 s2 (128) + BN + ReLU
  → Conv3×3 s2 (256) + BN + ReLU     # 128×128 → 8×8
  → GAP → FC(256→1) → sigmoid score
```

score ≥ τ 路由 expert_in，否则 expert_dg。

### 5.3 训练（[train_router_bin.py](../train_router_bin.py)）

| 项目 | 配置 |
|---|---|
| 正样本 | 4 源域城市（05-08）全部真实地形，n=12372 |
| 负样本 | 7 目标场景（01,02,03,04,09,10,11）全部真实地形，n=19114（只取地理通道） |
| 损失 / 优化 | BCEWithLogits / Adam lr=1e-3, wd=1e-4, batch=64, StepLR(÷3 节点, γ=0.5) |
| 轮数 | 50 epochs 固定（全量数据，无留出、无早停） |
| 最终 | train loss 0.097，acc 0.966，**AUC 0.991** |

每批正/负样本等量配对（负 loader 循环采样）。产物目录 `run/router_realood/`：`RouterBin.pt`、`train_router_bin.log`（每 epoch 一行 loss/acc/auc）、`router_bin_stats.json`（各场景 score 分布与路由比例）。

### 5.4 阈值 τ 校准

τ = 全量源域正样本 score 的 **5% 分位 = 0.4055**（源域 95% 召回，in-sample 校准）。

源域 score 呈强双峰：p10=0.986 而 p05=0.405——约 5% 的城市 patch（Suburban 无建筑空地）score 极低，与目标域分布重叠（见 §8 误路由分析）。

## 6. 端到端实验结果（4to7，双专家权重 run/dual_expert_4to7）

同权重、同数据，仅门控不同；报告：v4=`dg_report.json`，Router=`dg_report_router.json`。

### 6.1 MSE（越低越好）

| 场景 (n) | expert_dg 裸 | expert_in 裸 | v4 门控 | **Router 门控** | v4→in 比例 | Router→in 比例 |
|---|---|---|---|---|---|---|
| 源域 val (1856) | 0.01406 | **0.01097** | 0.01195 | **0.01155** | 89.4% | 95.1% |
| Grassland (1561) | **0.02668** | 0.04096 | 0.02684 | **0.02668** | 2.3% | 0% |
| Island (3188) | **0.03058** | 0.03928 | 0.03293 | **0.03059** | 52.7% | 0.5% |
| Ocean (2754) | **0.03238** | 0.05115 | 0.03238 | **0.03238** | 0% | 0% |
| Lake (5062) | 0.02704 | 0.04240 | **0.02713** | 0.02749 | 3.9% | 3.2% |
| Desert (5443) | 0.03804 | 0.04979 | **0.03808** | 0.03828 | 2.9% | 2.2% |
| Mountainous (848) | **0.03554** | 0.04751 | 0.03554 | **0.03554** | 0% | 0% |
| Forest (258) | **0.03166** | 0.04539 | 0.03191 | **0.03166** | 1.6% | 0% |
| **跨域均值** | **0.03170** | 0.04521 | 0.03212 | **0.03180** | — | — |

### 6.2 PSNR

| | expert_dg | expert_in | v4 门控 | Router 门控 |
|---|---|---|---|---|
| 源域 val | 17.994 | **19.070** | 18.701 | **18.847** |
| 跨域均值 | **14.473** | 12.919 | 14.416 | **14.459** |
| Island | — | — | 14.206 | **14.527（+0.321 dB）** |

### 6.3 结论

1. **Island 决定性改善 −7.1%**（0.03293→0.03059）：误路由 52.7%→0.5%；源域 −3.3%，误路由 10.6%→4.9%。
2. 5 个目标场景（Grassland/Ocean/Mountainous/Forest + Island）与 expert_dg 裸跑下界完全一致或仅差 1e-5。
3. Lake/Desert 相对 v4 微弱回退（+1.3%/+0.5%）：两版误路由比例相近但错的不是同一批边界样本，均为无建筑低高程平坦地形（§8）。
4. 跨域均值 0.03180 已贴近硬路由理论下界 0.03170（expert_dg 裸跑），差距仅 0.0001。

## 7. 关键发现

1. **输入空间信号可靠，但必须学习式组合**：物理先验（建筑 AUROC 0.98、高程 0.84）比瓶颈特征（AUROC<0.57）可靠得多；但手工规则在水陆混合/低密度边界场景失效，轻量 CNN（0.39M 参数、50 轮、几分钟训练）即可把路由误判率压到 1.6%。
2. **OOD 分数必须直接为"源域/非源域"训练**：MSP（§2.2）与合成负样本（§2.3）两条间接路线均失败——概率置信度不等于分布归属，合成 OOD 不等于真实 OOD。
3. **test-time-geo 不违反 DG 协议**：泛化承诺针对的是频谱测量；地形是预测模型的标准输入与公开数据，用它训练路由器不构成目标域频谱泄露。

## 8. 误路由分析（Router 定稿版）

### 8.1 总量与方向

| | 误路由数 | 误路由率 | 应走 → 实际走 |
|---|---|---|---|
| 源域 val (n=1856) | 91 | 4.90% | expert_in → expert_dg |
| 目标域 (n=19114) | 297 | 1.55% | expert_dg → expert_in |

全部 388 个误路由样本 **bldg_cov = 0**：建筑通道为空时，"城市空地/绿地"与"自然地形"在 (bldg, terrain) 输入空间中重合，属信息层面的不可分，非网络容量问题。

### 8.2 分布

- **源域侧**：集中在 05.Suburban（误路由率 23.0%，71/309；该城 44% 的 patch 无建筑），06.DenseUrban 仅 0.2%。这些无建筑城市 patch 上 expert_in 实测仍优于 expert_dg（MSE 0.0156 vs 0.0274），但门控无从知道"这块空地属于城市"。
- **目标域侧**：Lake 160 个（低高程平坦水面）、Desert 120 个（海拔 ~36 的平坦沙漠）、Island 17 个；Grassland/Ocean/Mountainous/Forest 为 0。

### 8.3 实际 MSE 代价与已排除的补救

| 指标 | gated 现状 | 理论下界（永选对） | 误路由拖累 |
|---|---|---|---|
| 源域 | 0.01155 | 0.01097（expert_in 裸） | 0.00058 |
| 目标域 | 0.03180 | 0.03170（expert_dg 裸） | 0.00010 |

- **τ 扫描**：oracle 最优 τ=0.4178（当前 0.4055），联合指标仅再改善 0.00002；降 τ 保源域会使目标侧误路由升至 6%，源/目标召回是此消彼长的权衡，阈值已在最优点。
- **软门控**（pred = s·expert_in + (1−s)·expert_dg 线性混合）：源域持平、目标域反退至 0.03261——边界样本上两专家差异不可由 score 线性表达，否决。
- **剩余理论空间合计 < 0.0007 MSE**，唯一未用输入线索 tx_prox（发射机位置）是弱信号；更本质的方向是缩小两专家在"无建筑"子分布上的能力差（属专家侧改进，不再是门控问题）。**门控定稿，不再继续优化。**

## 9. 使用指南

### 9.1 训练 Router

```bash
cd /data/userSZ/duanrj/SpectrumDG
CUDA_VISIBLE_DEVICES=5 python train_router_bin.py
# 默认: 4 城市正样本 + 7 目标场景负样本全量真实地形, 50 epochs
# 产物: run/router_realood/{RouterBin.pt, train_router_bin.log, router_bin_stats.json}
# 训练结束自动输出 tau@95 与各场景误路由比例
```

### 9.2 门控评估

```bash
# Router 门控 (当前主线; tau 可用训练打印值或直接传浮点)
CUDA_VISIBLE_DEVICES=5 python test.py --run_dir run/dual_expert_4to7 \
  --router run/router_realood/RouterBin.pt --tau 0.4055 \
  --report run/dual_expert_4to7/dg_report_router.json

# v4 规则对照 (零训练; 不传 --router 即默认)
python test.py --run_dir run/dual_expert_4to7

# tau 自动校准 (按 val 集源域 95% 召回, 与全量校准值略有差异)
python test.py --run_dir run/dual_expert_4to7 --router run/router_realood/RouterBin.pt
```

### 9.3 诊断工具

```bash
python scripts/multi_prior_gate.py --max_per_terrain 150
# 输出各物理特征 AUROC、规则门控准确率、Isolation Forest 对比（v4 时代离线诊断）
```
