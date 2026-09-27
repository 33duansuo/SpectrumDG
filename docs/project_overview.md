# SpectrumDG 项目总览

## 1. 项目简介

SpectrumDG 是一个**频谱地图预测跨域泛化**研究项目，基于 SpectrumNet 数据集（鹏城实验室, IEEE TCCN 2024），探索无线电地图预测模型从城市场景向自然场景的泛化能力。

核心问题：训练于城市地形（05.Suburban, 06.DenseUrban, 07.Rural, 08.OrdinaryUrban）的路径损耗预测模型，如何泛化到其余 7 个未见场景（陆地自然 Grassland/Desert/Mountainous/Forest + 水域 Island/Ocean/Lake），实现 11 个场景全部利用？（早期 4to4 实验仅测 4 个陆地自然场景，当前主线协议为 4to7。）

## 2. 技术架构

项目包含两代预测架构与一代学习式门控：

- **级联门控架构（§2.1，已完成验证）**：RadioWNet 双 U-Net 两阶段训练 + DG 正则 + v4 规则门控推理，跨域 NMSE 0.8206 → 0.5678。
- **双专家并行架构（§2.2，当前主线）**：独立域内专家与独立泛化专家联合训练，解决级联方案中 secondU 退化、域内-跨域目标耦合的问题，详见 [dual_expert.md](dual_expert.md)。
- **学习式门控 BinaryRouter（2026-09-27 定稿）**：test-time-geo 协议下用源域/目标域真实地形训练轻量二分类 CNN 做专家路由，替代在水域场景失效的 v4 手工规则（Island 误路由 52.7%→0.5%），详见 [gate_mechanism.md](gate_mechanism.md)。

### 2.1 级联门控架构

```
输入 (2ch)          firstU (泛化)           secondU (精修)          门控输出
┌──────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────┐
│ buildings│    │ Encoder (layer0-5)│    │ W-Encoder        │    │      │
│          │───→│   + CBAM skips   │───→│   + W-CBAM skips │───→│ Gate │──→ output
│ Tx_prox  │    │   + CIRL(500ch)  │    │   (150ch瓶颈)    │    │      │
└──────────┘    └──────────────────┘    └──────────────────┘    └──────┘
                     ↓ detach               ↑ concat(out1, x)        ↑ 物理先验
                                                                     (建筑+高程)
```

- **骨干**: RadioWNet — 双 U-Net 级联（14 个 CBAM 模块）
- **两阶段训练**: Stage 1 训练 firstU → Stage 2 训练 secondU，通过 `.detach()` 梯度隔离
- **域泛化**: CIRL 因果分解（firstU 瓶颈层）+ Fish 梯度对齐（损失函数层）
- **门控推理**: 多物理先验 v4 规则（建筑+高程），推理时自动选择 firstU/secondU 输出

> 历史上曾有 firstU→secondU 的特征桥接（feat_bridge），2026-09-26 消融证明无收益，代码已于重构中物理删除（详见 [validation_results.md](validation_results.md) §8.3）。

### 2.2 双专家并行架构（当前主线）

```
输入 (2ch: buildings + Tx_prox)
   ├──► expert_in:  CBAM RadioWNet（级联双 U-Net，纯 ERM）─────┐
   └──► expert_dg:  CBAM 单 U-Net + CIRL + Fish ──────────────┤
                                                              ▼
                              BinaryRouter(buildings, terrain_yx) → gated 输出
                              （v4 物理规则门控保留为零训练对照）
```

- **单阶段联合训练**：两专家同批前向、独立损失（ERM vs ERM+CIRL+Fish）、独立 best 选优、联合早停
- **目标解耦**：域内专家不经受任何去域正则，泛化专家不承担域内精修，门控只做择优
- 代码入口：`--exp dual_expert`，[model/dual_expert.py](../model/dual_expert.py)，[lib/trainer.py](../lib/trainer.py) `train_dual()`

## 3. 代码结构

```
SpectrumDG/
├── configs.py              # 实验配置 + 预设定义
├── train.py                # 训练入口: 两阶段 / 双专家联合训练 + 自动评估
├── train_router_bin.py     # BinaryRouter 训练入口 (test-time-geo 真实地形二分类)
├── test.py                 # 评估入口: 从 checkpoint 复现评估 (--router 挂学习式门控, 默认 v4)
│
├── model/                  # 网络模型
│   ├── radiownet.py        #   RadioWNet 双 U-Net 骨干; CBAMUNet 单 U-Net 泛化专家
│   ├── dual_expert.py      #   DualExpertNet 双专家并行组合 (expert_in + expert_dg)
│   ├── router.py           #   BinaryRouter 源域/目标域二分类门控 (0.39M 参数)
│   ├── attention.py        #   CBAM 注意力模块 (通道×空间)
│   └── causal.py           #   CIRL 因果分解 + GRL
│
├── data_process/           # 数据处理
│   └── dataset.py          #   SpectrumNet 数据集 + DataLoader (支持 terrain_yx)
│
├── lib/                    # 训练/评估工具
│   ├── trainer.py          #   两阶段训练器 + 双专家联合训练器 (早停 + DG 组合)
│   ├── evaluate.py         #   MAE+MSE+PSNR 评估 + Router/v4 两种门控 + τ 校准
│   └── dg.py               #   DG 组件 (Fish 梯度惩罚, CIRL 辅助损失)
│
├── scripts/                # 工具脚本
│   └── multi_prior_gate.py        # v4 门控诊断工具 (特征 AUROC + 规则对比)
│
├── run/                    # 实验产物 (gitignore; 模型权重 + 训练日志 + DG 报告)
│   ├── cbam_*_4to4/               # 4to4 历史实验 (基线/消融/级联门控, 见 docs 记录)
│   ├── dual_expert_4to7/          # 双专家并行 (4 城市→7 未见场景; dg_report*.json)
│   └── router_realood/            # BinaryRouter (RouterBin.pt + 训练日志 + 路由统计)
│
└── docs/                   # 项目文档
    ├── project_overview.md          # 本文档
    ├── dual_expert.md               # 双专家并行架构设计
    ├── attention_unet_two_stage.md  # 架构详细说明 (级联)
    ├── improvement_approach.md      # 改进策略说明
    ├── validation_results.md        # 实验验证结果 (§8 消融/4to7, §9 Router)
    └── gate_mechanism.md            # 门控机制 (v4 历史 + BinaryRouter 主线)
```

## 4. 数据流

### 4.1 数据组织 (PPData5D-success)

```
png/{terrain}/T{tt}C{c}D{dddd}_n{nn}_f{ff}_ss_z{zz}.png  — 128×128 uint8 增益图
npz/T{tt}C{c}D{dddd}_n{nn}_bdtr.npz  — inBldg_zyx(3,128,128) 二值建筑
```

### 4.2 数据处理流水线

```
PNG 增益图 → /255 → pathloss 阈值(0.2) → [0,1] 目标
                                    ↓ argmax → Tx 位置
NPZ 建筑掩码 → z_layer 切片 → 二值建筑输入
                                    ↓
Tx 位置 → log-distance 邻近图 → 归一化 → Tx 邻近输入

最终: input = [buildings, Tx_prox] (2ch), target = gain (1ch)
```

### 4.3 11 种地形场景 (TERRAIN_ORDER)

| ID | 地形 | 类型 |
|---|---|---|
| 0 | 01.Grassland | 自然 |
| 1 | 02.Island | 自然 |
| 2 | 03.Ocean | 自然 |
| 3 | 04.Lake | 自然 |
| 4 | 05.Suburban | 城市 |
| 5 | 06.DenseUrban | 城市 |
| 6 | 07.Rural | 城市 |
| 7 | 08.OrdinaryUrban | 城市 |
| 8 | 09.Desert | 自然 |
| 9 | 10.Mountainous | 自然 |
| 10 | 11.Forest | 自然 |

## 5. 训练流程

级联架构（`arch="wnet"`，默认）：

```
1. 解析配置 (configs.py: 实验预设 + 命令行覆盖)
2. 构建数据 (dataset.py: train/val 85:15 划分 + 跨域测试集)
3. 构建模型 (RadioWNet + 可选 CIRL)
4. Stage 1: firstU 训练 (trainer.py)
   - ERM + 可选 CIRL aux + 可选 Fish penalty
   - 验证集 MSE_firstU 选优 + 早停
5. Stage 2: secondU 训练 (firstU detach, 只训练 W-Net)
   - 同上损失结构, 监督 output2
6. 门控训练 (train_router_bin.py, 独立于专家)
   - BinaryRouter: 4 城市真实地形为正样本 + 7 目标场景真实地形为负样本
   - 只用地理通道 (bldg, terrain_yx), 不读目标域 radiomap; 全量 50 epoch
7. 评估 (evaluate.py / test.py)
   - in-domain (val 划分) + cross-domain (多场景)
   - 报告 MAE/MSE/PSNR (firstU + secondU + gated) → dg_report*.json
   - --router 挂 BinaryRouter 门控 (主线); 缺省回退 v4 规则门控 (对照)
```

双专家架构（`arch="dual"`，`--exp dual_expert`）：用单阶段联合训练替换步骤 4-5——两专家同批前向、损失独立、各自 best 选优、双方均 patience 轮无改善才早停；步骤 6 的 Router 训练仅需数分钟、可独立复跑；详见 [dual_expert.md](dual_expert.md)。

## 6. 实验预设

`configs.py` 只保留主线预设三个（历史 causal/fish/fish_causal/causal_gate 等消融预设已清理，实验记录见 [validation_results.md](validation_results.md)）：

| 预设 | CIRL | Fish | 说明 |
|---|---|---|---|
| baseline | - | - | ERM 基线 (CBAM RadioWNet 级联两阶段) |
| dg_gate | ✓ | ✓ | 级联门控: firstU=CIRL+Fish, Stage2 纯 ERM, v4 门控推理 |
| **dual_expert** | ✓（仅 dg 专家） | ✓（仅 dg 专家） | **双专家并行联合训练（当前主线）** |

## 7. 关键实验结果

### 7.1 DG 方法对比 (firstU, 3urban→Grassland)

| 方法 | cross NMSE | 改善 |
|---|---|---|
| Baseline | 0.4306 | — |
| **Fish+CIRL** | **0.3387** | **-21.3%** |
| CIRL | 0.3417 | -20.6% |
| Fish | 0.3461 | -19.6% |

### 7.2 级联门控方案对比 (4to4, CBAM 骨干)

| 版本 | 域内 NMSE ↓ | 跨域 NMSE ↓ | 说明 |
|------|-------------|-------------|------|
| 基线 (无DG无门控) | 0.2209 | 0.8206 | CBAM U-Net, ERM |
| DG 无门控 | 0.2482 | 0.6787 | CBAM + CIRL + Fish |
| DG + 多先验门控 (含SWAD) | 0.2477 | 0.6122 | 建筑+高程组合规则 |
| **DG + 多先验门控 (去SWAD)** | 0.2633 | **0.5678** | 级联方案实际最优 |

**总体改进**: 跨域泛化提升 **30.8%**，域内损失 19.2%；最新消融与双专家结果见 [validation_results.md](validation_results.md) §8。

### 7.3 门控机制

当前主线为 **BinaryRouter 学习式硬路由**（详见 [gate_mechanism.md](gate_mechanism.md)）：
- **协议**: test-time-geo——目标域频谱不可见、地形/建筑公开；Router 只用地理通道
- **模型**: 4 层 stride-2 CNN（0.39M 参数），score ≥ τ(0.4055) → 域内专家
- **4to7 路由**: 源域误路由 4.9%，目标域 1.55%（Island 0.5%，Grassland/Ocean/Mountainous/Forest 0%）
- **端到端**: 域内 gated MSE 0.01155，跨域 0.03180（贴近 expert_dg 裸跑下界 0.03170）

v4 物理规则（`bldg_strong OR (elev_low AND bldg_weak)`，建筑覆盖率+地形高程，零训练、4to4 准确率 90.7%）保留为对照，4to7 水域场景下 Island 误路由 52.7% 已被 Router 解决。两代架构下语义一致：True/高 score → 域内分支（secondU/expert_in），否则 → 泛化分支（firstU/expert_dg）。

### 7.4 双专家 4to7 最终结果（MSE，2026-09-27）

| 指标 | 纯 ERM 基线 | 双专家 + Router 门控 | 变化 |
|---|---|---|---|
| 域内 val | 0.01073 | 0.01155 | +7.6% |
| 跨域均值（7 场景） | 0.04443 | **0.03180** | **−28.4%** |
| PSNR 跨域均值 | — | **14.459 dB**（域内 18.847 dB） | — |

逐场景与理论下界分析见 [validation_results.md](validation_results.md) §8.6、§9。

## 8. 用法

```bash
# 1) 双专家并行训练 (当前主线; 默认 4 城市训练 → 7 未见场景测试, 覆盖全部 11 场景)
python train.py --exp dual_expert --save_dir run/dual_expert_4to7

# 2) 训练门控 Router (test-time-geo, 真实地形二分类, 数分钟)
CUDA_VISIBLE_DEVICES=5 python train_router_bin.py

# 3) Router 门控评估 (当前主线)
python test.py --run_dir run/dual_expert_4to7 \
  --router run/router_realood/RouterBin.pt --tau 0.4055

# 3') v4 规则门控对照 (零训练, 不传 --router)
python test.py --run_dir run/dual_expert_4to7

# 级联 DG + 门控训练 (历史方案)
python train.py --exp dg_gate --save_dir run/cbam_dg_gate_nswad_4to4 \
  --train_terrains 05.Suburban,06.DenseUrban,07.Rural,08.OrdinaryUrban \
  --test_terrains 01.Grassland,09.Desert,10.Mountainous,11.Forest
```

## 9. 依赖

- Python 3.10+
- PyTorch (CUDA)
- NumPy, PIL, scipy
- 数据集: SpectrumNet/PPData5D-success (Kaggle)
