# SpectrumDG 项目总览

## 1. 项目简介

SpectrumDG 是一个**频谱地图预测跨域泛化**研究项目，基于 SpectrumNet 数据集（鹏城实验室, IEEE TCCN 2024），探索无线电地图预测模型从城市场景向自然场景的泛化能力。

核心问题：训练于城市地形（Suburban, DenseUrban, OrdinaryUrban）的路径损耗预测模型，如何泛化到未见过的自然地形（Grassland, Desert, Mountainous, Forest）？

## 2. 技术架构

```
输入 (2ch)          firstU (泛化)           secondU (精修)          门控输出
┌──────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────┐
│ buildings│    │ Encoder (layer0-5)│    │ W-Encoder        │    │      │
│          │───→│   + CBAM skips   │───→│   + W-CBAM skips │───→│ Gate │──→ output
│ Tx_prox  │    │   + CIRL(500ch)  │    │   (150ch瓶颈)    │    │      │
└──────────┘    └──────────────────┘    └──────────────────┘    └──────┘
                     ↓ detach               ↑ concat(out1, x, bridge)  ↑
                     ↓                                                   物理先验
              feat_bridge(50ch)                                         (建筑+高程)
```

- **骨干**: RadioWNet — 双 U-Net 级联（14 个 CBAM 模块 + 特征桥接）
- **两阶段训练**: Stage 1 训练 firstU → Stage 2 训练 secondU，通过 `.detach()` 梯度隔离
- **域泛化**: CIRL 因果分解（firstU 瓶颈层）+ Fish 梯度对齐（损失函数层）
- **特征桥接**: firstU 编码器特征 (CIRL 前) 压缩后传入 secondU，解决域信息丢失问题
- **门控推理**: 多物理先验 v4 规则（建筑+高程），推理时自动选择 firstU/secondU 输出

## 3. 代码结构

```
SpectrumDG/
├── configs.py              # 实验配置 + 预设定义
├── train.py                # 训练入口: 两阶段训练 + 自动评估
├── test.py                 # 评估入口: 从 checkpoint 复现评估 (默认多物理先验门控)
│
├── model/                  # 网络模型
│   ├── radiownet.py        #   RadioWNet 双 U-Net 骨干 + 特征桥接
│   ├── attention.py        #   CBAM 注意力模块 (通道×空间)
│   └── causal.py           #   CIRL 因果分解 + GRL
│
├── data_process/           # 数据处理
│   └── dataset.py          #   SpectrumNet 数据集 + DataLoader (支持 terrain_yx)
│
├── lib/                    # 训练/评估工具
│   ├── trainer.py          #   两阶段训练器 (早停 + DG 组合)
│   ├── evaluate.py         #   MAE+MSE+PSNR 评估 + 多物理先验门控
│   └── dg.py               #   DG 组件 (Fish 梯度惩罚, CIRL 辅助损失)
│
├── scripts/                # 工具脚本
│   └── multi_prior_gate.py        # 门控诊断工具 (特征 AUROC + 规则对比)
│
├── run/                    # 实验产物 (模型权重 + 训练日志 + DG 报告)
│   ├── cbam_baseline_4to4/        # CBAM 基线
│   ├── cbam_dg_4to4/              # DG 无门控
│   ├── cbam_dg_gate_4to4/         # 旧版门控 (含 SWAD)
│   ├── cbam_dg_gate_nswad_4to4/   # 无 SWAD 版本
│   └── cbam_dg_gate_bridge_4to4/  # 当前版本 (特征桥接)
│
└── docs/                   # 项目文档
    ├── project_overview.md          # 本文档
    ├── attention_unet_two_stage.md  # 架构详细说明
    ├── improvement_approach.md      # 改进策略说明
    ├── validation_results.md        # 实验验证结果
    └── gate_mechanism.md            # 门控机制设计与验证
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

```
1. 解析配置 (configs.py: 实验预设 + 命令行覆盖)
2. 构建数据 (dataset.py: train/val 85:15 划分 + 跨域测试集)
3. 构建模型 (RadioWNet + 可选 CIRL)
4. Stage 1: firstU 训练 (trainer.py)
   - ERM + 可选 CIRL aux + 可选 Fish penalty
   - 验证集 MSE_firstU 选优 + 早停
5. Stage 2: secondU 训练 (firstU detach, 只训练 W-Net)
   - 同上损失结构, 监督 output2
6. 评估 (evaluate.py)
   - in-domain (val 划分) + cross-domain (多场景)
   - 报告 MAE/MSE/PSNR (firstU + secondU + gated) → dg_report.json
   - 默认使用多物理先验门控 (gate='multi_prior')
```

## 6. 实验预设

| 预设 | CIRL | Fish | 说明 |
|---|---|---|---|
| baseline | - | - | ERM 基线 |
| causal | ✓ | - | 瓶颈层因果分解 |
| fish | - | ✓ | 梯度对齐 |
| fish_causal | ✓ | ✓ | 最优组合 (DG) |
| dg_gate | ✓ | ✓ | DG + 门控 (Stage2 纯 ERM) |

## 7. 关键实验结果

### 7.1 DG 方法对比 (firstU, 3urban→Grassland)

| 方法 | cross NMSE | 改善 |
|---|---|---|
| Baseline | 0.4306 | — |
| **Fish+CIRL** | **0.3387** | **-21.3%** |
| CIRL | 0.3417 | -20.6% |
| Fish | 0.3461 | -19.6% |

### 7.2 最终方案对比 (4to4, CBAM 骨干)

| 版本 | 域内 NMSE ↓ | 跨域 NMSE ↓ | 说明 |
|------|-------------|-------------|------|
| 基线 (无DG无门控) | 0.2209 | 0.8206 | CBAM U-Net, ERM |
| DG 无门控 | 0.2482 | 0.6787 | CBAM + CIRL + Fish |
| **DG + 多先验门控** | **0.2477** | **0.6122** | 建筑+高程组合规则 |

**总体改进**: 跨域泛化提升 **25.4%**，域内损失 12.1%

### 7.3 门控机制

采用多物理先验融合硬路由（详见 [gate_mechanism.md](gate_mechanism.md)）：
- **信号**: 建筑覆盖率 + 地形高程
- **规则**: `bldg_strong OR (elev_low AND bldg_weak)` → secondU，否则 → firstU
- **准确率**: 90.7%（零训练，可解释）

## 8. 用法

```bash
# DG + 门控训练 (推荐)
python train.py --exp dg_gate --save_dir run/cbam_dg_gate_bridge_4to4 \
  --train_terrains 05.Suburban,06.DenseUrban,07.Rural,08.OrdinaryUrban \
  --test_terrains 01.Grassland,09.Desert,10.Mountainous,11.Forest

# 从 checkpoint 复现评估 (自动使用多物理先验门控)
python test.py --run_dir run/cbam_dg_gate_bridge_4to4

# 覆盖测试场景
python test.py --run_dir run/cbam_baseline_4to4 --test_terrain 01.Grassland,09.Desert
```

## 9. 依赖

- Python 3.10+
- PyTorch (CUDA)
- NumPy, PIL, scipy
- 数据集: SpectrumNet/PPData5D-success (Kaggle)
