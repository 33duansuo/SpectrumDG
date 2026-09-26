# SpectrumDG: 频谱地图预测跨域泛化

基于 SpectrumNet 数据集的无线电地图预测模型域泛化研究。

## 项目概述

**核心问题**：训练于城市地形（Suburban, DenseUrban, OrdinaryUrban, Rural）的路径损耗预测模型，如何泛化到未见过的自然地形（Grassland, Desert, Mountainous, Forest）？

**解决方案**：
- **双专家并行架构（当前主线，`--exp dual_expert`）**：域内专家（纯 ERM CBAM RadioWNet）+ 泛化专家（CBAM 单 U-Net + CIRL + Fish）单阶段联合训练，保住域内性能的同时不牺牲跨域泛化
- **级联门控架构（已验证）**：firstU（域泛化粗预测）+ secondU（域内精修）两阶段训练
- **域泛化技术**：CIRL 因果分解 + Fish 梯度对齐
- **智能门控**：多物理先验融合（建筑覆盖率 + 地形高程）实现域感知路由，两种架构共用

## 主要结果

级联门控方案（4 城市 → 4 自然场景，详见 [docs/validation_results.md](docs/validation_results.md)）：

| 版本 | 域内 NMSE ↓ | 跨域 NMSE ↓ | 说明 |
|------|-------------|-------------|------|
| 基线 (无DG) | 0.2209 | 0.8206 | CBAM RadioWNet, ERM |
| + DG 正则 | 0.2482 | 0.6787 | CIRL + Fish |
| + DG + 门控 (含SWAD) | 0.2477 | 0.6122 | 多物理先验门控 |
| **+ DG + 门控 (去SWAD)** | 0.2633 | **0.5678** | 级联方案实际最优 |

**关键改进**：
- ✅ DG 正则：跨域泛化提升 17.3%（0.8206 → 0.6787）
- ✅ 门控机制 + 去 SWAD：跨域最终 0.5678（较基线 **-30.8%**）
- ⚠️ 代价：域内 NMSE 0.2209 → 0.2633；双专家架构旨在补回这部分域内损失（[docs/dual_expert.md](docs/dual_expert.md)，4to7 实验进行中）

## 快速开始

### 环境配置

```bash
conda create -n spectrumdg python=3.10
conda activate spectrumdg
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pillow scipy
```

### 数据集

下载 [SpectrumNet/PPData5D-success](https://spectrum-net.github.io/) 数据集，解压后设置环境变量：

```bash
export SPECTRUMNET_ROOT=/path/to/PPData5D-success
```

### 训练

```bash
# 双专家并行训练 (当前主线推荐; 默认 4 城市训练 → 7 未见场景测试, 覆盖全部 11 场景)
python train.py --exp dual_expert --save_dir run/dual_expert_4to7

# 级联 DG + 门控训练
python train.py --exp dg_gate --save_dir run/cbam_dg_gate_nswad_4to4 \
  --train_terrains 05.Suburban,06.DenseUrban,07.Rural,08.OrdinaryUrban \
  --test_terrains 01.Grassland,09.Desert,10.Mountainous,11.Forest

# 基线训练（无DG）
python train.py --exp baseline --save_dir run/baseline_4to4 \
  --train_terrains 05.Suburban,06.DenseUrban,07.Rural,08.OrdinaryUrban \
  --test_terrains 01.Grassland,09.Desert,10.Mountainous,11.Forest
```

### 评估

```bash
# 标准评估（自动识别 dual/wnet 架构，统一使用多物理先验 v4 门控）
python test.py --run_dir run/dual_expert_4to7

# 覆盖测试场景
python test.py --run_dir run/cbam_baseline_4to4 --test_terrain 01.Grassland,09.Desert
```

## 架构设计

项目有两代架构，详见 [docs/dual_expert.md](docs/dual_expert.md) 与 [docs/attention_unet_two_stage.md](docs/attention_unet_two_stage.md)：

**双专家并行（当前主线）**
```
输入 ──► expert_in:  CBAM RadioWNet (纯ERM)          ──┐
     └─► expert_dg:  CBAM 单U-Net + CIRL + Fish      ──┴─► 多物理先验门控 ──► 输出
```

**级联门控（已验证）**
```
输入 (2ch)          firstU (泛化)           secondU (精修)          门控输出
┌──────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────┐
│ buildings│    │ Encoder (layer0-5)│    │ W-Encoder        │    │      │
│          │───→│   + CBAM skips   │───→│   + W-CBAM skips │───→│ Gate │──→ output
│ Tx_prox  │    │   + CIRL(500ch)  │    │   (150ch瓶颈)    │    │      │
└──────────┘    └──────────────────┘    └──────────────────┘    └──────┘
                     ↓ detach               ↑ concat(out1, x)        ↑ 物理先验
```

**门控策略**（两代架构共用，级联下 → secondU/firstU；双专家下 → expert_in/expert_dg）：
- 建筑覆盖率 > 1% OR (低高程 < 100 AND 有建筑痕迹) → 域内分支
- 否则 → 泛化分支

## 项目结构

```
SpectrumDG/
├── configs.py              # 实验配置 + 预设定义
├── train.py                # 训练入口: 两阶段 / 双专家联合训练 + 自动评估
├── test.py                 # 评估入口: 从 checkpoint 复现评估
│
├── model/                  # 网络模型
│   ├── radiownet.py        #   RadioWNet 双 U-Net 骨干; CBAMUNet 单 U-Net 泛化专家
│   ├── dual_expert.py      #   DualExpertNet 双专家并行组合
│   ├── attention.py        #   CBAM 注意力模块
│   └── causal.py           #   CIRL 因果分解 + GRL
│
├── data_process/           # 数据处理
│   └── dataset.py          #   SpectrumNet 数据集 + DataLoader
│
├── lib/                    # 训练/评估工具
│   ├── trainer.py          #   两阶段训练器 + 双专家联合训练器
│   ├── evaluate.py         #   MAE+MSE+PSNR 评估 + 多物理先验门控
│   └── dg.py               #   DG 组件
│
├── scripts/                # 工具脚本
│   └── multi_prior_gate.py       # 门控诊断工具
│
├── run/                    # 实验产物（模型权重 + 日志）
└── docs/                   # 详细文档
    ├── dual_expert.md           # 双专家并行架构（当前主线）
    ├── project_overview.md
    ├── validation_results.md
    ├── gate_mechanism.md
    └── ...
```

## 实验复现

### 主要实验结果

详见 [docs/validation_results.md](docs/validation_results.md)（含 2026-09-26 最新消融 §8）

### 双专家架构设计

详见 [docs/dual_expert.md](docs/dual_expert.md)

### 门控机制验证

详见 [docs/gate_mechanism.md](docs/gate_mechanism.md)

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{spectrumdg2026,
  title={SpectrumDG: 频谱地图预测跨域泛化},
  author={段仁俊},
  year={2026},
  howpublished={\url{https://github.com/33duansuo/SpectrumDG}}
}
```

## 许可证

MIT License

## 联系方式

- 问题反馈：请提交 GitHub Issue
- 邮箱：2270346122@qq.com
