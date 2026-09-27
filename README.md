# SpectrumDG: 频谱地图预测跨域泛化

基于 SpectrumNet 数据集的无线电地图预测模型域泛化研究。

## 项目概述

**核心问题**：训练于 4 类城市地形（Suburban, DenseUrban, OrdinaryUrban, Rural）的路径损耗预测模型，如何泛化到其余 7 个未见场景（Grassland, Island, Ocean, Lake, Desert, Mountainous, Forest）？

**解决方案**：
- **双专家并行架构（当前主线，`--exp dual_expert`）**：域内专家（纯 ERM CBAM RadioWNet）+ 泛化专家（CBAM 单 U-Net + CIRL + Fish）单阶段联合训练，保住域内性能的同时不牺牲跨域泛化
- **学习式门控 BinaryRouter（当前主线）**：test-time-geo 协议（目标域频谱不可见、地形公开）下用源域/目标域真实地形训练轻量二分类 CNN（0.39M 参数）做专家路由；v4 物理规则门控保留为零训练对照
- **级联门控架构（已验证）**：firstU（域泛化粗预测）+ secondU（域内精修）两阶段训练
- **域泛化技术**：CIRL 因果分解 + Fish 梯度对齐

## 主要结果

**双专家 + Router 门控（4to7：4 城市训练 → 7 未见场景测试，MSE 口径）**：

| 指标 | 纯 ERM 基线 | 双专家 + Router | 变化 |
|------|-------------|-----------------|------|
| 域内 val MSE ↓ | 0.01073 | 0.01155 | +7.6%（小幅代价） |
| **跨域 7 场景 MSE ↓** | 0.04443 | **0.03180** | **−28.4%** |
| 跨域 PSNR | — | 14.459 dB（域内 18.847 dB） | — |

门控路由准确率：源域误路由 4.9%，目标域 1.55%（Island 由 v4 规则的 52.7% 降至 0.5%）；跨域 gated 0.03180 已贴近泛化专家裸跑下界 0.03170。逐场景结果见 [docs/validation_results.md](docs/validation_results.md) §8.6。

**历史级联门控方案（4to4：4 城市 → 4 陆地自然场景，旧 NMSE 口径）**：

| 版本 | 域内 NMSE ↓ | 跨域 NMSE ↓ | 说明 |
|------|-------------|-------------|------|
| 基线 (无DG) | 0.2209 | 0.8206 | CBAM RadioWNet, ERM |
| + DG 正则 | 0.2482 | 0.6787 | CIRL + Fish |
| **+ DG + v4 门控 (去SWAD)** | 0.2633 | **0.5678** | 级联方案实际最优（−30.8%） |

> 4to7 与 4to4 场景集、指标口径不同（MSE vs NMSE），两组数字不直接可比。

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
# 1) 双专家并行训练 (当前主线推荐; 默认 4 城市训练 → 7 未见场景测试, 覆盖全部 11 场景)
python train.py --exp dual_expert --save_dir run/dual_expert_4to7

# 2) 训练门控 Router (test-time-geo: 4 城市真实地形为正样本, 7 目标场景真实地形为负样本;
#    只用 bldg+terrain 地理通道, 不读目标域 radiomap; 50 epoch, 单卡数分钟)
CUDA_VISIBLE_DEVICES=5 python train_router_bin.py
#    → run/router_realood/RouterBin.pt, 结束时打印 tau@95

# 历史: 级联 DG + v4 门控训练
python train.py --exp dg_gate --save_dir run/cbam_dg_gate_nswad_4to4 \
  --train_terrains 05.Suburban,06.DenseUrban,07.Rural,08.OrdinaryUrban \
  --test_terrains 01.Grassland,09.Desert,10.Mountainous,11.Forest
```

### 评估

```bash
# Router 学习式门控评估 (当前主线; 自动识别 dual/wnet 架构)
python test.py --run_dir run/dual_expert_4to7 \
  --router run/router_realood/RouterBin.pt --tau 0.4055

# v4 物理规则门控对照 (零训练; 不传 --router 即默认)
python test.py --run_dir run/dual_expert_4to7

# 覆盖测试场景
python test.py --run_dir run/cbam_baseline_4to4 --test_terrain 01.Grassland,02.Island
```

## 架构设计

项目有两代架构，详见 [docs/dual_expert.md](docs/dual_expert.md) 与 [docs/attention_unet_two_stage.md](docs/attention_unet_two_stage.md)：

**双专家并行（当前主线）**
```
输入 ──► expert_in:  CBAM RadioWNet (纯ERM)          ──┐
     └─► expert_dg:  CBAM 单U-Net + CIRL + Fish      ──┴─► BinaryRouter 门控 ──► 输出
                                          (输入: bldg + terrain_yx, v4 规则为对照)
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
- **BinaryRouter（主线）**：轻量 CNN 对 (建筑, 地形高程) 打源域 score，score ≥ τ=0.4055 → 域内分支，否则 → 泛化分支；test-time-geo 协议训练（目标域频谱不可见、地形公开）
- v4 物理规则（对照，零训练）：建筑覆盖率 > 1% OR (低高程 < 100 AND 有建筑痕迹) → 域内分支；4to4 有效但 Island 误路由 52.7%
- 详见 [docs/gate_mechanism.md](docs/gate_mechanism.md)（含 MSP、合成负样本等失败路线与误路由分析）

## 项目结构

```
SpectrumDG/
├── configs.py              # 实验配置 + 预设定义
├── train.py                # 训练入口: 两阶段 / 双专家联合训练 + 自动评估
├── train_router_bin.py     # BinaryRouter 训练入口 (test-time-geo 真实地形二分类)
├── test.py                 # 评估入口: 从 checkpoint 复现评估 (--router 挂学习式门控)
│
├── model/                  # 网络模型
│   ├── radiownet.py        #   RadioWNet 双 U-Net 骨干; CBAMUNet 单 U-Net 泛化专家
│   ├── dual_expert.py      #   DualExpertNet 双专家并行组合
│   ├── router.py           #   BinaryRouter 源域/目标域二分类门控 CNN
│   ├── attention.py        #   CBAM 注意力模块
│   └── causal.py           #   CIRL 因果分解 + GRL
│
├── data_process/           # 数据处理
│   └── dataset.py          #   SpectrumNet 数据集 + DataLoader
│
├── lib/                    # 训练/评估工具
│   ├── trainer.py          #   两阶段训练器 + 双专家联合训练器
│   ├── evaluate.py         #   MAE+MSE+PSNR 评估 + Router/v4 双门控 + τ 校准
│   └── dg.py               #   DG 组件
│
├── scripts/                # 工具脚本
│   └── multi_prior_gate.py       # v4 门控诊断工具 (特征 AUROC + 规则对比)
│
├── run/                    # 实验产物（gitignore；权重 + 日志 + dg_report JSON）
└── docs/                   # 详细文档
    ├── dual_expert.md           # 双专家并行架构（当前主线）
    ├── gate_mechanism.md        # 门控机制（v4 历史 + BinaryRouter 主线 + 失败路线）
    ├── project_overview.md
    ├── validation_results.md    # 全部实验记录（§8 消融/4to7，§9 Router）
    └── ...
```

## 实验复现

### 主要实验结果

详见 [docs/validation_results.md](docs/validation_results.md)（§8 消融与双专家 4to7、§9 Router 门控）

### 双专家架构设计

详见 [docs/dual_expert.md](docs/dual_expert.md)

### 门控机制验证

详见 [docs/gate_mechanism.md](docs/gate_mechanism.md)（v4 规则门控、BinaryRouter、MSP/合成负样本失败记录、误路由分析）

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
