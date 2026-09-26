# SpectrumDG: 频谱地图预测跨域泛化

基于 SpectrumNet 数据集的无线电地图预测模型域泛化研究。

## 项目概述

**核心问题**：训练于城市地形（Suburban, DenseUrban, OrdinaryUrban, Rural）的路径损耗预测模型，如何泛化到未见过的自然地形（Grassland, Desert, Mountainous, Forest）？

**解决方案**：
- **双分支架构**：firstU（域泛化粗预测）+ secondU（域内精修 + 特征桥接）
- **域泛化技术**：CIRL 因果分解 + Fish 梯度对齐
- **特征桥接**：firstU 编码器特征 (CIRL 前) 压缩后传入 secondU，解决域信息丢失
- **智能门控**：多物理先验融合（建筑覆盖率 + 地形高程）实现域感知路由

## 主要结果

| 版本 | 域内 NMSE ↓ | 跨域 NMSE ↓ | 说明 |
|------|-------------|-------------|------|
| 基线 (无DG) | 0.2209 | 0.8206 | CBAM U-Net, ERM |
| + DG 正则 | 0.2482 | 0.6787 | CIRL + Fish |
| **+ DG + 门控** | **0.2477** | **0.6122** | 多物理先验门控 |

**关键改进**：
- ✅ DG 正则：跨域泛化提升 **17.3%**（0.8206 → 0.6787）
- ✅ 门控机制：跨域再提升 **9.8%**（0.6787 → 0.6122）
- ✅ 总体：跨域泛化提升 **25.4%**，域内性能基本持平

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
# DG + 门控训练 (推荐)
python train.py --exp dg_gate --save_dir run/cbam_dg_gate_bridge_4to4 \
  --train_terrains 05.Suburban,06.DenseUrban,07.Rural,08.OrdinaryUrban \
  --test_terrains 01.Grassland,09.Desert,10.Mountainous,11.Forest

# 基线训练（无DG）
python train.py --exp baseline --save_dir run/baseline_4to4 \
  --train_terrains 05.Suburban,06.DenseUrban,07.Rural,08.OrdinaryUrban \
  --test_terrains 01.Grassland,09.Desert,10.Mountainous,11.Forest
```

### 评估

```bash
# 标准评估（默认使用多物理先验门控）
python test.py --run_dir run/cbam_dg_gate_bridge_4to4

# 覆盖测试场景
python test.py --run_dir run/cbam_baseline_4to4 --test_terrain 01.Grassland,09.Desert
```

## 架构设计

```
输入 (2ch)          firstU (泛化)           secondU (精修)          门控输出
┌──────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────┐
│ buildings│    │ Encoder (layer0-5)│    │ W-Encoder        │    │      │
│          │───→│   + CBAM skips   │───→│   + W-CBAM skips │───→│ Gate │──→ output
│ Tx_prox  │    │   + CIRL(500ch)  │    │   (150ch瓶颈)    │    │      │
└──────────┘    └──────────────────┘    └──────────────────┘    └──────┘
                     ↓ detach               ↑ concat(out1, x, bridge)  ↑
                     ↓                                                   物理先验
              feat_bridge(50ch)
```

**门控策略**：
- 建筑覆盖率 > 1% OR (低高程 < 100 AND 有建筑痕迹) → secondU
- 否则 → firstU

## 项目结构

```
SpectrumDG/
├── configs.py              # 实验配置 + 预设定义
├── train.py                # 训练入口: 两阶段训练 + 自动评估
├── test.py                 # 评估入口: 从 checkpoint 复现评估
│
├── model/                  # 网络模型
│   ├── radiownet.py        #   RadioWNet 双 U-Net 骨干 + 特征桥接
│   ├── attention.py        #   CBAM 注意力模块
│   └── causal.py           #   CIRL 因果分解 + GRL
│
├── data_process/           # 数据处理
│   └── dataset.py          #   SpectrumNet 数据集 + DataLoader
│
├── lib/                    # 训练/评估工具
│   ├── trainer.py          #   两阶段训练器
│   ├── evaluate.py         #   MAE+MSE+PSNR 评估 + 多物理先验门控
│   └── dg.py               #   DG 组件
│
├── scripts/                # 工具脚本
│   └── multi_prior_gate.py       # 门控诊断工具
│
├── run/                    # 实验产物（模型权重 + 日志）
└── docs/                   # 详细文档
    ├── project_overview.md
    ├── validation_results.md
    ├── gate_mechanism.md
    └── ...
```

## 实验复现

### 主要实验结果

详见 [docs/validation_results.md](docs/validation_results.md)

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
