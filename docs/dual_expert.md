# 双专家并行架构（Dual-Expert Parallel Architecture）

## 1. 提出背景

此前的主线方案是 RadioWNet **级联双 U-Net + 两阶段训练**：firstU 施加 DG（CIRL+Fish）负责跨域，secondU 纯 ERM 负责域内，推理时用多物理先验门控在二者间硬路由。该方案将跨域 NMSE 从 0.8206 改善到 0.5678，但 2026-09-26 的消融实验暴露出级联/两阶段方案的三个结构性问题：

1. **secondU 域内分支在强 DG 正则下退化**：Fish+CIRL+Bridge 实验中 Stage 2 仅 15 轮就触顶、30 轮早停，secondU 域内 MSE 0.01455 甚至差于 firstU 的 0.01451，精修分支失去存在价值，双分支退化为单分支。
2. **CIRL 主动牺牲域内拟合**：因果分解的设计目标就是放弃域特定信息，同一个 firstU 无法同时做到域内最优与跨域最优——这是"单网络戴两顶帽子"的内在矛盾，不是超参能调和的。
3. **误路由代价不对称**：门控约 10% 的域内样本（Suburban 物理特征接近自然域）会被误路由到 DG 分支，而 DG 网络域内 NMSE 约 0.30，比 ERM 网络（0.22）差，导致 gated 输出的域内性能被拖累。

核心思路转变：**不再要求一个网络同时优化两个目标，而是并行训练两个目标单一的专家网络，门控只需"择优"而非"补救"**。

## 2. 架构总览

```
                         输入 (2ch: buildings + Tx_prox)
                                  │
                ┌─────────────────┴──────────────────┐
                ▼                                     ▼
   expert_in（域内专家）                    expert_dg（泛化专家）
   CBAM RadioWNet 级联双 U-Net             CBAM 单 U-Net (firstU 同构)
   纯 ERM，无 CIRL/Fish/桥接               CIRL@500ch 瓶颈因果分解
   不做任何去域信息正则                     + Fish 逐域梯度对齐
                │                                     │
            output_in (output2)                  output_dg
                └─────────────────┬──────────────────┘
                                  ▼
                  BinaryRouter 门控（学习式，当前主线）
              score(bldg, terrain) ≥ τ → output_in，否则 output_dg
              （v4 物理规则门控保留为零训练对照，见 gate_mechanism.md）
```

- **expert_in**：[RadioWNet](../model/radiownet.py)（firstU+secondU 级联，14 个 CBAM 模块），**纯 ERM**，结构与训练目标完全对齐 CBAM 基线（无桥接、无 CIRL），域内 NMSE 目标复现基线 0.2209 水平。
- **expert_dg**：[CBAMUNet](../model/radiownet.py)（单 U-Net，7 个 CBAM 模块 + layer5 瓶颈 `CausalDecomposer(500ch)`），损失 = ERM + CIRL 辅助损失 + Fish 梯度惩罚，跨域能力对齐此前最优 firstU。
- 两个专家**输入相同、权重独立、无共享**，由 [DualExpertNet](../model/dual_expert.py) 组合。

## 3. 联合训练（单阶段）

与级联方案的两阶段 detach 训练不同，双专家在**同一个训练阶段、同一个 batch、同一个优化器**中同时更新：

```
L = MSE(o1_in, t) + MSE(o2_in, t)          # 域内专家: firstU+secondU 都受 ERM 监督
  + MSE(o_dg,   t)                          # 泛化专家: ERM
  + λ_dom·CE_spe + λ_inv·CE_inv(GRL) + λ_orth·orth   # 仅作用于 expert_dg
  + λ_fish·Σ_d ||∇ L_d - ∇ L̄||²                      # 仅对 o_dg 的 ERM 逐域二阶梯度
```

实现要点（[lib/trainer.py](../lib/trainer.py) `train_dual`）：

1. **级联内部梯度隔离**：expert_in 内部 secondU 的输入为 `output1.detach()`（`phase="joint"`），两个 U-Net 同批更新但不互相回传，避免精修网络的强梯度污染粗预测。
2. **专家间损失独立**：CIRL 辅助损失只从 `expert_dg.causal` 取缓存；Fish 惩罚只对 `o_dg` 计算（expert_in 参数梯度为 None 被自动过滤）。
3. **独立选优、联合早停**：每个 epoch 在 val 上分别记录 `val_mse_in`（secondU 输出）与 `val_mse_dg`，各自保存 best 权重；**仅当两个专家都连续 patience=15 轮无改善才停止**，恢复时各取各的 best，互不拖累。
4. **日志**：`run/<exp>/train_loss_dual.log` 记录 `train_erm_in / train_erm_dg / train_dom / train_inv / train_orth / val_mse_in / val_mse_dg`，每轮一行。

超参与最新主线一致：100 epoch 上限、Adam lr=1e-4、StepLR(÷3 节点, γ=0.5)、batch=32、λ_dom=1.0、λ_inv=0.1、λ_orth=0.1、λ_fish=0.1、val 从训练集 85/15 划分。

## 4. 推理路由

门控为独立训练的 **BinaryRouter**（2026-09-27 起为主线；协议、训练与分析详见 [gate_mechanism.md](gate_mechanism.md) §5、§8）：

- 轻量二分类 CNN（0.39M 参数），输入 (bldg, terrain_yx)，输出源域 score；score ≥ τ(=0.4055) → expert_in，否则 → expert_dg。
- test-time-geo 协议：目标域频谱不可见、地形公开；Router 以 4 城市真实地形为正样本、7 目标场景真实地形为负样本训练，双专家本身仍对目标域频谱零样本。
- v4 物理规则门控（`bldg_strong OR (elev_low AND bldg_weak)`）保留在 [lib/evaluate.py](../lib/evaluate.py) 中作为零训练对照（`test.py` 不传 `--router` 时使用），4to7 上 Island 误路由 52.7%，已被 Router 替代。

评估输出键沿用 firstU/secondU/gated 命名，含义映射为：`firstU = expert_dg`，`secondU = expert_in`，报告 JSON 增加 `"arch": "dual"` 与 `"gate": {"type": "router", "tau": ...}` 标记。

## 5. 实测性质（4to7，Router 门控）

1. **域内基本保住**：gated 域内 MSE 0.01155（expert_in 裸跑 0.01097，expert_dg 裸跑 0.01406）；4.9% 误路由全部来自 Suburban 无建筑 patch，代价有限。
2. **跨域接近泛化专家下界**：gated 跨域均值 0.03180，expert_dg 裸跑 0.03170；5 个场景与裸跑完全一致，Lake/Desert 仅差 0.0002-0.0004。
3. **误路由代价对称化兑现**：expert_in 跨域 MSE 0.04521（级联 secondU 时代 0.75-1.09 NMSE 的塌陷不再出现），Router 把目标域误路由率压到 1.55%，Island 由 v4 的 52.7% 降到 0.5%。
4. **训练预算可控**：单阶段联合训练（参数量 24.74M），一次早停完成；Router 训练另需 50 epoch（数分钟，0.39M 参数）。

## 6. 实验设置（4to7，11 场景全利用）

- **训练域（4 城市，n_train=10516 / n_val=1856）**：05.Suburban, 06.DenseUrban, 07.Rural, 08.OrdinaryUrban
- **跨域测试（7 未见场景，n=19114）**：

| 类别 | 场景 | z00 样本数 | Router 实测路由到 expert_in |
|---|---|---|---|
| 陆地自然（原 4to4） | 01.Grassland | 1561 | 0% |
| 陆地自然（原 4to4） | 09.Desert | 5443 | 2.2% |
| 陆地自然（原 4to4） | 10.Mountainous | 848 | 0% |
| 陆地自然（原 4to4） | 11.Forest | 258 | 0% |
| 水域（新增） | 02.Island | 3188 | 0.5%（v4 规则曾误路由 52.7%） |
| 水域（新增） | 03.Ocean | 2754 | 0%（高程全 0、无建筑） |
| 水域（新增） | 04.Lake | 5062 | 3.2%（低海拔平坦水面边界样本） |

> 口径注意：新增 3 个水域场景与城市传播机制差异极大，4to7 的总体跨域 MSE 与历史 4to4 NMSE（0.5678）不直接可比；报告按场景分别列出，对比时以同协议、同场景集合为准。

## 7. 用法

```bash
# 1) 双专家训练（默认配置即 4 城市 → 7 未见场景；train_terrains/test_terrains 均可显式覆盖）
CUDA_VISIBLE_DEVICES=4 python train.py --exp dual_expert \
  --save_dir run/dual_expert_4to7

# 2) 训练门控 Router（test-time-geo: 源域/目标域真实地形二分类, 仅数分钟）
CUDA_VISIBLE_DEVICES=5 python train_router_bin.py
#    → run/router_realood/RouterBin.pt, 训练结束打印 tau@95 (=0.4055)

# 3) Router 门控评估（当前主线）
python test.py --run_dir run/dual_expert_4to7 \
  --router run/router_realood/RouterBin.pt --tau 0.4055 \
  --report run/dual_expert_4to7/dg_report_router.json

# 3') v4 规则门控对照（零训练，不传 --router）
python test.py --run_dir run/dual_expert_4to7
```

产物：`Trained_Model_DualExpert.pt`（两专家权重）、`checkpoint_dual.pt`（续训用）、`train_loss_dual.log`（每 epoch 一行 4 列：`train_avg_in / train_avg_dg / val_avg_in / val_avg_dg`；CIRL 的 dom/inv/orth 辅助损失仅在控制台打印用于诊断）、`dg_report.json`（v4 门控）与 `dg_report_router.json`（Router 门控，含两专家及 gated 三套 MAE/MSE/PSNR 与门控路由比例）。

## 8. 与级联门控方案的对照

| 维度 | 级联门控（cbam_dg_gate_nswad） | 双专家并行（dual_expert） |
|---|---|---|
| 域内分支 | secondU，依赖 firstU detach 输入，Stage 2 单独训练 | 独立完整 RadioWNet，联合训练，不受 DG 分支影响 |
| 泛化分支 | firstU（级联第一级） | 独立 CBAMUNet |
| 训练流程 | 两阶段串行，各自早停 | 单阶段联合，独立选优+联合早停 |
| DG 正则施加 | firstU | 仅 expert_dg |
| 误路由到的网络强度 | secondU 跨域弱（0.75-1.09） | 两方向均为专职强专家（expert_in 跨域 MSE 0.0452） |
| 门控 | v4 物理规则（4to4 90.7% 准确） | **BinaryRouter 学习式**（4to7 源域误路由 4.9%/目标 1.55%；v4 保留对照） |
| 域内 best（4to7 MSE） | — | 0.01097（expert_in 裸）/ 0.01155（gated） |
| 跨域 best（4to7 MSE） | — | 0.03170（expert_dg 裸）/ **0.03180（gated）**，vs 纯 ERM 基线 0.04443 **−28.4%** |

> 4to7 与 4to4 场景集/指标口径不同（MSE vs 旧 NMSE），不与级联 0.2633/0.5678 直接比大小。
