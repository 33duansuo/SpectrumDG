"""SpectrumDG 实验配置。

实验预设通过 train.py --exp <name> 选择, 命令行开关可覆盖预设。
"""
from dataclasses import dataclass

from data_process.dataset import DEFAULT_DATA_ROOT


@dataclass
class TrainConfig:
    # ---- 数据 ----
    train_terrains: str = "05.Suburban,06.DenseUrban,08.OrdinaryUrban"
    test_terrains: str = "01.Grassland"
    z_layer: str = "00"
    freqs: str = ""              # 逗号分隔 freq 白名单; 空 = 全部
    data_root: str = DEFAULT_DATA_ROOT  # PPData5D-success 根目录
    max_per_terrain: int = None  # 每场景样本上限 (快速验证)
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    # ---- 优化 ----
    num_epochs: int = 100        # 每阶段最大 epoch 数 (早停可能提前结束)
    patience: int = 15           # 验证集 patience 轮无改善则早停
    lr: float = 1e-4
    weight_decay: float = 0.0
    # ---- DG 模块: CIRL 因果分解 (瓶颈层) ----
    use_causal: bool = False
    causal_lambda_dom: float = 1.0   # z_spe 域分类 CE 权重 (应可分)
    causal_lambda_inv: float = 0.1   # z_inv 对抗域分类 CE 权重 (经 GRL)
    causal_lambda_orth: float = 0.1  # z_inv/z_spe 正交解相关权重
    causal_lambda_grl: float = 1.0   # GRL 梯度反转系数
    # ---- DG 模块: Fish 梯度对齐 (损失函数层, 参考 Fishr) ----
    use_fish: bool = False
    fish_lambda: float = 0.1
    # ---- DG 模块: SWAD 权重平均 (训练策略层, 参考 SWAD) ----
    use_swad: bool = False
    swad_start_ratio: float = 0.25   # 开始权重平均的 epoch 占比
    # ---- 两阶段 DG 分工 ----
    dg_second: bool = True           # False = Stage2 (secondU) 纯 ERM, 不挂 causal_2nd/Fish/SWAD
    gate_quantile: float = 0.05      # 已废弃: 门控改为 MahalanobisGate + LODO 自动校准 (保留字段兼容旧 config.json)
    # ---- 输出 ----
    save_dir: str = "run/baseline"
    report: str = ""                 # DG 报告 JSON 路径; 空 = save_dir/dg_report.json


# 实验预设: train.py --exp fish_causal --save_dir run/fish_causal_3urban
EXPERIMENTS = {
    "baseline": {},
    "causal": {"use_causal": True},
    "fish": {"use_fish": True},
    "fish_causal": {"use_causal": True, "use_fish": True},
    "fish_causal_swad": {"use_causal": True, "use_fish": True, "use_swad": True},
    # DG-firstU + 纯ERM-secondU + classifier_spe 置信度门控
    "dg_gate": {"use_causal": True, "use_fish": True, "use_swad": True, "dg_second": False},
}
