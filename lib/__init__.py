"""训练与评估支撑库: 训练器 / DG 组件 / 评估指标。"""
from .dg import SWADAverager, build_domain_map, build_domain_label_tensor, cirl_aux_loss, fish_penalty
from .evaluate import evaluate, evaluate_report
from .trainer import Trainer

__all__ = [
    "SWADAverager",
    "build_domain_map",
    "build_domain_label_tensor",
    "cirl_aux_loss",
    "fish_penalty",
    "evaluate",
    "evaluate_report",
    "Trainer",
]
