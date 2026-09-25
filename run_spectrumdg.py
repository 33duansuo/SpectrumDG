"""兼容入口 (已废弃): 训练+评估请使用 train.py, 仅评估请使用 test.py。

保留旧命令行习惯: python run_spectrumdg.py ... 等价于 python train.py ...
"""
from train import main

if __name__ == "__main__":
    main()
