"""BinaryRouter 训练: 源域 vs 目标域二分类 (test-time-geo 设定, 全量地形训练)。

协议假设: 目标域的频谱测量 (radiomap) 不可见, 但地形/建筑信息是公开先验,
可用于门控训练。正样本 = 4 源域城市 (05-08) 全部真实 (bldg, terrain);
负样本 = 7 目标场景 (01,02,03,04,09,10,11) 全部真实 (bldg, terrain),
只取地理通道, 不读取任何目标域 radiomap target。

门控部署时所有目标区域地形均公开可得, 无需为泛化评估留出未见数据,
因此正负样本均使用全量数据, 固定训练 num_epochs 轮 (无验证早停)。

频谱预测双专家仍只用源域训练, 对目标域频谱零样本; Router 仅学习
"部署环境是否落在专家训练分布内"。

阈值 tau = 全量源域正样本 score 的 5% 分位 (95% 召回, in-sample 校准)。

用法:
  cd /data/userSZ/duanrj/SpectrumDG
  CUDA_VISIBLE_DEVICES=5 python train_router_bin.py
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from data_process.dataset import DEFAULT_DATA_ROOT, SpectrumNetDataset
from model.router import BinaryRouter


def parse_args():
    p = argparse.ArgumentParser(description="SpectrumDG: BinaryRouter 全量源域/目标域二分类训练")
    p.add_argument("--train_terrains", type=str,
                   default="05.Suburban,06.DenseUrban,07.Rural,08.OrdinaryUrban")
    p.add_argument("--test_terrains", type=str,
                   default="01.Grassland,02.Island,03.Ocean,04.Lake,"
                           "09.Desert,10.Mountainous,11.Forest")
    p.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--z_layer", type=str, default="00")
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--save_dir", type=str, default="run/router_realood")
    return p.parse_args()


def _auc(pos_scores, neg_scores):
    """Mann-Whitney AUC。"""
    s = np.concatenate([pos_scores, neg_scores])
    ranks = rankdata(s)
    n_p, n_n = len(pos_scores), len(neg_scores)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def _as_b4(t):
    """(B,H,W) → (B,1,H,W), 与 bldg 通道对齐。"""
    return t.unsqueeze(1) if t.dim() == 3 else t


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def run_epoch(router, pos_loader, neg_iter, device, crit, opt):
    """每批全量正样本 (源域) + 等量负样本 (目标域真实地形), 返回 (loss, acc, auc)。"""
    router.train()
    loss_sum, n = 0.0, 0
    pos_sc, neg_sc = [], []
    for inputs, _, _, terrain_yx in pos_loader:
        pb = inputs[:, 0:1].to(device)
        pt = _as_b4(terrain_yx.to(device))
        ni, _, _, nt_yx = next(neg_iter)
        nb = ni[:, 0:1].to(device)
        nt = _as_b4(nt_yx.to(device))
        m = min(pb.size(0), nb.size(0))
        pb, pt, nb, nt = pb[:m], pt[:m], nb[:m], nt[:m]
        logits = router(torch.cat([pb, nb], 0), torch.cat([pt, nt], 0))
        labels = torch.cat([torch.ones(m, device=device),
                            torch.zeros(m, device=device)])
        loss = crit(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            sc = torch.sigmoid(logits)
            pos_sc.append(sc[:m].cpu().numpy())
            neg_sc.append(sc[m:].cpu().numpy())
            loss_sum += loss.item() * (2 * m)
            n += 2 * m
    pos_sc, neg_sc = np.concatenate(pos_sc), np.concatenate(neg_sc)
    acc = float(np.concatenate([pos_sc >= 0.5, neg_sc < 0.5]).mean())
    return loss_sum / n, acc, _auc(pos_sc, neg_sc)


@torch.no_grad()
def collect_scores(router, loader, device):
    router.eval()
    out = []
    for inputs, _, _, terrain_yx in loader:
        out.append(router.score(inputs[:, 0:1].to(device),
                                _as_b4(terrain_yx.to(device))).cpu().numpy())
    return np.concatenate(out)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_terrs = [t.strip() for t in args.train_terrains.split(",") if t.strip()]
    test_terrs = [t.strip() for t in args.test_terrains.split(",") if t.strip()]
    common = dict(z_layer=args.z_layer, root=args.data_root, seed=args.seed,
                  return_terrain=True)
    # 全量数据 (无 train/val 留出)
    pos_ds = SpectrumNetDataset(train_terrs, split="all", **common)
    neg_ds = SpectrumNetDataset(test_terrs, split="all", **common)
    pos_dl = DataLoader(pos_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)
    neg_dl = DataLoader(neg_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)
    # τ 校准 / 诊断用顺序完整 loader (不丢批)
    pos_full_dl = DataLoader(pos_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    print(f"pos(full)={len(pos_ds)} neg(full)={len(neg_ds)} "
          f"epochs={args.num_epochs} device={device}")

    router = BinaryRouter(width=args.width).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.num_epochs // 3),
                                            gamma=0.5)
    crit = nn.BCEWithLogitsLoss()

    log_path = os.path.join(args.save_dir, "train_router_bin.log")
    with open(log_path, "w") as f:
        f.write("epoch\ttrain_loss\ttrain_acc\ttrain_auc\n")
    for epoch in range(1, args.num_epochs + 1):
        tic = time.time()
        tr_loss, tr_acc, tr_auc = run_epoch(
            router, pos_dl, _infinite(neg_dl), device, crit, opt)
        sched.step()
        line = f"{epoch}\t{tr_loss:.6f}\t{tr_acc:.4f}\t{tr_auc:.4f}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        print(f"epoch {epoch}/{args.num_epochs} {line} ({time.time() - tic:.1f}s)")

    ckpt_path = os.path.join(args.save_dir, "RouterBin.pt")
    torch.save({"type": "binary", "protocol": "test_time_geo_full",
                "state_dict": router.state_dict(), "width": args.width,
                "num_epochs": args.num_epochs,
                "train_terrains": train_terrs, "ood_terrains": test_terrs}, ckpt_path)
    print(f"saved: {ckpt_path}")

    # ---- τ in-sample 校准 (全量源域正样本 score 分位) + 各场景路由统计 ----
    val_scores = collect_scores(router, pos_full_dl, device)
    taus = {str(r): float(np.quantile(val_scores, 1.0 - r))
            for r in (0.90, 0.95, 0.99)}
    stats = {
        "source": {"n": int(len(val_scores)), "score_mean": float(val_scores.mean()),
                   "score_p01": float(np.quantile(val_scores, 0.01)),
                   "score_p05": float(np.quantile(val_scores, 0.05)),
                   "score_p10": float(np.quantile(val_scores, 0.10))},
        "tau_by_recall": taus,
        "target": {},
    }
    tau = taus["0.95"]
    print(f"[source] n={len(val_scores)} score mean={val_scores.mean():.4f} "
          f"p05={stats['source']['score_p05']:.4f} | tau@0.95={tau:.4f}")
    for tt in test_terrs:
        tds = SpectrumNetDataset([tt], split="all", **common)
        tdl = DataLoader(tds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)
        sc = collect_scores(router, tdl, device)
        stats["target"][tt] = {
            "n": int(len(sc)), "score_mean": float(sc.mean()),
            "score_median": float(np.median(sc)),
            "frac_ge_tau95": float((sc >= tau).mean()),
        }
        print(f"[target {tt}] n={len(sc)} score mean={sc.mean():.4f} "
              f"median={np.median(sc):.4f} frac>=tau95={stats['target'][tt]['frac_ge_tau95']:.4f}")

    with open(os.path.join(args.save_dir, "router_bin_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"saved: {os.path.join(args.save_dir, 'router_bin_stats.json')}")


if __name__ == "__main__":
    main()
