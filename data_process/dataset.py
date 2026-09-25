"""SpectrumNet 数据集 dataloader。

数据组织 (PPData5D-success):
  png/{terrain}/T{tt}C{c}D{dddd}_n{nn}_f{ff}_ss_z{zz}.png  — 128x128 uint8 增益图
  npz/T{tt}C{c}D{dddd}_n{nn}_bdtr.npz  — inBldg_zyx(3,128,128) 二值建筑, terrain_yx(128,128) 高程

每张 png 对应一个 (terrain, climate, data_id, sample, freq, z) 组合。
Tx(发射机) = z00 增益图的全局最大像素(已验证径向衰减, 物理 Tx 位置)。

输入(2通道): [buildings(z00 二值), Tx 邻近图(log-distance, 近Tx=1)]
目标(1通道): 增益图 /255 后 pathloss 阈值 0.2 → [0,1]
domain: terrain 整数 id (用于 CIRL 域分类 + 分域评估)
"""
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_ROOT = "/data/userSZ/duanrj/.cache/kagglehub/datasets/SpectrumNet/PPData5D-success"

# terrain 全序 (决定全局 domain id)
TERRAIN_ORDER = ["01.Grassland", "02.Island", "03.Ocean", "04.Lake",
                 "05.Suburban", "06.DenseUrban", "07.Rural",
                 "08.OrdinaryUrban", "09.Desert", "10.Mountainous", "11.Forest"]

_FILE_RE = re.compile(r'^(T\d+C\d+D\d+_n\d+)_f(\d+)_ss_z(\d+)\.png$')


def _tx_proximity_map(tx_y, tx_x, H=128, W=128):
    """log10 距离 → 归一化 → 1-normalize, 近 Tx=1, 远=0。与 RadioDUN MixDataset 一致。"""
    yy, xx = np.indices((H, W))
    d = np.log10(np.sqrt((yy - tx_y) ** 2 + (xx - tx_x) ** 2) + 1.0)
    d = 1.0 - (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d.astype(np.float32)


def _apply_pathloss_threshold(gain01, thresh=0.2):
    """RadioUNet loaders.py 同款 pathloss 阈值变换: <thresh → thresh, 再 (x-thresh)/(1-thresh)。"""
    if thresh > 0:
        mask = gain01 < thresh
        gain01 = gain01.copy()
        gain01[mask] = thresh
        gain01 = (gain01 - thresh) / (1.0 - thresh)
    return gain01


def _parse_list(s):
    """逗号分隔字符串 → 去空白后的非空列表。"""
    return [t.strip() for t in s.split(",") if t.strip()]


class SpectrumNetDataset(Dataset):
    """单域或跨域 SpectrumNet 数据集。

    Args:
        terrains: 地形名列表, 如 ["05.Suburban", "06.DenseUrban"]; 也接受 "ALL" 用全部 11 个
        z_layer: 选用的 z 层, "00"/"01"/"02"; 仅 z00 的 Tx 提取最可靠
        freqs: 选用的 freq 集合, 如 ["00","01"]; None=全部
        split: "train"/"val"/"test"/"all"
        split_ratio: train 占比, 剩余 val (test 走独立 dataset 实例)
        root: PPData5D-success 根目录
        thresh: pathloss 阈值
        max_per_terrain: 每个 terrain 上限 (None=不限), 用于快速小规模验证
        seed: 确定性划分种子
    """
    TERRAIN_ORDER = TERRAIN_ORDER

    def __init__(self, terrains, z_layer="00", freqs=None,
                 split="train", split_ratio=0.85,
                 root=DEFAULT_DATA_ROOT,
                 thresh=0.2, max_per_terrain=None, seed=42):
        self.root = root
        self.dir_png = os.path.join(root, "png")
        self.dir_npz = os.path.join(root, "npz")
        self.z_layer = z_layer
        self.freqs = set(freqs) if freqs is not None else None
        self.thresh = thresh

        if terrains == "ALL":
            terrains = self.TERRAIN_ORDER
        # terrain → 整数 domain id (按 TERRAIN_ORDER 的位置)
        terrain_to_id = {t: i for i, t in enumerate(self.TERRAIN_ORDER)}
        self.terrains = list(terrains)

        # 索引所有匹配文件
        rng = np.random.RandomState(seed)
        all_items = []  # list of (png_path, npz_path, terrain_id)
        for terr in self.terrains:
            tdir = os.path.join(self.dir_png, terr)
            if not os.path.isdir(tdir):
                raise FileNotFoundError(f"terrain dir 不存在: {tdir}")
            collected = []
            for fn in sorted(os.listdir(tdir)):
                m = _FILE_RE.match(fn)
                if not m:
                    continue
                base, f, z = m.group(1), m.group(2), m.group(3)
                if z != self.z_layer:
                    continue
                if self.freqs is not None and f not in self.freqs:
                    continue
                png_path = os.path.join(tdir, fn)
                npz_path = os.path.join(self.dir_npz, base + "_bdtr.npz")
                if not os.path.exists(npz_path):
                    continue
                collected.append((png_path, npz_path, terrain_to_id[terr]))
            if max_per_terrain is not None and len(collected) > max_per_terrain:
                # 确定性下采样
                idx = rng.permutation(len(collected))[:max_per_terrain]
                collected = [collected[i] for i in idx]
            all_items.extend(collected)

        # 确定性划分
        order = rng.permutation(len(all_items))
        n_train = int(len(all_items) * split_ratio)
        if split == "train":
            self.items = [all_items[i] for i in order[:n_train]]
        elif split == "val":
            self.items = [all_items[i] for i in order[n_train:]]
        elif split in ("all", "test"):
            # test 为跨域独立场景, 不做 split
            self.items = all_items
        else:
            raise ValueError(f"unknown split: {split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        png_path, npz_path, terrain_id = self.items[idx]
        # 增益图
        gain = np.array(Image.open(png_path), dtype=np.float32) / 255.0  # [0,1]
        gain = _apply_pathloss_threshold(gain, self.thresh)             # [0,1]
        # Tx = 全局最大像素
        tx_idx = np.unravel_index(np.argmax(gain), gain.shape)
        tx_y, tx_x = int(tx_idx[0]), int(tx_idx[1])
        tx_prox = _tx_proximity_map(tx_y, tx_x, gain.shape[0], gain.shape[1])
        # 建筑 (z_layer 对应 inBldg_zyx 的第 z 层)
        z_i = int(self.z_layer)
        npz = np.load(npz_path)
        bldg = npz['inBldg_zyx'][z_i].astype(np.float32)  # 0/1
        npz.close()
        # 输入 2 通道
        inp = np.stack([bldg, tx_prox], axis=0)  # (2,H,W)
        target = gain[None, ...].astype(np.float32)  # (1,H,W)
        return torch.from_numpy(inp), torch.from_numpy(target), terrain_id


def build_dataloaders(train_terrains, test_terrain, z_layer="00", freqs=None,
                      batch_size=32, num_workers=4, max_per_terrain=None,
                      root=DEFAULT_DATA_ROOT, seed=42):
    """构造 train/val(从训练场景划分) + test(跨域, 独立场景) dataloaders。"""
    common = dict(z_layer=z_layer, freqs=freqs, root=root,
                  max_per_terrain=max_per_terrain, seed=seed)
    train_ds = SpectrumNetDataset(train_terrains, split="train", split_ratio=0.85, **common)
    val_ds = SpectrumNetDataset(train_terrains, split="val", split_ratio=0.85, **common)
    test_ds = SpectrumNetDataset([test_terrain], split="test", split_ratio=1.0, **common)
    g = torch.Generator()
    g.manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True, generator=g)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds


def build_test_dataloader(test_terrain, z_layer="00", freqs=None,
                          batch_size=32, num_workers=4, max_per_terrain=None,
                          root=DEFAULT_DATA_ROOT, seed=42):
    """只构造单个跨域测试场景的 dataloader (评估复用, 免建训练集索引)。"""
    test_ds = SpectrumNetDataset([test_terrain], split="test", split_ratio=1.0,
                                 z_layer=z_layer, freqs=freqs, root=root,
                                 max_per_terrain=max_per_terrain, seed=seed)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


def build_data(cfg):
    """按 TrainConfig 构造 dataloaders。

    Returns:
        (train_terrs, test_terrs, train_dl, val_dl, test_dls):
        test_dls 为 {terrain: DataLoader}, 每个跨域测试场景独立一份。
    """
    freqs = _parse_list(cfg.freqs) or None
    train_terrs = _parse_list(cfg.train_terrains)
    test_terrs = _parse_list(cfg.test_terrains)
    common = dict(z_layer=cfg.z_layer, freqs=freqs, batch_size=cfg.batch_size,
                  num_workers=cfg.num_workers, max_per_terrain=cfg.max_per_terrain,
                  root=cfg.data_root, seed=cfg.seed)
    train_dl, val_dl, *_ = build_dataloaders(train_terrs, test_terrs[0], **common)
    # 为每个 test 场景构建独立 DataLoader
    test_dls = {}
    for tt in test_terrs:
        _, _, tdl, *_ = build_dataloaders(train_terrs, tt, **common)
        test_dls[tt] = tdl
    print(f"train={len(train_dl.dataset)} val={len(val_dl.dataset)} "
          f"test={{{', '.join(f'{t}={len(test_dls[t].dataset)}' for t in test_terrs)}}}")
    return train_terrs, test_terrs, train_dl, val_dl, test_dls


if __name__ == "__main__":
    # 自检
    ds = SpectrumNetDataset(["05.Suburban", "06.DenseUrban"], split="all", z_layer="00")
    print(f"dataset size: {len(ds)}")
    inp, tgt, did = ds[0]
    print(f"input: {inp.shape} {inp.dtype} range [{inp.min():.3f},{inp.max():.3f}]")
    print(f"target: {tgt.shape} {tgt.dtype} range [{tgt.min():.3f},{tgt.max():.3f}]")
    print(f"domain id: {did}")
    print(f"  ch0 buildings sum: {inp[0].sum():.0f}")
    print(f"  ch1 tx_prox at center: {inp[1].max():.3f}")
