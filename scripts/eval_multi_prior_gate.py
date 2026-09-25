"""多物理先验门控评估脚本。

使用建筑覆盖率 + 地形高程组合规则 (v4) 评估门控性能。
需要从 npz 文件加载 terrain_yx 以提取高程信息。

用法:
  cd /data/userSZ/duanrj/SpectrumDG
  python scripts/eval_multi_prior_gate.py --run_dir run/cbam_dg_gate_4to4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from data_process.dataset import SpectrumNetDataset, TERRAIN_ORDER, _FILE_RE, _apply_pathloss_threshold, _tx_proximity_map
from model.radiownet import RadioWNet
from PIL import Image


class SpectrumNetDatasetWithTerrain(Dataset):
    """扩展版数据集，返回地形高程图。"""
    
    def __init__(self, terrains, z_layer="00", root=None, max_per_terrain=None, seed=42, split="val", split_ratio=0.85):
        self.root = root or "/data/userSZ/duanrj/.cache/kagglehub/datasets/SpectrumNet/PPData5D-success"
        self.dir_png = os.path.join(self.root, "png")
        self.dir_npz = os.path.join(self.root, "npz")
        self.z_layer = z_layer
        self.thresh = 0.2
        self.z_i = int(z_layer)
        
        terrain_to_id = {t: i for i, t in enumerate(TERRAIN_ORDER)}
        self.terrains = list(terrains)
        
        # 索引文件
        rng = np.random.RandomState(seed)
        all_items = []
        for terr in self.terrains:
            tdir = os.path.join(self.dir_png, terr)
            collected = []
            for fn in sorted(os.listdir(tdir)):
                m = _FILE_RE.match(fn)
                if not m:
                    continue
                base, f, z = m.group(1), m.group(2), m.group(3)
                if z != self.z_layer:
                    continue
                png_path = os.path.join(tdir, fn)
                npz_path = os.path.join(self.dir_npz, base + "_bdtr.npz")
                if not os.path.exists(npz_path):
                    continue
                collected.append((png_path, npz_path, terrain_to_id[terr]))
            if max_per_terrain is not None and len(collected) > max_per_terrain:
                idx = rng.permutation(len(collected))[:max_per_terrain]
                collected = [collected[i] for i in idx]
            all_items.extend(collected)
        
        # 按 split 划分
        order = rng.permutation(len(all_items))
        n_train = int(len(all_items) * split_ratio)
        if split == "train":
            self.items = [all_items[i] for i in order[:n_train]]
        elif split == "val":
            self.items = [all_items[i] for i in order[n_train:]]
        elif split in ("all", "test"):
            self.items = all_items
        else:
            raise ValueError(f"unknown split: {split}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        png_path, npz_path, terrain_id = self.items[idx]
        
        # 增益图
        gain = np.array(Image.open(png_path), dtype=np.float32) / 255.0
        gain = _apply_pathloss_threshold(gain, self.thresh)
        tx_idx = np.unravel_index(np.argmax(gain), gain.shape)
        tx_y, tx_x = int(tx_idx[0]), int(tx_idx[1])
        tx_prox = _tx_proximity_map(tx_y, tx_x, gain.shape[0], gain.shape[1])
        
        # 建筑 + 地形高程
        npz = np.load(npz_path)
        bldg = npz['inBldg_zyx'][self.z_i].astype(np.float32)
        terrain_yx = npz['terrain_yx'].astype(np.float32)
        npz.close()
        
        # 输入 2 通道
        inp = np.stack([bldg, tx_prox], axis=0)
        target = gain[None, ...].astype(np.float32)
        
        return (torch.from_numpy(inp), 
                torch.from_numpy(target), 
                terrain_id,
                torch.from_numpy(terrain_yx))


def extract_gate_features(bldg, terrain_yx):
    """提取门控特征（简化版，仅用于推理）。"""
    feats = {}
    feats['bldg_cov'] = float(bldg.mean())
    labeled, ncomp = ndimage.label(bldg > 0.5)
    feats['bldg_ncomp'] = float(ncomp)
    feats['elev_mean'] = float(terrain_yx.mean())
    return feats


def multi_prior_gate_v4(feats):
    """多物理先验门控 v4: 建筑 OR (低高程 AND 弱建筑)。"""
    bldg_strong = feats['bldg_cov'] > 0.01 or feats['bldg_ncomp'] > 5
    elev_low = feats['elev_mean'] < 100
    bldg_weak = feats['bldg_cov'] > 0.001 or feats['bldg_ncomp'] > 0
    return bldg_strong or (elev_low and bldg_weak)


def evaluate_with_multi_prior_gate(model, dataloader, device):
    """使用多物理先验门控评估模型。"""
    model.eval()
    crit = nn.MSELoss()
    
    metrics = {
        'firstU': {'mse': 0.0, 'nmse': 0.0, 'n': 0},
        'secondU': {'mse': 0.0, 'nmse': 0.0, 'n': 0},
        'gated': {'mse': 0.0, 'nmse': 0.0, 'n': 0, 'n_second': 0},
    }
    
    with torch.no_grad():
        for inp, target, terrain_id, terrain_yx in dataloader:
            inp = inp.to(device)
            target = target.to(device)
            terrain_yx = terrain_yx.to(device)
            
            # 模型前向
            o1, o2 = model(inp)
            
            # 计算指标
            tgt_mse = crit(target, torch.zeros_like(target)) + 1e-12
            
            # firstU
            mse1 = crit(o1, target)
            metrics['firstU']['mse'] += mse1.item() * target.size(0)
            metrics['firstU']['nmse'] += (mse1 / tgt_mse).item() * target.size(0)
            metrics['firstU']['n'] += target.size(0)
            
            # secondU
            mse2 = crit(o2, target)
            metrics['secondU']['mse'] += mse2.item() * target.size(0)
            metrics['secondU']['nmse'] += (mse2 / tgt_mse).item() * target.size(0)
            metrics['secondU']['n'] += target.size(0)
            
            # 门控：逐样本判断
            for i in range(inp.size(0)):
                bldg = inp[i, 0].cpu().numpy()
                t_yx = terrain_yx[i].cpu().numpy()
                feats = extract_gate_features(bldg, t_yx)
                use_second = multi_prior_gate_v4(feats)
                
                pred = o2[i] if use_second else o1[i]
                mse_g = crit(pred.unsqueeze(0), target[i:i+1])
                metrics['gated']['mse'] += mse_g.item()
                metrics['gated']['nmse'] += (mse_g / tgt_mse).item()
                metrics['gated']['n'] += 1
                if use_second:
                    metrics['gated']['n_second'] += 1
    
    # 计算平均值
    for key in metrics:
        n = metrics[key]['n']
        if n > 0:
            metrics[key]['mse'] /= n
            metrics[key]['nmse'] /= n
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--train_terrains", type=str, default="05.Suburban,06.DenseUrban,07.Rural,08.OrdinaryUrban")
    parser.add_argument("--test_terrains", type=str, default="01.Grassland,09.Desert,10.Mountainous,11.Forest")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = RadioWNet(use_causal=True, num_domains=4, use_causal_2nd=False)
    ckpt_path = os.path.join(args.run_dir, "checkpoint_secondU.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, "Trained_Model_SecondU.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    print(f"模型加载自: {ckpt_path}")
    
    train_terrs = [t.strip() for t in args.train_terrains.split(",")]
    test_terrs = [t.strip() for t in args.test_terrains.split(",")]
    
    report = {}
    
    # 域内评估（从训练集划分 val）
    print("\n===== 域内评估 (val) =====")
    val_ds = SpectrumNetDatasetWithTerrain(train_terrs, z_layer="00", max_per_terrain=None, split="val", split_ratio=0.85)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_metrics = evaluate_with_multi_prior_gate(model, val_dl, device)
    print(f"域内 (n={len(val_ds)}):")
    print(f"  firstU  NMSE: {val_metrics['firstU']['nmse']:.6f}")
    print(f"  secondU NMSE: {val_metrics['secondU']['nmse']:.6f}")
    print(f"  gated   NMSE: {val_metrics['gated']['nmse']:.6f}  (gate_second_frac: {val_metrics['gated']['n_second']/val_metrics['gated']['n']:.3f})")
    report['indomain'] = {k: v['nmse'] for k, v in val_metrics.items()}
    report['indomain']['gate_second_frac'] = val_metrics['gated']['n_second'] / val_metrics['gated']['n']
    
    # 跨域评估
    print("\n===== 跨域评估 =====")
    cross_nmse = {'firstU': [], 'secondU': [], 'gated': []}
    for tt in test_terrs:
        test_ds = SpectrumNetDatasetWithTerrain([tt], z_layer="00", max_per_terrain=None)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_metrics = evaluate_with_multi_prior_gate(model, test_dl, device)
        print(f"{tt} (n={len(test_ds)}):")
        print(f"  firstU  NMSE: {test_metrics['firstU']['nmse']:.6f}")
        print(f"  secondU NMSE: {test_metrics['secondU']['nmse']:.6f}")
        print(f"  gated   NMSE: {test_metrics['gated']['nmse']:.6f}  (gate_second_frac: {test_metrics['gated']['n_second']/test_metrics['gated']['n']:.3f})")
        report[f'cross_{tt}'] = {k: v['nmse'] for k, v in test_metrics.items()}
        report[f'cross_{tt}']['gate_second_frac'] = test_metrics['gated']['n_second'] / test_metrics['gated']['n']
        for key in ['firstU', 'secondU', 'gated']:
            cross_nmse[key].append(test_metrics[key]['nmse'])
    
    # 跨域平均
    report['cross_avg'] = {key: sum(vals)/len(vals) for key, vals in cross_nmse.items()}
    print(f"\n跨域平均:")
    print(f"  firstU  NMSE: {report['cross_avg']['firstU']:.6f}")
    print(f"  secondU NMSE: {report['cross_avg']['secondU']:.6f}")
    print(f"  gated   NMSE: {report['cross_avg']['gated']:.6f}")
    
    # 保存报告
    report_path = os.path.join(args.run_dir, "dg_report_multi_prior.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n报告保存至: {report_path}")


if __name__ == "__main__":
    main()
