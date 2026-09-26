"""统一两阶段训练器。

Stage 1 (firstU):  监督 output1, 验证选优 MSE_firstU
Stage 2 (secondU): 监督 output2 (firstU 路径由模型内 detach 冻结), 验证选优 MSE_secondU

DG 模块可任意组合, 由 cfg 开关控制 (损失组件见 lib/dg.py):
  CIRL 因果分解 (瓶颈层): ERM + λ_dom·CE_spe + λ_inv·CE_inv(GRL) + λ_orth·orth
  Fish 梯度对齐 (损失层): ERM + λ_fish·Σ_d ||∇_θ L_d - ∇_θ L̄||² (仅对 ERM loss 逐域梯度,
    不干扰 CIRL 辅助损失的梯度动态)
"""
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from .dg import build_domain_label_tensor, cirl_aux_loss, fish_penalty
from .evaluate import evaluate


class Trainer:
    """两阶段训练器 (firstU → secondU), 训练循环 + 早停。

    双专家架构 (arch="dual") 时调用 train_dual() 联合训练:
    expert_in (RadioWNet 纯ERM) 与 expert_dg (CBAMUNet+CIRL+Fish)
    同批前向、独立损失、独立早停选优。

    Args:
        model: RadioWNet 或 DualExpertNet
        loaders: {'train': DataLoader, 'val': DataLoader}
        device: torch.device
        cfg: configs.TrainConfig
        domain_map: {全局 terrain_id: 训练域局部 id}, 仅 use_causal 时使用
    """

    STAGES = ("firstU", "secondU")

    def __init__(self, model, loaders, device, cfg, domain_map):
        self.model = model
        self.loaders = loaders
        self.device = device
        self.cfg = cfg
        self.domain_map = domain_map

    # ------------------------------------------------------------------ #
    # 双专家联合训练
    # ------------------------------------------------------------------ #

    def train_dual(self, resume_from=None, checkpoint_interval=10):
        """双专家同批联合训练, 返回 best_val 权重 (两专家独立早停选优)。

        每个训练 batch:
            expert_in (RadioWNet phase=joint): ERM(output1) + ERM(output2)
            expert_dg (CBAMUNet): ERM + CIRL aux + Fish penalty
        验证/早停: MSE_secondU (域内专家) 与 MSE_dg (泛化专家) 各自独立计数,
        保存各自 best_state, 仅当两个专家均连续 patience 轮无改善时才停止训练
        (任一专家仍在改善则继续, 提前收敛的专家沿用其 best 权重)。
        """
        cfg = self.cfg
        model = self.model
        log_path = os.path.join(cfg.save_dir, "train_loss_dual.log")

        opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = lr_scheduler.StepLR(opt, step_size=max(1, cfg.num_epochs // 3), gamma=0.5)
        crit = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        params = [p for p in model.parameters() if p.requires_grad]
        dmap = build_domain_label_tensor(self.domain_map, self.device) if cfg.use_causal else None
        decomposer = model.expert_dg.causal

        start_epoch = 0
        best = {'in': float('inf'), 'dg': float('inf')}
        best_state = {'in': None, 'dg': None}  # 各专家独立快照, 互不污染
        no_improve = {'in': 0, 'dg': 0}

        if resume_from is not None:
            ckpt = self.load_checkpoint(resume_from, model, opt, sched, self.device)
            start_epoch = ckpt['epoch'] + 1
            best = ckpt.get('best', best)
            best_state = ckpt.get('best_state', best_state)
            no_improve = ckpt.get('no_improve', {'in': 0, 'dg': 0})
            print(f"[Resume] dual 从 epoch {start_epoch} 恢复 "
                  f"(best_in={best['in']:.6f}, best_dg={best['dg']:.6f})")
        else:
            with open(log_path, "w") as f:
                f.write(f"# SpectrumDG dual-expert | train={cfg.train_terrains} test={cfg.test_terrains} "
                        f"in=RadioWNet(ERM) dg=CBAMUNet(CIRL+Fish) z={cfg.z_layer} freqs={cfg.freqs or 'all'}\n")
                f.write("epoch\ttrain_avg_in\ttrain_avg_dg\tval_avg_in\tval_avg_dg\n")

        ckpt_path = os.path.join(cfg.save_dir, "checkpoint_dual.pt")

        for epoch in range(start_epoch, cfg.num_epochs):
            model.train()
            since = time.time()
            stats = self._train_dual_epoch(opt, crit, ce, params, dmap, decomposer)
            sched.step()

            # 验证: 两个专家独立指标
            val_in, val_dg = self._validate_dual()
            n = max(stats['n'], 1)
            train_in = stats['erm_in'] / n
            train_dg = stats['erm_dg'] / n
            # 日志文件只记录两专家主损失 + 验证 MSE (4 列); CIRL 辅助项仅控制台诊断
            with open(log_path, "a") as f:
                f.write(f"{epoch}\t{train_in:.6f}\t{train_dg:.6f}"
                        f"\t{val_in:.6f}\t{val_dg:.6f}\n")

            if val_in < best['in']:
                best['in'] = val_in
                no_improve['in'] = 0
                best_state['in'] = {k: v.clone() for k, v in self.model.expert_in.state_dict().items()}
            else:
                no_improve['in'] += 1
            if val_dg < best['dg']:
                best['dg'] = val_dg
                no_improve['dg'] = 0
                best_state['dg'] = {k: v.clone() for k, v in self.model.expert_dg.state_dict().items()}
            else:
                no_improve['dg'] += 1

            print(f"Epoch {epoch:3d}/{cfg.num_epochs} [dual in=ERM dg=CIRL+Fish] "
                  f"train_in={train_in:.6f} train_dg={train_dg:.6f} "
                  f"dom={stats['dom'] / n:.4f} inv={stats['inv'] / n:.4f} orth={stats['orth'] / n:.4f} "
                  f"val_in={val_in:.6f} val_dg={val_dg:.6f} ({time.time() - since:.0f}s)")

            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(
                    ckpt_path, epoch, model, opt, sched,
                    best['dg'], best_state, no_improve)
                print(f"[Checkpoint] saved at epoch {epoch} → {ckpt_path}")

            if no_improve['in'] >= cfg.patience and no_improve['dg'] >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (both experts no improvement "
                      f"for {cfg.patience} epochs)")
                break

        if best_state['in'] is not None:
            self.model.expert_in.load_state_dict(best_state['in'])
        if best_state['dg'] is not None:
            self.model.expert_dg.load_state_dict(best_state['dg'])
        return model

    def _train_dual_epoch(self, opt, crit, ce, params, dmap, decomposer):
        """双专家单 epoch 训练。"""
        cfg = self.cfg
        stats = {"erm_in": 0.0, "erm_dg": 0.0, "dom": 0.0, "inv": 0.0, "orth": 0.0, "n": 0}
        for inputs, targets, domain_ids in self.loaders['train']:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            domain_ids = domain_ids.to(self.device)
            opt.zero_grad()

            # 域内专家: RadioWNet joint 相位 (output1 + output2 均计入 ERM)
            o1_in, o2_in = self.model.expert_in(inputs)
            loss_in = crit(o1_in, targets) + crit(o2_in, targets)

            # 泛化专家: CBAMUNet + CIRL + Fish
            out_dg = self.model.expert_dg(inputs)
            loss_dg = crit(out_dg, targets)

            total = loss_in + loss_dg
            if cfg.use_causal:
                aux = cirl_aux_loss(decomposer, dmap[domain_ids], ce)
                if aux is not None:
                    loss_dom, loss_inv, loss_orth = aux
                    total = (total
                             + cfg.causal_lambda_dom * loss_dom
                             + cfg.causal_lambda_inv * loss_inv
                             + cfg.causal_lambda_orth * loss_orth)
                    stats["dom"] += loss_dom.item() * inputs.size(0)
                    stats["inv"] += loss_inv.item() * inputs.size(0)
                    stats["orth"] += loss_orth.item() * inputs.size(0)

            if cfg.use_fish and cfg.fish_lambda > 0:
                pen = fish_penalty(out_dg, targets, domain_ids, params, crit)
                if pen is not None:
                    total = total + cfg.fish_lambda * pen

            total.backward()
            opt.step()
            stats["erm_in"] += loss_in.item() * inputs.size(0)
            stats["erm_dg"] += loss_dg.item() * inputs.size(0)
            stats["n"] += inputs.size(0)
        return stats

    def _validate_dual(self):
        """双专家验证, 返回 (MSE_expert_in, MSE_expert_dg)。"""
        model = self.model
        model.eval()
        sums = {'in': 0.0, 'dg': 0.0}
        n = 0
        crit = nn.MSELoss()
        with torch.no_grad():
            for batch in self.loaders['val']:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                _, o2_in = model.expert_in(inputs)
                out_dg = model.expert_dg(inputs)
                sums['in'] += crit(o2_in, targets).item() * targets.size(0)
                sums['dg'] += crit(out_dg, targets).item() * targets.size(0)
                n += targets.size(0)
        return sums['in'] / max(n, 1), sums['dg'] / max(n, 1)

    # ------------------------------------------------------------------ #
    # 级联两阶段训练 (原架构)
    # ------------------------------------------------------------------ #

    def train_stage(self, stage, resume_from=None, checkpoint_interval=10):
        """训练单个阶段并返回 best_val 权重。

        Args:
            stage: "firstU" 或 "secondU"
            resume_from: checkpoint 路径, 从中断处恢复训练
            checkpoint_interval: 每隔多少 epoch 自动保存一次 checkpoint (0=不保存)
        """
        assert stage in self.STAGES, f"unknown stage: {stage}"
        cfg = self.cfg
        model = self.model
        out_idx = self.STAGES.index(stage)
        val_key = f"MSE_{stage}"
        # dg_second=False: Stage2 纯 ERM (不挂 causal_2nd/Fish), secondU 专做域内精修
        apply_dg = (stage == "firstU") or cfg.dg_second
        log_path = os.path.join(
            cfg.save_dir,
            "train_loss.log" if stage == "firstU" else "train_loss_secondU.log")

        opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = lr_scheduler.StepLR(opt, step_size=max(1, cfg.num_epochs // 3), gamma=0.5)
        crit = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        params = [p for p in model.parameters() if p.requires_grad]
        dmap = build_domain_label_tensor(self.domain_map, self.device) if cfg.use_causal else None
        decomposer = model.causal if stage == "firstU" else model.causal_2nd

        start_epoch = 0
        best_val = float('inf')
        best_state = None
        no_improve = 0

        # ---- 从 checkpoint 恢复 ----
        if resume_from is not None:
            ckpt = self.load_checkpoint(resume_from, model, opt, sched, self.device)
            start_epoch = ckpt['epoch'] + 1
            best_val = ckpt['best_val']
            best_state = ckpt['best_state']
            no_improve = ckpt['no_improve']
            print(f"[Resume] {stage} 从 epoch {start_epoch} 恢复 "
                  f"(best_val={best_val:.6f}, no_improve={no_improve})")
        else:
            tag = self._method_tag(stage)
            self._init_log(log_path, stage, tag)

        ckpt_path = os.path.join(cfg.save_dir, f"checkpoint_{stage}.pt")

        for epoch in range(start_epoch, cfg.num_epochs):
            model.train()
            since = time.time()
            stats = self._train_epoch(opt, crit, ce, params, dmap, decomposer, out_idx,
                                      apply_dg=apply_dg)
            sched.step()

            train_avg = stats["erm"] / max(stats["n"], 1)
            val_metrics = evaluate(model, self.loaders['val'], self.device)
            val_avg = val_metrics[val_key]
            with open(log_path, "a") as f:
                f.write(f"{epoch}\t{train_avg:.6f}\t{val_avg:.6f}\n")

            if val_avg < best_val:
                best_val = val_avg
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            self._print_epoch(epoch, stage, self._method_tag(stage),
                              stats, val_key, val_avg, time.time() - since)

            # ---- 定期保存 checkpoint ----
            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(
                    ckpt_path, epoch, model, opt, sched,
                    best_val, best_state, no_improve)
                print(f"[Checkpoint] saved at epoch {epoch} → {ckpt_path}")

            if no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {cfg.patience} epochs)")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    # ------------------------------------------------------------------ #
    # Checkpoint 保存 / 加载
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path, epoch, model, opt, sched,
                        best_val, best_state, no_improve):
        """保存训练 checkpoint (模型权重 + 优化器 + 调度器 + 训练状态)。"""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'best_val': best_val,
            'best_state': best_state,
            'no_improve': no_improve,
        }
        torch.save(ckpt, path)

    @staticmethod
    def load_checkpoint(path, model, opt, sched, device):
        """加载 checkpoint 并恢复到 model / optimizer / scheduler。"""
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        sched.load_state_dict(ckpt['scheduler_state_dict'])
        return ckpt

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _train_epoch(self, opt, crit, ce, params, dmap, decomposer, out_idx, apply_dg=True):
        """单个 epoch 训练, 返回按样本数累计的损失统计 {erm, dom, inv, orth, n}。

        apply_dg=False 时纯 ERM (Stage2 域内精修模式, 跳过 Causal/Fish)。
        """
        cfg = self.cfg
        stats = {"erm": 0.0, "dom": 0.0, "inv": 0.0, "orth": 0.0, "n": 0}
        for inputs, targets, domain_ids in self.loaders['train']:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            domain_ids = domain_ids.to(self.device)
            opt.zero_grad()
            o1, o2 = self.model(inputs)
            out = (o1, o2)[out_idx]
            loss_erm = crit(out, targets)
            total = loss_erm

            if apply_dg and cfg.use_causal:
                aux = cirl_aux_loss(decomposer, dmap[domain_ids], ce)
                if aux is not None:
                    loss_dom, loss_inv, loss_orth = aux
                    total = (loss_erm
                             + cfg.causal_lambda_dom * loss_dom
                             + cfg.causal_lambda_inv * loss_inv
                             + cfg.causal_lambda_orth * loss_orth)
                    stats["dom"] += loss_dom.item() * inputs.size(0)
                    stats["inv"] += loss_inv.item() * inputs.size(0)
                    stats["orth"] += loss_orth.item() * inputs.size(0)

            if apply_dg and cfg.use_fish and cfg.fish_lambda > 0:
                pen = fish_penalty(out, targets, domain_ids, params, crit)
                if pen is not None:
                    total = total + cfg.fish_lambda * pen

            total.backward()
            opt.step()
            stats["erm"] += loss_erm.item() * inputs.size(0)
            stats["n"] += inputs.size(0)
        return stats

    def _method_tag(self, stage):
        """拼接当前启用的 DG 方法标签, 如 "CIRL(...)+Fish(0.1)"。

        dg_second=False 的 Stage2 标注 "ERM(no-DG)" 以区分纯 ERM 域内精修。
        """
        cfg = self.cfg
        if stage == "secondU" and not cfg.dg_second:
            return "ERM(no-DG)"
        methods = []
        if cfg.use_causal:
            methods.append(f"Causal(λdom={cfg.causal_lambda_dom},λinv={cfg.causal_lambda_inv},"
                           f"λorth={cfg.causal_lambda_orth})")
        if cfg.use_fish:
            methods.append(f"Fish(λ={cfg.fish_lambda})")
        return "+".join(methods) if methods else "ERM"

    def _init_log(self, log_path, stage, tag):
        cfg = self.cfg
        with open(log_path, "w") as f:
            f.write(f"# SpectrumDG {stage} | train={cfg.train_terrains} test={cfg.test_terrains} "
                    f"method={tag} z={cfg.z_layer} freqs={cfg.freqs or 'all'}\n")
            f.write("epoch\ttrain_avg\tval_avg\n")

    def _print_epoch(self, epoch, stage, tag, stats, val_key, val_avg, dt):
        cfg = self.cfg
        n = max(stats["n"], 1)
        print(f"Epoch {epoch:3d}/{cfg.num_epochs} [{stage} {tag}] erm={stats['erm'] / n:.6f} "
              f"dom={stats['dom'] / n:.4f} inv={stats['inv'] / n:.4f} "
              f"orth={stats['orth'] / n:.4f} val_{val_key.lower()}={val_avg:.6f}"
              f" ({dt:.0f}s)")
