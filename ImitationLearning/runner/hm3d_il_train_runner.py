import datetime
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ImitationLearning.runner.il_train_runner import ILTrainRunner


class HM3DILTrainRunner:
    """Behavior-cloning runner for Habitat/HM3D expert trajectories."""

    def __init__(
        self,
        agent,
        dataset,
        device=None,
        lr=1e-4,
        batch_size=8,
        val_split=0.15,
        topk=(1, 2, 3),
        freeze_encoder=False,
        log_dir=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = agent.to(self.device)
        self.topk = tuple(int(k) for k in topk)

        if freeze_encoder:
            for param in self.agent.encoder.parameters():
                param.requires_grad_(False)

        val_size = max(1, int(len(dataset) * float(val_split))) if len(dataset) > 1 else 0
        train_size = len(dataset) - val_size
        if val_size > 0:
            train_set, val_set = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            train_set, val_set = dataset, dataset

        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ILTrainRunner.seq_collate,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ILTrainRunner.seq_collate,
            num_workers=0,
        )

        params = [param for param in self.agent.parameters() if param.requires_grad]
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
        self.criterion = self._build_weighted_loss(dataset)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = log_dir or f"RL_training/runs/HM3D_Imitation_{timestamp}"
        self.writer = SummaryWriter(self.log_dir)
        print(f"[INFO] HM3D IL TensorBoard logs: {self.log_dir}")

    def _build_weighted_loss(self, dataset):
        counts = np.zeros(int(dataset.num_actions), dtype=np.float64)
        for path in dataset.files:
            with np.load(path, allow_pickle=False) as data:
                actions = data["actions"].astype(np.int64)
            for action in actions:
                if 0 <= int(action) < len(counts):
                    counts[int(action)] += 1
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        print(f"[INFO] HM3D IL action counts: {counts.astype(int).tolist()}")
        print(f"[INFO] HM3D IL class weights: {[round(float(w), 3) for w in weights]}")
        return nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
        )

    def run(self, num_epochs=30, save_folder=None):
        save_folder = Path(save_folder).expanduser().resolve() if save_folder else None
        best_acc = -1.0
        best_path = None

        for epoch in range(1, int(num_epochs) + 1):
            train_loss, train_acc = self._run_epoch(epoch)
            val_loss, val_metrics = self.evaluate(epoch)
            top1 = val_metrics.get(1, 0.0)

            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/top1", train_acc, epoch)
            self.writer.add_scalar("val/loss", val_loss, epoch)
            for k, value in val_metrics.items():
                self.writer.add_scalar(f"val/top{k}", value, epoch)
            self.writer.flush()

            print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} train_top1={train_acc:.2f}% "
                f"val_loss={val_loss:.4f} val_top1={top1:.2f}%"
            )

            if save_folder and top1 >= best_acc:
                best_acc = top1
                save_folder.mkdir(parents=True, exist_ok=True)
                best_path = save_folder / "hm3d_imitation_best.pth"
                self.save_checkpoint(best_path, epoch, best_acc)

        if save_folder:
            final_path = save_folder / "hm3d_imitation_final.pth"
            self.save_checkpoint(final_path, int(num_epochs), best_acc)
            print(f"[INFO] Final HM3D IL checkpoint: {final_path}")
            if best_path:
                print(f"[INFO] Best HM3D IL checkpoint: {best_path} (top1={best_acc:.2f}%)")

    def _run_epoch(self, epoch):
        self.agent.train()
        total_loss = 0.0
        all_logits = []
        all_targets = []
        iterator = tqdm(self.train_loader, desc=f"HM3D IL Epoch {epoch}", leave=False)
        for x_batch, last_act, tgt_act, lengths in iterator:
            last_act = last_act.to(self.device)
            tgt_act = tgt_act.to(self.device)
            logits = self.agent.forward(x_batch, last_act)
            pack_logits = nn.utils.rnn.pack_padded_sequence(
                logits, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            pack_tgt = nn.utils.rnn.pack_padded_sequence(
                tgt_act, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            loss = self.criterion(pack_logits.data, pack_tgt.data)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            all_logits.append(pack_logits.data.detach().cpu())
            all_targets.append(pack_tgt.data.detach().cpu())
            iterator.set_postfix(loss=f"{loss.item():.4f}")

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return total_loss / max(len(self.train_loader), 1), self.compute_metrics(logits, targets)[1]

    def evaluate(self, epoch=0):
        self.agent.eval()
        total_loss = 0.0
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for x_batch, last_act, tgt_act, lengths in tqdm(self.val_loader, desc="HM3D IL Val", leave=False):
                last_act = last_act.to(self.device)
                tgt_act = tgt_act.to(self.device)
                logits = self.agent.forward(x_batch, last_act)
                pack_logits = nn.utils.rnn.pack_padded_sequence(
                    logits, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                pack_tgt = nn.utils.rnn.pack_padded_sequence(
                    tgt_act, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                loss = self.criterion(pack_logits.data, pack_tgt.data)
                total_loss += float(loss.item())
                all_logits.append(pack_logits.data.cpu())
                all_targets.append(pack_tgt.data.cpu())

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return total_loss / max(len(self.val_loader), 1), self.compute_metrics(logits, targets)

    def compute_metrics(self, logits, targets):
        metrics = {}
        max_k = min(max(self.topk), logits.shape[-1])
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        for k in self.topk:
            kk = min(int(k), max_k)
            metrics[int(k)] = correct[:, :kk].any(dim=1).float().mean().item() * 100.0
        return metrics

    def save_checkpoint(self, path, epoch, best_top1):
        payload = {
            "model_state_dict": self.agent.state_dict(),
            "epoch": int(epoch),
            "best_top1": float(best_top1),
            "navigation_config": self.agent.navigation_config,
            "num_actions": int(self.agent.num_actions),
        }
        torch.save(payload, str(path))
        print(f"[INFO] Saved HM3D IL checkpoint: {path}")
