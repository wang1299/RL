import datetime
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
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
        label_smoothing=0.05,
        early_stopping_patience=6,
        min_delta=1e-4,
        split_seed=42,
        val_split_mode="file",
        train_sampling_mode="scene_sqrt",
        train_epoch_samples=None,
        val_epoch_samples=None,
        num_workers=0,
        log_dir=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = agent.to(self.device)
        self.topk = tuple(int(k) for k in topk)
        self.label_smoothing = float(label_smoothing)
        self.early_stopping_patience = int(early_stopping_patience)
        self.min_delta = float(min_delta)
        self.split_seed = int(split_seed)
        self.val_split_mode = str(val_split_mode).lower()
        self.train_sampling_mode = str(train_sampling_mode).lower()
        self.train_epoch_samples = None if train_epoch_samples is None else int(train_epoch_samples)
        self.val_epoch_samples = None if val_epoch_samples is None else int(val_epoch_samples)
        self.num_workers = max(int(num_workers), 0)

        if freeze_encoder:
            for param in self.agent.encoder.parameters():
                param.requires_grad_(False)

        train_set, val_set = self._make_splits(dataset, float(val_split))

        train_sampler = self._make_train_sampler(dataset, train_set)
        collate_fn = dataset.seq_collate if getattr(dataset, "returns_features", False) else ILTrainRunner.seq_collate

        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

        params = [param for param in self.agent.parameters() if param.requires_grad]
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
        self.criterion = self._build_weighted_loss(dataset)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = log_dir or f"RL_training/runs/HM3D_Imitation_{timestamp}"
        self.writer = SummaryWriter(self.log_dir)
        print(f"[INFO] HM3D IL TensorBoard logs: {self.log_dir}")

    def _base_dataset_index(self, subset_or_dataset, idx):
        if isinstance(subset_or_dataset, Subset):
            return int(subset_or_dataset.indices[int(idx)])
        return int(idx)

    def _make_train_sampler(self, dataset, train_set):
        mode = self.train_sampling_mode
        if mode in {"none", "natural", "shuffle"}:
            print(f"[INFO] HM3D IL train sampling mode={mode}")
            return None
        if mode not in {"scene_sqrt"}:
            raise ValueError(f"Unsupported HM3D IL train_sampling_mode={mode!r}; use scene_sqrt or none")

        scene_counts = defaultdict(int)
        for local_idx in range(len(train_set)):
            base_idx = self._base_dataset_index(train_set, local_idx)
            path = Path(dataset.windows[base_idx]["path"])
            scene_counts[path.parent.name] += 1

        weights = []
        for local_idx in range(len(train_set)):
            base_idx = self._base_dataset_index(train_set, local_idx)
            path = Path(dataset.windows[base_idx]["path"])
            count = max(scene_counts[path.parent.name], 1)
            weights.append(1.0 / np.sqrt(float(count)))

        scene_weight_totals = defaultdict(float)
        for local_idx, weight in enumerate(weights):
            base_idx = self._base_dataset_index(train_set, local_idx)
            scene = Path(dataset.windows[base_idx]["path"]).parent.name
            scene_weight_totals[scene] += float(weight)
        totals = np.asarray(list(scene_weight_totals.values()), dtype=np.float64)
        if totals.size:
            print(
                "[INFO] HM3D IL train sampling mode=scene_sqrt "
                f"scenes={len(scene_weight_totals)} scene_weight_range="
                f"{totals.min():.3f}-{totals.max():.3f}"
            )

        num_samples = len(weights)
        if self.train_epoch_samples is not None and self.train_epoch_samples > 0:
            num_samples = min(int(self.train_epoch_samples), len(weights))
            print(
                "[INFO] HM3D IL train epoch sampling "
                f"num_samples={num_samples}/{len(weights)}"
            )

        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=True,
            generator=torch.Generator().manual_seed(self.split_seed),
        )

    def _make_splits(self, dataset, val_split):
        if len(dataset) <= 1 or val_split <= 0.0:
            print("[INFO] HM3D IL split: train=all, val=all")
            return dataset, dataset

        mode = self.val_split_mode
        if mode == "window":
            val_size = max(1, int(len(dataset) * val_split))
            val_size = min(val_size, len(dataset) - 1)
            train_size = len(dataset) - val_size
            train_set, val_set = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.split_seed),
            )
            print(
                f"[INFO] HM3D IL split mode=window train_windows={train_size} "
                f"val_windows={val_size}"
            )
            return train_set, val_set

        if mode not in {"file", "scene"}:
            raise ValueError(f"Unsupported HM3D IL val_split_mode={mode!r}; use file, scene, or window")

        grouped_indices = defaultdict(list)
        for idx, window in enumerate(dataset.windows):
            path = Path(window["path"])
            key = str(path) if mode == "file" else path.parent.name
            grouped_indices[key].append(idx)

        keys = sorted(grouped_indices)
        if len(keys) <= 1:
            print(f"[INFO] HM3D IL split mode={mode}: only one group; train=all, val=all")
            return dataset, dataset

        val_groups = max(1, int(len(keys) * val_split))
        val_groups = min(val_groups, len(keys) - 1)
        perm = torch.randperm(len(keys), generator=torch.Generator().manual_seed(self.split_seed)).tolist()
        val_keys = {keys[i] for i in perm[:val_groups]}

        train_indices = []
        val_indices = []
        for key in keys:
            if key in val_keys:
                val_indices.extend(grouped_indices[key])
            else:
                train_indices.extend(grouped_indices[key])

        print(
            f"[INFO] HM3D IL split mode={mode} train_groups={len(keys) - val_groups} "
            f"val_groups={val_groups} train_windows={len(train_indices)} "
            f"val_windows={len(val_indices)}"
        )
        return Subset(dataset, train_indices), Subset(dataset, val_indices)

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
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device),
            label_smoothing=self.label_smoothing,
        )

    def run(self, num_epochs=30, save_folder=None):
        save_folder = Path(save_folder).expanduser().resolve() if save_folder else None
        best_top1 = -1.0
        best_balanced_top1 = -1.0
        best_val_loss = float("inf")
        best_loss_path = None
        best_top1_path = None
        best_balanced_path = None
        epochs_without_loss_improvement = 0
        last_epoch = 0

        for epoch in range(1, int(num_epochs) + 1):
            last_epoch = epoch
            train_loss, train_acc = self._run_epoch(epoch)
            val_loss, val_metrics = self.evaluate(epoch)
            top1 = val_metrics.get(1, 0.0)
            balanced_top1 = float(val_metrics.get("balanced_top1", 0.0))

            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/top1", train_acc, epoch)
            self.writer.add_scalar("val/loss", val_loss, epoch)
            for k, value in val_metrics.items():
                tag = f"val/top{k}" if isinstance(k, int) else f"val/{k}"
                self.writer.add_scalar(tag, value, epoch)
            self.writer.flush()

            print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} train_top1={train_acc:.2f}% "
                f"val_loss={val_loss:.4f} val_top1={top1:.2f}% "
                f"val_balanced_top1={balanced_top1:.2f}%"
            )

            loss_improved = val_loss < (best_val_loss - self.min_delta)
            top1_improved = top1 > (best_top1 + self.min_delta)
            balanced_improved = balanced_top1 > (best_balanced_top1 + self.min_delta)
            checkpoint_top1 = max(best_top1, top1)

            if save_folder:
                save_folder.mkdir(parents=True, exist_ok=True)

                if loss_improved:
                    best_val_loss = val_loss
                    epochs_without_loss_improvement = 0
                    best_loss_path = save_folder / "hm3d_imitation_best_loss.pth"
                    self.save_checkpoint(best_loss_path, epoch, checkpoint_top1, best_val_loss)
                    self.save_checkpoint(save_folder / "hm3d_imitation_best.pth", epoch, checkpoint_top1, best_val_loss)
                else:
                    epochs_without_loss_improvement += 1

                if top1_improved:
                    best_top1 = top1
                    best_top1_path = save_folder / "hm3d_imitation_best_top1.pth"
                    self.save_checkpoint(best_top1_path, epoch, best_top1, val_loss)

                if balanced_improved:
                    best_balanced_top1 = balanced_top1
                    best_balanced_path = save_folder / "hm3d_imitation_best_balanced.pth"
                    self.save_checkpoint(best_balanced_path, epoch, checkpoint_top1, val_loss)

                self.save_checkpoint(save_folder / "hm3d_imitation_latest.pth", epoch, best_top1, val_loss)
            elif loss_improved:
                best_val_loss = val_loss
                epochs_without_loss_improvement = 0
            else:
                epochs_without_loss_improvement += 1

            if (
                self.early_stopping_patience > 0
                and epochs_without_loss_improvement >= self.early_stopping_patience
            ):
                print(
                    f"[INFO] Early stopping at epoch {epoch}: val_loss has not improved "
                    f"for {self.early_stopping_patience} epochs"
                )
                break

        if save_folder:
            final_path = save_folder / "hm3d_imitation_final.pth"
            self.save_checkpoint(final_path, last_epoch, best_top1, best_val_loss)
            print(f"[INFO] Final HM3D IL checkpoint: {final_path}")
            if best_loss_path:
                print(
                    f"[INFO] Best-loss HM3D IL checkpoint: {best_loss_path} "
                    f"(val_loss={best_val_loss:.4f})"
                )
            if best_top1_path:
                print(
                    f"[INFO] Best-top1 HM3D IL checkpoint: {best_top1_path} "
                    f"(top1={best_top1:.2f}%)"
                )
            if best_balanced_path:
                print(
                    f"[INFO] Best-balanced HM3D IL checkpoint: {best_balanced_path} "
                    f"(balanced_top1={best_balanced_top1:.2f}%)"
                )

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
        batches = 0
        with torch.no_grad():
            for batch_idx, (x_batch, last_act, tgt_act, lengths) in enumerate(
                tqdm(self.val_loader, desc="HM3D IL Val", leave=False)
            ):
                if (
                    self.val_epoch_samples is not None
                    and self.val_epoch_samples > 0
                    and batch_idx * self.val_loader.batch_size >= self.val_epoch_samples
                ):
                    break
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
                batches += 1

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return total_loss / max(batches, 1), self.compute_metrics(logits, targets)

    def compute_metrics(self, logits, targets):
        metrics = {}
        max_k = min(max(self.topk), logits.shape[-1])
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        for k in self.topk:
            kk = min(int(k), max_k)
            metrics[int(k)] = correct[:, :kk].any(dim=1).float().mean().item() * 100.0
        top1_pred = pred[:, 0]
        recalls = []
        for action_idx in range(logits.shape[-1]):
            target_mask = targets == action_idx
            pred_mask = top1_pred == action_idx
            target_count = int(target_mask.sum().item())
            pred_count = int(pred_mask.sum().item())
            if target_count > 0:
                recall = (top1_pred[target_mask] == action_idx).float().mean().item() * 100.0
                recalls.append(recall)
            else:
                recall = 0.0
            metrics[f"recall_action_{action_idx}"] = recall
            metrics[f"pred_frac_action_{action_idx}"] = pred_count / max(int(targets.numel()), 1) * 100.0
            metrics[f"target_frac_action_{action_idx}"] = target_count / max(int(targets.numel()), 1) * 100.0
        metrics["balanced_top1"] = float(np.mean(recalls)) if recalls else 0.0
        return metrics

    def save_checkpoint(self, path, epoch, best_top1, best_val_loss=None):
        payload = {
            "model_state_dict": self.agent.state_dict(),
            "epoch": int(epoch),
            "best_top1": float(best_top1),
            "best_val_loss": None if best_val_loss is None else float(best_val_loss),
            "label_smoothing": float(self.label_smoothing),
            "val_split_mode": self.val_split_mode,
            "train_sampling_mode": self.train_sampling_mode,
            "split_seed": int(self.split_seed),
            "navigation_config": self.agent.navigation_config,
            "num_actions": int(self.agent.num_actions),
        }
        torch.save(payload, str(path))
        print(f"[INFO] Saved HM3D IL checkpoint: {path}")
