"""
Parallel RL training runner for Habitat environment.

Key differences from serial runner:
1. Uses ParallelHabitatCollector to spawn N worker processes, each with independent HabitatEnv
2. Per step: collect obs from all workers -> batch forward -> distribute actions
3. Collects multiple trajectories per update (one per environment)
4. Maintains per-environment hidden states for LSTM

This should significantly speed up wall-clock training by:
- Parallelizing environment stepping (done on separate CPU cores)
- Batching policy inference to utilize GPU better
"""

import json
import sys
import os
import csv
from collections import deque
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from components.environments.parallel_habitat_collector import ParallelHabitatCollector
from components.perception.hm3d_labels import HM3D_REWARD_EXCLUDED_LABELS
from components.utils.observation import Observation
from components.utils.rollout_buffer import RolloutBuffer
from ImitationLearning.dataset.hm3d_feature_il_dataset import HM3DFeatureImitationLearningDataset


class ParallelHabitatRLTrainRunner:
    """
    Training runner for Habitat with parallel environment sampling.
    """
    
    def __init__(
        self,
        agent,
        dataset_root: str,
        config_file: str,
        num_workers: int = 4,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None,
        base_scene_ids: Optional[List[str]] = None,
        detection_service: Optional[Any] = None,
        env_config: Optional[Dict[str, Any]] = None,
        scene_count: Optional[int] = None,
        env_gpu_ids: Optional[List[int]] = None,
    ):
        self.agent = agent
        self.num_workers = num_workers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self.detection_service = detection_service
        self.env_config = env_config or {}
        self.base_scene_ids = [str(scene_id) for scene_id in (base_scene_ids or [])]
        self.scene_count = max(int(scene_count or len(base_scene_ids or []) or 1), 1)
        
        # Move agent to device
        self.agent.to(self.device)
        self.agent_config = agent.agent_config
        self.navigation_config = agent.navigation_config
        self.action_temperature = max(float(self.agent_config.get("action_temperature", 1.0)), 1e-3)
        self.rollout_action_mode = str(self.agent_config.get("rollout_action_mode", "sample")).strip().lower()
        self.rollout_epsilon = min(max(float(self.agent_config.get("rollout_epsilon", 0.0)), 0.0), 1.0)
        self.bc_coef = max(float(self.agent_config.get("bc_coef", 0.0)), 0.0)
        self.bc_min_coef = max(float(self.agent_config.get("bc_min_coef", 0.0)), 0.0)
        self.bc_coef_decay = bool(self.agent_config.get("bc_coef_decay", False))
        self.bc_updates_per_rl_update = max(int(self.agent_config.get("bc_updates_per_rl_update", 0)), 0)
        self.bc_loader = None
        self.bc_iter = None
        self.bc_criterion = None
        self.discovery_bonus_scale = float(self.env_config.get("discovery_bonus_scale", 1.0))
        self.det_score_thr = float(self.env_config.get("det_score_thr", 0.20))
        self.success_recall_threshold = float(self.env_config.get("success_recall_threshold", 1.00))
        self.success_min_coverage = float(self.env_config.get("success_min_coverage", 0.30))
        self.success_reward = float(self.env_config.get("success_reward", 10.0))
        self.reward_excluded_labels = {
            str(label) for label in self.env_config.get("reward_excluded_labels", HM3D_REWARD_EXCLUDED_LABELS)
        }
        self.reward_allow_semantic_iou_only = bool(self.env_config.get("reward_allow_semantic_iou_only", False))
        self.detection_log_interval = max(int(self.env_config.get("detection_log_interval", 100)), 1)
        self.update_batch_size = max(int(self.env_config.get("update_batch_size", 256)), 1)
        self.forward_escape_enabled = bool(self.env_config.get("forward_escape_enabled", True))
        self.forward_stuck_window = max(int(self.env_config.get("forward_stuck_window", 4)), 1)
        self.forward_stuck_min_distance = max(float(self.env_config.get("forward_stuck_min_distance", 0.02)), 0.0)
        self.forward_escape_cooldown_steps = max(int(self.env_config.get("forward_escape_cooldown_steps", 8)), 0)
        self.forward_blocked_heading_ttl = max(int(self.env_config.get("forward_blocked_heading_ttl", 24)), 0)
        self.no_move_escape_window = max(int(self.env_config.get("no_move_escape_window", 20)), 1)
        self.no_move_turn_escape_window = max(int(self.env_config.get("no_move_turn_escape_window", 3)), 1)
        self.turn_loop_escape_enabled = bool(self.env_config.get("turn_loop_escape_enabled", True))
        self.turn_loop_escape_window = max(int(self.env_config.get("turn_loop_escape_window", 12)), 1)
        self.turn_loop_escape_forward_min_prob = max(
            float(self.env_config.get("turn_loop_escape_forward_min_prob", 0.0)),
            0.0,
        )
        self._detection_log_counter = 0
        
        # Create parallel environment collector
        env_kwargs = {
            "render": False,
            "width": 300,
            "height": 300,
            "use_detector": False,
            "detector": None,
            "det_score_thr": self.det_score_thr,
            "max_actions": int(self.agent_config.get("num_steps", 4000)),
        }
        if save_dir:
            env_kwargs["save_debug_path"] = save_dir
        for key in [
            "width",
            "height",
            "instance_merge_dist",
            "coverage_cell_size",
            "nav_sample_points",
            "topdown_meters_per_pixel",
            "agent_radius",
            "agent_height",
            "agent_max_climb",
            "navmesh_cell_height",
            "navmesh_cell_size",
            "fill_position_from_gt",
            "rho",
            "coverage_bonus_scale",
            "visible_coverage_bonus_scale",
            "new_cell_reward",
            "discovery_bonus_scale",
            "collision_penalty",
            "no_progress_window",
            "no_progress_penalty",
            "no_progress_min_coverage_delta",
            "turn_penalty",
            "stationary_turn_window",
            "stationary_turn_penalty",
            "stationary_turn_min_distance",
            "stationary_turn_termination_window",
            "dino_max_box_area_ratio",
            "dino_max_box_aspect_ratio",
            "gt_validation_iou_threshold",
            "gt_validation_mode",
            "success_recall_threshold",
            "success_min_coverage",
            "success_reward",
            "reward_allow_semantic_iou_only",
            "reward_excluded_labels",
            "visible_coverage_stride",
            "visible_coverage_max_depth",
            "visible_coverage_min_ray_step",
            "random_start_yaw_degrees",
            "max_actions",
            "save_debug_interval",
            "save_debug_path",
        ]:
            if key in self.env_config and self.env_config[key] is not None:
                env_kwargs[key] = self.env_config[key]
        
        # Increase timeout to allow for slower environment initialization
        self.env_collector = ParallelHabitatCollector(
            num_workers=num_workers,
            dataset_root=dataset_root,
            config_file=config_file,
            base_scene_ids=base_scene_ids,
            env_kwargs=env_kwargs,
            timeout=float(self.env_config.get("worker_init_timeout", 900.0)),
            env_gpu_ids=env_gpu_ids,
        )
        
        # Per-environment state tracking
        self.num_envs = num_workers
        self.per_env_buffers: List[RolloutBuffer] = [
            RolloutBuffer(self.agent_config.get("num_steps", 4000))
            for _ in range(num_workers)
        ]
        self.per_env_last_actions = [-1] * num_workers
        self.per_env_forward_stuck_counts = [0] * num_workers
        self.per_env_forward_escape_cooldowns = [0] * num_workers
        self.per_env_blocked_forward_headings = [dict() for _ in range(num_workers)]
        self.per_env_no_move_counts = [0] * num_workers
        self.per_env_turn_loop_counts = [0] * num_workers
        self.per_env_scene_indices = [i % self.scene_count for i in range(num_workers)]
        self.per_env_discovered_objects = [set() for _ in range(num_workers)]
        self.per_env_discovered_instances = [set() for _ in range(num_workers)]
        self.per_env_discovered_gt_ids = [set() for _ in range(num_workers)]
        self.per_env_prev_score = [0.0] * num_workers
        self.per_env_episode_counts = [0] * num_workers
        self.scene_episode_counts = [0] * self.scene_count
        self.per_env_scene_epochs = [1] * num_workers
        self.per_env_scene_assignment_numbers = [0] * num_workers
        self.next_scene_assignment = 0
        self.per_env_hidden_states = [
            {
                "lssg": None,
                "gssg": None,
                "policy": None,
            }
            for _ in range(num_workers)
        ]
        self.transformer_context_len = max(
            int(
                self.agent_config.get(
                    "transformer_context_len",
                    self.agent_config.get("bc_seq_len", 16),
                )
            ),
            1,
        )
        self.per_env_obs_history = [deque(maxlen=self.transformer_context_len) for _ in range(num_workers)]
        self.per_env_last_action_history = [deque(maxlen=self.transformer_context_len) for _ in range(num_workers)]
        self.per_env_action_diagnostics = [{} for _ in range(num_workers)]
        
        # Config
        self.total_episodes = self.agent_config.get("episodes", 500)
        self.num_steps_per_rollout = self.agent_config.get("num_steps", 4000)
        self.log_buffer_size = 40
        self.episode_step_counters = [0] * num_workers
        
        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_name = agent.get_agent_info().get("Agent Name", "Agent").replace(" ", "_")
        if self.navigation_config.get("use_transformer"):
            agent_name += "_Transformer"
        else:
            agent_name += "_LSTM"
        log_dir = f"RL_training/runs/{agent_name}_{timestamp}_parallel_{num_workers}w"
        self.writer = SummaryWriter(log_dir)
        print(f"[INFO] TensorBoard logs: {log_dir}")
        
        full_config = {
            "agent_config": self.agent_config,
            "navigation_config": self.navigation_config,
            "num_workers": num_workers,
        }
        self.writer.add_text("config", json.dumps(full_config, indent=2), 0)
        
        self.ep_info_buffer = deque(maxlen=self.log_buffer_size)
        self.global_step = 0
        self.was_interrupted = False
        self._init_bc_regularization()

    def _init_bc_regularization(self):
        if self.bc_coef <= 0.0 or self.bc_updates_per_rl_update <= 0:
            return
        data_dir = self.agent_config.get("bc_data_dir")
        if not data_dir:
            print("[INFO] BC regularization disabled: bc_data_dir is not set")
            return

        try:
            dataset = HM3DFeatureImitationLearningDataset(
                data_dir,
                seq_len=int(self.agent_config.get("bc_seq_len", 16)),
                feature_key=str(self.agent_config.get("bc_feature_key", "rgb_features")),
            )
        except Exception as exc:
            print(f"[WARN] BC regularization disabled: failed to load {data_dir}: {exc}")
            return

        if int(dataset.feature_dim) != int(self.navigation_config["rgb_dim"]):
            print(
                "[WARN] BC regularization disabled: feature_dim "
                f"{dataset.feature_dim} != navigation rgb_dim {self.navigation_config['rgb_dim']}"
            )
            return

        batch_size = max(int(self.agent_config.get("bc_batch_size", 256)), 1)
        num_workers = max(int(self.agent_config.get("bc_num_workers", 0)), 0)
        self.bc_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataset.seq_collate,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        self.bc_iter = iter(self.bc_loader)
        self.bc_criterion = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=float(self.agent_config.get("bc_label_smoothing", 0.0)),
        )
        print(
            "[INFO] BC regularization enabled: "
            f"files={len(dataset.files)} windows={len(dataset)} batch_size={batch_size} "
            f"coef={self.bc_coef:.4f} min_coef={self.bc_min_coef:.4f} "
            f"updates_per_rl_update={self.bc_updates_per_rl_update}"
        )

    def _current_bc_coef(self) -> float:
        if self.bc_loader is None or self.bc_coef <= 0.0:
            return 0.0
        if not self.bc_coef_decay:
            return self.bc_coef
        progress = 0.0
        if self.total_episodes:
            progress = min(float(getattr(self, "next_scene_assignment", 0)) / float(max(self.total_episodes, 1)), 1.0)
        coef = self.bc_coef - (self.bc_coef - self.bc_min_coef) * progress
        return max(float(coef), self.bc_min_coef)

    def _next_bc_batch(self):
        if self.bc_loader is None:
            return None
        try:
            return next(self.bc_iter)
        except StopIteration:
            self.bc_iter = iter(self.bc_loader)
            return next(self.bc_iter)

    def _perform_bc_regularization(self):
        if self.bc_loader is None or self.bc_criterion is None:
            return None

        coef = self._current_bc_coef()
        if coef <= 0.0:
            return None

        losses = []
        accs = []
        self.agent.train()
        for _ in range(self.bc_updates_per_rl_update):
            batch = self._next_bc_batch()
            if batch is None:
                break
            x_batch, last_actions, target_actions, _lengths = batch
            last_actions = last_actions.to(self.device)
            target_actions = target_actions.to(self.device)
            if isinstance(x_batch.get("rgb_features"), torch.Tensor):
                x_batch["rgb_features"] = x_batch["rgb_features"].to(self.device, non_blocking=True)

            state_seq, _, _ = self.agent.encoder.forward_seq(x_batch, last_actions)
            pad_mask = target_actions.eq(-100) if self.navigation_config.get("use_transformer") else None
            logits, _value, _hidden = self.agent.policy(state_seq, pad_mask=pad_mask)
            loss_raw = self.bc_criterion(logits.reshape(-1, logits.size(-1)), target_actions.reshape(-1))
            loss = coef * loss_raw

            self.agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.agent.optimizer.step()

            with torch.no_grad():
                valid = target_actions.ne(-100)
                if valid.any():
                    pred = logits.argmax(dim=-1)
                    acc = pred.eq(target_actions).masked_select(valid).float().mean().item()
                    accs.append(acc)
            losses.append(float(loss_raw.item()))

        if not losses:
            return None
        return {
            "bc_loss": float(np.mean(losses)),
            "bc_acc": float(np.mean(accs)) if accs else 0.0,
            "bc_coef": float(coef),
        }

    def _scene_name(self, scene_index: int) -> str:
        if 0 <= scene_index < len(self.base_scene_ids):
            return self.base_scene_ids[scene_index]
        return f"scene_{scene_index + 1}"

    def _reset_scene_scheduler(self):
        """Reset the global scene cursor so one epoch visits every scene once."""
        self.next_scene_assignment = 0
        self.per_env_scene_indices = [0] * self.num_workers
        self.per_env_scene_epochs = [1] * self.num_workers
        self.per_env_scene_assignment_numbers = [0] * self.num_workers

    def _claim_next_scene(self):
        """
        Return the next globally scheduled scene.

        Assignment order is scene 1..N for epoch 1, then scene 1..N for
        epoch 2, regardless of which worker finished the previous episode.
        """
        assignment_index = self.next_scene_assignment
        self.next_scene_assignment += 1
        scene_index = assignment_index % self.scene_count
        epoch_no = assignment_index // self.scene_count + 1
        assignment_no = assignment_index + 1
        return scene_index, epoch_no, assignment_no

    def _claim_scene_for_env(self, env_id: int) -> int:
        scene_index, epoch_no, assignment_no = self._claim_next_scene()
        self.per_env_scene_indices[env_id] = scene_index
        self.per_env_scene_epochs[env_id] = epoch_no
        self.per_env_scene_assignment_numbers[env_id] = assignment_no
        return scene_index

    def _episode_tag(self, env_id: int, scene_index: int) -> str:
        scene_name = self._scene_name(scene_index)
        local_ep_no = self.per_env_episode_counts[env_id] + 1
        epoch_no = self.per_env_scene_epochs[env_id]
        assignment_no = self.per_env_scene_assignment_numbers[env_id]
        return (
            f"epoch_{epoch_no:04d}_assign_{assignment_no:05d}_"
            f"env_{env_id + 1:02d}_worker_ep_{local_ep_no:04d}_"
            f"scene_{scene_index + 1:02d}_{scene_name}"
        )
    
    def _build_batch_dict(self, obs_list: List[Observation]) -> Dict[str, List[List[Any]]]:
        """Convert per-env observations into a [B, T=1] batch dict."""
        return {
            "rgb": [[obs.state[0]] for obs in obs_list],
            "lssg": [[obs.state[1]] for obs in obs_list],
            "gssg": [[obs.state[2]] for obs in obs_list],
            "occupancy": [[obs.state[3]] for obs in obs_list],
            "agent_pos": [[obs.info.get("agent_pos", obs.info.get("agent_position", None))] for obs in obs_list],
        }

    def _reset_transformer_history(self, env_id: Optional[int] = None):
        if env_id is None:
            for obs_history, action_history in zip(
                self.per_env_obs_history,
                self.per_env_last_action_history,
            ):
                obs_history.clear()
                action_history.clear()
            return

        self.per_env_obs_history[env_id].clear()
        self.per_env_last_action_history[env_id].clear()

    def _build_transformer_history_batch(self, obs_list: List[Observation]):
        """
        Build a padded [B, T<=context] online history batch for Transformer policy inference.

        The Transformer IL policy was trained on temporal windows, so online inference
        should see the same recent history instead of a single frame. Padding uses -100,
        matching the IL dataset padding sentinel, and masks padded tokens downstream.
        """
        histories = []
        action_histories = []
        max_t = 1

        for env_id, obs in enumerate(obs_list):
            self.per_env_obs_history[env_id].append(obs)
            self.per_env_last_action_history[env_id].append(self.per_env_last_actions[env_id])
            obs_history = list(self.per_env_obs_history[env_id])
            action_history = list(self.per_env_last_action_history[env_id])
            histories.append(obs_history)
            action_histories.append(action_history)
            max_t = max(max_t, len(obs_history))

        rgb_batch = []
        lssg_batch = []
        gssg_batch = []
        occ_batch = []
        pos_batch = []
        masks = []
        last_actions = torch.full((len(obs_list), max_t), -100, dtype=torch.long, device=self.device)

        for batch_idx, obs_history in enumerate(histories):
            length = len(obs_history)
            pad_len = max_t - length
            rgb_seq = [obs.state[0] for obs in obs_history] + [0] * pad_len
            lssg_seq = [obs.state[1] for obs in obs_history] + [None] * pad_len
            gssg_seq = [obs.state[2] for obs in obs_history] + [None] * pad_len
            occ_seq = [obs.state[3] for obs in obs_history] + [0] * pad_len
            pos_seq = [
                obs.info.get("agent_pos", obs.info.get("agent_position", None))
                for obs in obs_history
            ] + [None] * pad_len
            mask = [1] * length + [0] * pad_len

            rgb_batch.append(rgb_seq)
            lssg_batch.append(lssg_seq)
            gssg_batch.append(gssg_seq)
            occ_batch.append(occ_seq)
            pos_batch.append(pos_seq)
            masks.append(mask)

            last_actions[batch_idx, :length] = torch.as_tensor(
                action_histories[batch_idx],
                dtype=torch.long,
                device=self.device,
            )

        batch_dict = {
            "rgb": rgb_batch,
            "lssg": lssg_batch,
            "gssg": gssg_batch,
            "occupancy": occ_batch,
            "agent_pos": pos_batch,
            "lssg_mask": masks,
            "gssg_mask": masks,
        }
        pad_mask = ~torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        last_indices = torch.as_tensor([len(history) - 1 for history in histories], dtype=torch.long, device=self.device)
        return batch_dict, last_actions, pad_mask, last_indices

    def _stack_lstm_hidden(self, hidden_list, hidden_size: int):
        """Stack per-env LSTM hidden states into [num_layers, B, H]."""
        num_layers = 2
        batch_size = len(hidden_list)
        h_list = []
        c_list = []
        for hidden in hidden_list:
            if hidden is None:
                h_list.append(torch.zeros(num_layers, 1, hidden_size, device=self.device))
                c_list.append(torch.zeros(num_layers, 1, hidden_size, device=self.device))
            else:
                h, c = hidden
                h_list.append(h.to(self.device))
                c_list.append(c.to(self.device))
        h = torch.cat(h_list, dim=1) if batch_size > 0 else torch.zeros(num_layers, 0, hidden_size, device=self.device)
        c = torch.cat(c_list, dim=1) if batch_size > 0 else torch.zeros(num_layers, 0, hidden_size, device=self.device)
        return h, c

    def _split_lstm_hidden(self, hidden):
        """Split batched LSTM hidden states back into per-env tuples."""
        if hidden is None:
            return [None] * self.num_workers
        h, c = hidden
        return [(h[:, i:i+1, :].contiguous(), c[:, i:i+1, :].contiguous()) for i in range(h.size(1))]

    def _build_detection_key(self, detection: Dict[str, Any], label: str):
        box = detection.get("bbox", detection.get("box"))
        if box is not None and len(box) == 4:
            cx = 0.5 * (float(box[0]) + float(box[2]))
            cy = 0.5 * (float(box[1]) + float(box[3]))
            bw = float(box[2]) - float(box[0])
            bh = float(box[3]) - float(box[1])
            return (label, round(cx, 1), round(cy, 1), round(bw, 1), round(bh, 1))
        return (label,)

    def _run_detection_batch(self, obs_list: List[Observation]):
        if not obs_list:
            return []

        rgb_batch = [obs.state[0] for obs in obs_list]

        if self.detection_service is not None:
            try:
                return self.detection_service.detect_batch(rgb_batch)
            except Exception as exc:
                print(f"[WARN] DINO service batch failed, falling back to local call: {exc}")

        return [[] for _ in rgb_batch]

    def _summarize_detection_batches(self, detection_batches):
        """Print low-frequency DINO validation stats so zero-score runs are diagnosable."""
        if not detection_batches:
            return
        self._detection_log_counter += 1
        if self._detection_log_counter % self.detection_log_interval != 0:
            return

        total = 0
        valid = 0
        low_score = 0
        excluded = 0
        semantic_iou_only = 0
        reject_reasons = {}
        labels = {}

        for detections in detection_batches:
            for det in detections or []:
                total += 1
                label = str(det.get("canonical_label") or det.get("label") or "unknown")
                labels[label] = labels.get(label, 0) + 1
                if float(det.get("score", 0.0)) < self.det_score_thr:
                    low_score += 1
                    continue
                if label in self.reward_excluded_labels:
                    excluded += 1
                    continue
                if det.get("gt_match_mode") == "semantic_iou_only":
                    semantic_iou_only += 1
                if det.get("is_gt_valid") is True:
                    valid += 1
                else:
                    reason = str(det.get("reject_reason") or "not_gt_valid")
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

        top_reasons = sorted(reject_reasons.items(), key=lambda item: item[1], reverse=True)[:4]
        top_labels = sorted(labels.items(), key=lambda item: item[1], reverse=True)[:5]
        print(
            "[DINO] "
            f"step={self.global_step} total={total} valid={valid} low_score={low_score} "
            f"excluded={excluded} semantic_iou_only={semantic_iou_only} "
            f"top_rejects={top_reasons} top_labels={top_labels}"
        )

    def _apply_detection_reward(self, env_id: int, obs: Observation, detections):
        """Convert DINO detections into GT-object recall and discovery reward."""
        if obs is None:
            return 0.0, 0.0

        target_gt_ids = set()
        if obs.info:
            try:
                target_gt_ids = {int(item) for item in obs.info.get("scene_reward_gt_ids", [])}
            except Exception:
                target_gt_ids = set()

        for det in detections or []:
            if float(det.get("score", 0.0)) < self.det_score_thr:
                continue
            if self.detection_service is not None and det.get("is_gt_valid") is not True:
                continue
            if not self.reward_allow_semantic_iou_only and det.get("gt_match_mode") == "semantic_iou_only":
                continue

            label = str(det.get("canonical_label") or det.get("label") or "unknown")
            if label in self.reward_excluded_labels:
                continue

            try:
                gt_semantic_id = int(det.get("gt_semantic_id"))
            except Exception:
                continue
            if not target_gt_ids or gt_semantic_id not in target_gt_ids:
                continue

            self.per_env_discovered_objects[env_id].add(label)
            self.per_env_discovered_gt_ids[env_id].add(gt_semantic_id)
            # Kept for diagnostics only; score/reward use GT semantic IDs.
            self.per_env_discovered_instances[env_id].add(self._build_detection_key(det, label))

        discovered_gt_count = len(self.per_env_discovered_gt_ids[env_id])
        target_gt_count = len(target_gt_ids)
        discovered_instance_count = len(self.per_env_discovered_instances[env_id])
        discovered_label_count = len(self.per_env_discovered_objects[env_id])
        current_score = (
            min(discovered_gt_count / target_gt_count, 1.0)
            if target_gt_count > 0
            else 0.0
        )
        score_gain = current_score - self.per_env_prev_score[env_id]
        self.per_env_prev_score[env_id] = current_score

        if obs.info is None:
            obs.info = {}
        obs.info["score"] = float(current_score)
        obs.info["object_recall"] = float(current_score)
        obs.info["num_discovered"] = discovered_gt_count
        obs.info["num_discovered_gt"] = discovered_gt_count
        obs.info["num_discovered_labels"] = discovered_label_count
        obs.info["num_discovered_instances"] = discovered_instance_count
        obs.info["scene_reward_gt_count"] = int(target_gt_count)

        return current_score, self.discovery_bonus_scale * score_gain

    def _is_success(self, obs: Observation, score: float) -> bool:
        if obs is None or obs.info is None:
            return False
        coverage = float(obs.info.get("coverage", 0.0) or 0.0)
        return (
            score >= self.success_recall_threshold
            and coverage >= self.success_min_coverage
        )
    
    def _get_batch_actions(self, obs_list: List[Observation], deterministic: bool = False):
        """
        Get actions for all environments in parallel via a single batch forward.
        """
        if not obs_list:
            return [], np.array([])

        def yaw_bucket(obs: Observation) -> Optional[int]:
            if not obs.info:
                return None
            yaw = obs.info.get("yaw_deg")
            if yaw is None:
                return None
            try:
                # Habitat turns are 30 degrees here, so bucketing to 30-degree bins
                # preserves the actionable heading while tolerating float noise.
                return int(round(float(yaw) / 30.0)) % 12
            except (TypeError, ValueError):
                return None

        with torch.no_grad():
            if self.navigation_config.get("use_transformer"):
                batch_dict, last_actions, pad_mask, last_indices = self._build_transformer_history_batch(obs_list)
                state_seq, _, _ = self.agent.encoder(batch_dict, last_actions)
                logits, value, _ = self.agent.policy(
                    state_seq,
                    hidden=None,
                    pad_mask=pad_mask,
                )
                policy_hidden = None
                lssg_hidden_out = [None] * len(obs_list)
                gssg_hidden_out = [None] * len(obs_list)
            else:
                batch_dict = self._build_batch_dict(obs_list)
                last_actions = torch.tensor(
                    [[action] for action in self.per_env_last_actions],
                    dtype=torch.long,
                    device=self.device,
                )
                hidden_size = self.agent.encoder.lssg_encoder.lstm.hidden_size
                lssg_hidden = self._stack_lstm_hidden(
                    [hidden["lssg"] for hidden in self.per_env_hidden_states],
                    hidden_size,
                )
                gssg_hidden = self._stack_lstm_hidden(
                    [hidden["gssg"] for hidden in self.per_env_hidden_states],
                    hidden_size,
                )
                policy_hidden = self._stack_lstm_hidden(
                    [hidden["policy"] for hidden in self.per_env_hidden_states],
                    self.agent.policy.core.hidden_size,
                )

                state_seq, new_lssg, new_gssg = self.agent.encoder(
                    batch_dict,
                    last_actions,
                    lssg_hidden=lssg_hidden,
                    gssg_hidden=gssg_hidden,
                )
                lssg_hidden_out = self._split_lstm_hidden(new_lssg)
                gssg_hidden_out = self._split_lstm_hidden(new_gssg)

                logits, value, new_policy_hidden = self.agent.policy(
                    state_seq,
                    hidden=policy_hidden,
                )

            if self.navigation_config.get("use_transformer"):
                batch_indices = torch.arange(len(obs_list), dtype=torch.long, device=self.device)
                logits = logits[batch_indices, last_indices, :]
                if value is not None:
                    value = value[batch_indices, last_indices]
            else:
                policy_hidden_out = self._split_lstm_hidden(new_policy_hidden)
                for env_id in range(self.num_workers):
                    self.per_env_hidden_states[env_id]["lssg"] = lssg_hidden_out[env_id]
                    self.per_env_hidden_states[env_id]["gssg"] = gssg_hidden_out[env_id]
                    self.per_env_hidden_states[env_id]["policy"] = policy_hidden_out[env_id]
                logits = logits[:, -1, :]
                if value is not None:
                    value = value[:, -1]

            probs = torch.softmax(logits / self.action_temperature, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
            if deterministic:
                actions = torch.argmax(probs, dim=-1).tolist()
            elif self.rollout_action_mode in {"argmax", "greedy"}:
                actions = torch.argmax(probs, dim=-1).tolist()
            elif self.rollout_action_mode in {"argmax_epsilon", "epsilon_argmax", "epsilon_greedy"}:
                from torch.distributions import Categorical
                greedy_actions = torch.argmax(probs, dim=-1)
                if self.rollout_epsilon > 0.0:
                    dist = Categorical(probs=probs)
                    sampled_actions = dist.sample()
                    explore = torch.rand(len(obs_list), device=probs.device) < self.rollout_epsilon
                    actions_tensor = torch.where(explore, sampled_actions, greedy_actions)
                else:
                    actions_tensor = greedy_actions
                actions = actions_tensor.tolist()
            else:
                from torch.distributions import Categorical
                dist = Categorical(probs=probs)
                actions = dist.sample().tolist()
            raw_actions = [int(action) for action in actions]
            if self.forward_escape_enabled:
                for env_id, action in enumerate(actions):
                    blocked_headings = (
                        self.per_env_blocked_forward_headings[env_id]
                        if env_id < len(self.per_env_blocked_forward_headings)
                        else {}
                    )
                    heading_bucket = yaw_bucket(obs_list[env_id])
                    heading_blocked = (
                        heading_bucket is not None
                        and blocked_headings.get(heading_bucket, 0) > 0
                    )
                    if (
                        int(action) == 2
                        and env_id < len(self.per_env_forward_stuck_counts)
                        and (
                            self.per_env_forward_stuck_counts[env_id] >= self.forward_stuck_window
                            or self.per_env_forward_escape_cooldowns[env_id] > 0
                            or heading_blocked
                        )
                    ):
                        # If repeated forward actions are not moving the agent, rotate for a
                        # short cooldown so the next forward is tried from a different heading.
                        actions[env_id] = 0 if probs[env_id, 0] >= probs[env_id, 1] else 1
            values = value.tolist() if value is not None else [0.0] * len(actions)
            probs_cpu = probs.detach().cpu().numpy()
            entropy_cpu = entropy.detach().cpu().numpy()
            self.per_env_action_diagnostics = []
            for env_id, action in enumerate(actions):
                policy_action = raw_actions[env_id] if env_id < len(raw_actions) else int(action)
                overridden = int(action) != policy_action
                heading_bucket = yaw_bucket(obs_list[env_id])
                heading_blocked = (
                    heading_bucket is not None
                    and env_id < len(self.per_env_blocked_forward_headings)
                    and self.per_env_blocked_forward_headings[env_id].get(heading_bucket, 0) > 0
                )
                if (
                    self.turn_loop_escape_enabled
                    and policy_action in (0, 1)
                    and env_id < len(self.per_env_turn_loop_counts)
                    and self.per_env_turn_loop_counts[env_id] >= (
                        self.no_move_turn_escape_window
                        if (
                            env_id < len(self.per_env_no_move_counts)
                            and self.per_env_no_move_counts[env_id] >= self.no_move_escape_window
                        )
                        else self.turn_loop_escape_window
                    )
                    and probs_cpu[env_id, 2] >= self.turn_loop_escape_forward_min_prob
                    and not heading_blocked
                ):
                    action = 2
                    actions[env_id] = action
                    overridden = True
                row = {
                    "action": int(action),
                    "policy_action": policy_action,
                    "action_overridden": int(overridden),
                    "forward_escape_cooldown": int(self.per_env_forward_escape_cooldowns[env_id])
                    if env_id < len(self.per_env_forward_escape_cooldowns)
                    else 0,
                    "forward_heading_blocked": int(heading_blocked),
                    "turn_loop_count": int(self.per_env_turn_loop_counts[env_id])
                    if env_id < len(self.per_env_turn_loop_counts)
                    else 0,
                    "no_move_count": int(self.per_env_no_move_counts[env_id])
                    if env_id < len(self.per_env_no_move_counts)
                    else 0,
                    "last_action": int(self.per_env_last_actions[env_id]) if env_id < len(self.per_env_last_actions) else -1,
                    "prob_left": float(probs_cpu[env_id, 0]) if probs_cpu.shape[1] > 0 else 0.0,
                    "prob_right": float(probs_cpu[env_id, 1]) if probs_cpu.shape[1] > 1 else 0.0,
                    "prob_forward": float(probs_cpu[env_id, 2]) if probs_cpu.shape[1] > 2 else 0.0,
                    "entropy": float(entropy_cpu[env_id]),
                }
                self.per_env_action_diagnostics.append(row)

        return actions, np.array(values)
    
    def run(self):
        """
        Main training loop with parallel environment sampling.
        """
        use_tqdm = sys.stderr.isatty()
        pbar = None
        if use_tqdm:
            pbar = tqdm(total=self.total_episodes, desc="Episodes", ncols=160, leave=False)
        
        episode_count = 0
        total_steps_collected = 0
        rollout_steps_collected = 0
        self._reset_scene_scheduler()
        for env_id in range(self.num_workers):
            self._claim_scene_for_env(env_id)
        self.per_env_episode_counts = [0] * self.num_workers
        self.scene_episode_counts = [0] * self.scene_count
        
        # Initialize environments
        print("[INFO] Initializing parallel environments...")
        try:
            initial_scene_ids = [str(scene_index + 1) for scene_index in self.per_env_scene_indices]
            initial_episode_tags = [
                self._episode_tag(env_id, scene_index)
                for env_id, scene_index in enumerate(self.per_env_scene_indices)
            ]
            obs_list = self.env_collector.reset_all(
                scene_ids=initial_scene_ids,
                random_start=True,
                episode_tags=initial_episode_tags,
            )
        except Exception as e:
            print(f"[ERROR] Failed to reset environments: {e}")
            self.env_collector.close()
            return
        
        # Reset buffers and hidden states
        for buffer in self.per_env_buffers:
            buffer.clear()
        self.per_env_last_actions = [-1] * self.num_workers
        self.per_env_forward_stuck_counts = [0] * self.num_workers
        self.per_env_forward_escape_cooldowns = [0] * self.num_workers
        self.per_env_blocked_forward_headings = [dict() for _ in range(self.num_workers)]
        self.per_env_no_move_counts = [0] * self.num_workers
        self.per_env_turn_loop_counts = [0] * self.num_workers
        self.per_env_discovered_objects = [set() for _ in range(self.num_workers)]
        self.per_env_discovered_instances = [set() for _ in range(self.num_workers)]
        self.per_env_discovered_gt_ids = [set() for _ in range(self.num_workers)]
        self.per_env_prev_score = [0.0] * self.num_workers
        for hidden_dict in self.per_env_hidden_states:
            hidden_dict["lssg"] = None
            hidden_dict["gssg"] = None
            hidden_dict["policy"] = None
        self._reset_transformer_history()
        self.episode_step_counters = [0] * self.num_workers
        
        step_in_rollout = 0
        max_score = 0.0
        
        try:
            while episode_count < self.total_episodes:
                # Collect rollout data from all environments
                step_in_rollout += 1
                
                # Get actions for all environments
                prev_obs_list = obs_list
                prev_last_actions = list(self.per_env_last_actions)
                prev_hidden_states = [
                    {
                        "lssg": hidden["lssg"],
                        "gssg": hidden["gssg"],
                        "policy": hidden["policy"],
                    }
                    for hidden in self.per_env_hidden_states
                ]
                actions, values = self._get_batch_actions(obs_list)
                
                # Step all environments
                try:
                    obs_list = self.env_collector.step_all(actions, self.per_env_action_diagnostics)
                except Exception as e:
                    print(f"[ERROR] Parallel step failed: {e}")
                    break

                detection_batches = self._run_detection_batch(obs_list)
                if self.detection_service is not None:
                    try:
                        detection_batches = self.env_collector.annotate_detections_all(detection_batches)
                    except Exception as exc:
                        print(f"[WARN] Failed to annotate DINO detections in workers: {exc}")
                self._summarize_detection_batches(detection_batches)
                
                # Collect transitions for each environment
                for env_id, obs in enumerate(obs_list):
                    # Extract info
                    reward = obs.reward
                    terminated = obs.terminated
                    truncated = obs.truncated
                    done = terminated or truncated

                    score, det_bonus = self._apply_detection_reward(env_id, obs, detection_batches[env_id] if env_id < len(detection_batches) else [])
                    reward = float(reward or 0.0) + float(det_bonus)
                    if obs.info is None:
                        obs.info = {}
                    obs.info["score"] = float(score)

                    success = self._is_success(obs, score)
                    obs.info["success"] = bool(success)
                    if success:
                        obs.info["success_reason"] = "object_recall_and_coverage"
                        reward += self.success_reward
                        if not done:
                            terminated = True
                            done = True
                            obs.terminated = True
                    obs.reward = reward
                    
                    # Add to this env's buffer
                    buffer = self.per_env_buffers[env_id]
                    prev_obs = prev_obs_list[env_id]
                    
                    # Store the state that produced this action. The reward/done
                    # comes from the next observation returned by env.step().
                    buffer.add(
                        state=prev_obs.state,
                        action=actions[env_id],
                        reward=reward,
                        done=done,
                        hiddens=prev_hidden_states[env_id],
                        last_action=prev_last_actions[env_id],
                        agent_position=prev_obs.info.get("agent_position", prev_obs.info.get("agent_pos", None)),
                    )
                    
                    self.per_env_last_actions[env_id] = actions[env_id]
                    moved_distance = 0.0
                    if obs.info:
                        moved_distance = float(obs.info.get("moved_distance", 0.0) or 0.0)
                    if moved_distance <= self.forward_stuck_min_distance:
                        self.per_env_no_move_counts[env_id] += 1
                    else:
                        self.per_env_no_move_counts[env_id] = 0
                    blocked_headings = self.per_env_blocked_forward_headings[env_id]
                    for heading in list(blocked_headings):
                        blocked_headings[heading] -= 1
                        if blocked_headings[heading] <= 0:
                            del blocked_headings[heading]
                    if int(actions[env_id]) == 2 and moved_distance <= self.forward_stuck_min_distance:
                        self.per_env_forward_stuck_counts[env_id] += 1
                        if self.per_env_forward_stuck_counts[env_id] >= self.forward_stuck_window:
                            self.per_env_forward_escape_cooldowns[env_id] = self.forward_escape_cooldown_steps
                        if self.forward_blocked_heading_ttl > 0 and obs.info:
                            yaw = obs.info.get("yaw_deg")
                            if yaw is not None:
                                try:
                                    blocked_headings[int(round(float(yaw) / 30.0)) % 12] = self.forward_blocked_heading_ttl
                                except (TypeError, ValueError):
                                    pass
                    else:
                        self.per_env_forward_stuck_counts[env_id] = 0
                    if self.per_env_forward_escape_cooldowns[env_id] > 0 and int(actions[env_id]) != 2:
                        self.per_env_forward_escape_cooldowns[env_id] -= 1
                    if int(actions[env_id]) in (0, 1):
                        self.per_env_turn_loop_counts[env_id] += 1
                    else:
                        self.per_env_turn_loop_counts[env_id] = 0
                    self.episode_step_counters[env_id] += 1
                    total_steps_collected += 1
                    rollout_steps_collected += 1
                    
                    # Track episode info
                    if obs.info:
                        score = obs.info.get("score", 0.0)
                        coverage = obs.info.get("coverage", 0.0)
                        path_coverage = obs.info.get("path_coverage", 0.0)
                        visible_coverage = obs.info.get("visible_coverage", 0.0)
                        self.writer.add_scalar(f"env_{env_id}/reward", reward, self.global_step)
                        self.writer.add_scalar(f"env_{env_id}/score", score, self.global_step)
                        self.writer.add_scalar(f"env_{env_id}/coverage", coverage, self.global_step)
                        self.writer.add_scalar(f"env_{env_id}/path_coverage", path_coverage, self.global_step)
                        self.writer.add_scalar(f"env_{env_id}/visible_coverage", visible_coverage, self.global_step)
                    
                    # Reset if done
                    if done:
                        if obs.info:
                            ep_score = obs.info.get("score", 0.0)
                            ep_coverage = obs.info.get("coverage", 0.0)
                            ep_path_coverage = obs.info.get("path_coverage", 0.0)
                            ep_visible_coverage = obs.info.get("visible_coverage", 0.0)
                            ep_discovered_gt = int(obs.info.get("num_discovered_gt", obs.info.get("num_discovered", 0)) or 0)
                            ep_total_gt = int(obs.info.get("scene_reward_gt_count", 0) or 0)
                            ep_steps = self.episode_step_counters[env_id]
                            scene_index = self.per_env_scene_indices[env_id]
                            scene_number = scene_index + 1
                            scene_name = self._scene_name(scene_index)
                            worker_episode_no = self.per_env_episode_counts[env_id] + 1
                            scene_episode_no = self.scene_episode_counts[scene_index] + 1
                            completed_episode_no = episode_count + 1
                            assigned_episode_no = self.per_env_scene_assignment_numbers[env_id]
                            epoch_no = self.per_env_scene_epochs[env_id]
                            self.scene_episode_counts[scene_index] = scene_episode_no
                            self.ep_info_buffer.append({
                                "score": ep_score,
                                "coverage": ep_coverage,
                                "path_coverage": ep_path_coverage,
                                "visible_coverage": ep_visible_coverage,
                                "num_discovered_gt": ep_discovered_gt,
                                "scene_reward_gt_count": ep_total_gt,
                                "steps": ep_steps,
                                "env_id": env_id,
                                "scene_index": scene_index,
                                "scene_name": scene_name,
                                "worker_episode": worker_episode_no,
                                "scene_episode": scene_episode_no,
                                "assigned_episode": assigned_episode_no,
                                "completed_episode": completed_episode_no,
                                "epoch": epoch_no,
                                "parallel_round": epoch_no,
                            })
                            
                            if ep_score > max_score:
                                max_score = ep_score
                            
                            print(
                                f"[Epoch {epoch_no}] "
                                f"Env={env_id + 1}/{self.num_workers}, "
                                f"AssignedEp={assigned_episode_no}, "
                                f"CompletedEp={completed_episode_no}, "
                                f"WorkerEp={worker_episode_no}, "
                                f"Scene={scene_number}/{self.scene_count}({scene_name}), "
                                f"SceneEp={scene_episode_no}, "
                                f"Score={ep_score:.2f}, Coverage={ep_coverage:.2f}, "
                                f"PathCov={ep_path_coverage:.2f}, VisCov={ep_visible_coverage:.2f}, "
                                f"GT={ep_discovered_gt}/{ep_total_gt}, Steps={ep_steps}"
                            )

                        try:
                            reason = "success" if obs.info and obs.info.get("success") else "done"
                            self.env_collector.finalize_one(env_id, reason=reason)
                        except Exception as exc:
                            print(f"[WARN] Failed to finalize worker {env_id} episode: {exc}")
                        
                        # Reset only this environment; do not disturb the other workers.
                        self.per_env_episode_counts[env_id] += 1
                        next_scene_index = self._claim_scene_for_env(env_id)
                        obs_list[env_id] = self.env_collector.reset_one(
                            env_id,
                            scene_id=str(next_scene_index + 1),
                            random_start=True,
                            episode_tag=self._episode_tag(env_id, next_scene_index),
                        )

                        self.per_env_discovered_objects[env_id].clear()
                        self.per_env_discovered_instances[env_id].clear()
                        self.per_env_discovered_gt_ids[env_id].clear()
                        self.per_env_prev_score[env_id] = 0.0
                        
                        self.episode_step_counters[env_id] = 0
                        self.per_env_last_actions[env_id] = -1
                        self.per_env_forward_stuck_counts[env_id] = 0
                        self.per_env_forward_escape_cooldowns[env_id] = 0
                        self.per_env_blocked_forward_headings[env_id] = {}
                        self.per_env_no_move_counts[env_id] = 0
                        self.per_env_turn_loop_counts[env_id] = 0
                        
                        # Reset hidden states
                        self.per_env_hidden_states[env_id]["lssg"] = None
                        self.per_env_hidden_states[env_id]["gssg"] = None
                        self.per_env_hidden_states[env_id]["policy"] = None
                        self._reset_transformer_history(env_id)
                        
                        episode_count += 1
                        
                        if pbar:
                            pbar.update(1)
                self.global_step += 1
                
                # Check if we've collected enough steps for an update
                if step_in_rollout >= self.num_steps_per_rollout:
                    print(
                        f"\n[UPDATE] Rollout collected {rollout_steps_collected} transitions "
                        f"({step_in_rollout} env-steps x {self.num_workers} envs; "
                        f"cumulative={total_steps_collected}), updating..."
                    )
                    
                    # Perform update using bounded chunks so RGB encoder does not see
                    # num_workers * num_steps images at once.
                    self._perform_update()
                    
                    # Reset buffers
                    for buffer in self.per_env_buffers:
                        buffer.clear()
                    
                    step_in_rollout = 0
                    rollout_steps_collected = 0
                    
                    # Log stats
                    if len(self.ep_info_buffer) > 0:
                        avg_score = np.mean([e["score"] for e in self.ep_info_buffer])
                        avg_coverage = np.mean([e.get("coverage", 0.0) for e in self.ep_info_buffer])
                        avg_path_coverage = np.mean([e.get("path_coverage", 0.0) for e in self.ep_info_buffer])
                        avg_visible_coverage = np.mean([e.get("visible_coverage", 0.0) for e in self.ep_info_buffer])
                        avg_steps = np.mean([e["steps"] for e in self.ep_info_buffer])
                        self.writer.add_scalar("train/avg_score", avg_score, episode_count)
                        self.writer.add_scalar("train/avg_coverage", avg_coverage, episode_count)
                        self.writer.add_scalar("train/avg_path_coverage", avg_path_coverage, episode_count)
                        self.writer.add_scalar("train/avg_visible_coverage", avg_visible_coverage, episode_count)
                        self.writer.add_scalar("train/avg_steps", avg_steps, episode_count)
                        self.writer.add_scalar("train/max_score", max_score, episode_count)
                        print(
                            f"[STATS] Avg Score: {avg_score:.2f}, Avg Coverage: {avg_coverage:.2f}, "
                            f"PathCov: {avg_path_coverage:.2f}, VisCov: {avg_visible_coverage:.2f}, "
                            f"Avg Steps: {avg_steps:.1f}, Max Score: {max_score:.2f}"
                        )
        
        except KeyboardInterrupt:
            self.was_interrupted = True
            print("\n[INFO] Training interrupted by user")
        finally:
            if self.detection_service is not None:
                try:
                    self.detection_service.close()
                except Exception:
                    pass
            self.env_collector.close()
            self.writer.close()
            print("[INFO] Training finished")

    def evaluate_policy(
        self,
        deterministic: bool = True,
        output_csv: Optional[str] = None,
        random_start: bool = True,
        stop_on_success: bool = True,
    ):
        """
        Run policy rollouts in the Habitat RL environment without any optimizer update.

        This keeps the same scene scheduler, DINO validation, score, and coverage
        accounting used by training, but never fills rollout buffers or calls
        agent.update(). It is meant to measure the real behavior of an IL
        checkpoint inside the RL environment.
        """
        use_tqdm = sys.stderr.isatty()
        pbar = None
        if use_tqdm:
            mode = "argmax" if deterministic else "sample"
            pbar = tqdm(total=self.total_episodes, desc=f"Eval {mode}", ncols=160, leave=False)

        self.agent.eval()
        episode_count = 0
        total_steps_collected = 0
        max_score = 0.0
        results = []
        csv_file = None
        csv_writer = None

        if output_csv:
            output_dir = os.path.dirname(output_csv)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            csv_file = open(output_csv, "w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "completed_episode",
                    "epoch",
                    "assigned_episode",
                    "env_id",
                    "worker_episode",
                    "scene_index",
                    "scene_number",
                    "scene_name",
                    "scene_episode",
                    "score",
                    "coverage",
                    "path_coverage",
                    "visible_coverage",
                    "num_discovered_gt",
                    "scene_reward_gt_count",
                    "steps",
                    "success",
                    "success_reason",
                ],
            )
            csv_writer.writeheader()

        self._reset_scene_scheduler()
        for env_id in range(self.num_workers):
            self._claim_scene_for_env(env_id)
        self.per_env_episode_counts = [0] * self.num_workers
        self.scene_episode_counts = [0] * self.scene_count

        print("[INFO] Initializing parallel environments for policy evaluation...")
        try:
            initial_scene_ids = [str(scene_index + 1) for scene_index in self.per_env_scene_indices]
            initial_episode_tags = [
                self._episode_tag(env_id, scene_index)
                for env_id, scene_index in enumerate(self.per_env_scene_indices)
            ]
            obs_list = self.env_collector.reset_all(
                scene_ids=initial_scene_ids,
                random_start=random_start,
                episode_tags=initial_episode_tags,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to reset environments for evaluation: {exc}")
            self.env_collector.close()
            if csv_file is not None:
                csv_file.close()
            return results

        self.per_env_last_actions = [-1] * self.num_workers
        self.per_env_forward_stuck_counts = [0] * self.num_workers
        self.per_env_forward_escape_cooldowns = [0] * self.num_workers
        self.per_env_blocked_forward_headings = [dict() for _ in range(self.num_workers)]
        self.per_env_no_move_counts = [0] * self.num_workers
        self.per_env_turn_loop_counts = [0] * self.num_workers
        self.per_env_discovered_objects = [set() for _ in range(self.num_workers)]
        self.per_env_discovered_instances = [set() for _ in range(self.num_workers)]
        self.per_env_discovered_gt_ids = [set() for _ in range(self.num_workers)]
        self.per_env_prev_score = [0.0] * self.num_workers
        for hidden_dict in self.per_env_hidden_states:
            hidden_dict["lssg"] = None
            hidden_dict["gssg"] = None
            hidden_dict["policy"] = None
        self._reset_transformer_history()
        self.episode_step_counters = [0] * self.num_workers

        try:
            while episode_count < self.total_episodes:
                actions, _values = self._get_batch_actions(obs_list, deterministic=deterministic)

                try:
                    obs_list = self.env_collector.step_all(actions, self.per_env_action_diagnostics)
                except Exception as exc:
                    print(f"[ERROR] Parallel eval step failed: {exc}")
                    break

                detection_batches = self._run_detection_batch(obs_list)
                if self.detection_service is not None:
                    try:
                        detection_batches = self.env_collector.annotate_detections_all(detection_batches)
                    except Exception as exc:
                        print(f"[WARN] Failed to annotate DINO detections in workers: {exc}")
                self._summarize_detection_batches(detection_batches)

                for env_id, obs in enumerate(obs_list):
                    if episode_count >= self.total_episodes:
                        break

                    terminated = obs.terminated
                    truncated = obs.truncated
                    done = terminated or truncated

                    score, _det_bonus = self._apply_detection_reward(
                        env_id,
                        obs,
                        detection_batches[env_id] if env_id < len(detection_batches) else [],
                    )
                    if obs.info is None:
                        obs.info = {}
                    obs.info["score"] = float(score)

                    success = self._is_success(obs, score)
                    obs.info["success"] = bool(success)
                    if success:
                        obs.info["success_reason"] = "object_recall_and_coverage"
                        if stop_on_success and not done:
                            terminated = True
                            done = True
                            obs.terminated = True

                    self.per_env_last_actions[env_id] = actions[env_id]
                    moved_distance = 0.0
                    if obs.info:
                        moved_distance = float(obs.info.get("moved_distance", 0.0) or 0.0)
                    if moved_distance <= self.forward_stuck_min_distance:
                        self.per_env_no_move_counts[env_id] += 1
                    else:
                        self.per_env_no_move_counts[env_id] = 0
                    blocked_headings = self.per_env_blocked_forward_headings[env_id]
                    for heading in list(blocked_headings):
                        blocked_headings[heading] -= 1
                        if blocked_headings[heading] <= 0:
                            del blocked_headings[heading]
                    if int(actions[env_id]) == 2 and moved_distance <= self.forward_stuck_min_distance:
                        self.per_env_forward_stuck_counts[env_id] += 1
                        if self.per_env_forward_stuck_counts[env_id] >= self.forward_stuck_window:
                            self.per_env_forward_escape_cooldowns[env_id] = self.forward_escape_cooldown_steps
                        if self.forward_blocked_heading_ttl > 0 and obs.info:
                            yaw = obs.info.get("yaw_deg")
                            if yaw is not None:
                                try:
                                    blocked_headings[int(round(float(yaw) / 30.0)) % 12] = self.forward_blocked_heading_ttl
                                except (TypeError, ValueError):
                                    pass
                    else:
                        self.per_env_forward_stuck_counts[env_id] = 0
                    if self.per_env_forward_escape_cooldowns[env_id] > 0 and int(actions[env_id]) != 2:
                        self.per_env_forward_escape_cooldowns[env_id] -= 1
                    if int(actions[env_id]) in (0, 1):
                        self.per_env_turn_loop_counts[env_id] += 1
                    else:
                        self.per_env_turn_loop_counts[env_id] = 0
                    self.episode_step_counters[env_id] += 1
                    total_steps_collected += 1

                    if obs.info:
                        self.writer.add_scalar(f"eval/env_{env_id}/score", obs.info.get("score", 0.0), self.global_step)
                        self.writer.add_scalar(f"eval/env_{env_id}/coverage", obs.info.get("coverage", 0.0), self.global_step)
                        self.writer.add_scalar(f"eval/env_{env_id}/path_coverage", obs.info.get("path_coverage", 0.0), self.global_step)
                        self.writer.add_scalar(f"eval/env_{env_id}/visible_coverage", obs.info.get("visible_coverage", 0.0), self.global_step)

                    if done:
                        ep_score = float(obs.info.get("score", 0.0) if obs.info else 0.0)
                        ep_coverage = float(obs.info.get("coverage", 0.0) if obs.info else 0.0)
                        ep_path_coverage = float(obs.info.get("path_coverage", 0.0) if obs.info else 0.0)
                        ep_visible_coverage = float(obs.info.get("visible_coverage", 0.0) if obs.info else 0.0)
                        ep_discovered_gt = int(obs.info.get("num_discovered_gt", obs.info.get("num_discovered", 0)) or 0) if obs.info else 0
                        ep_total_gt = int(obs.info.get("scene_reward_gt_count", 0) or 0) if obs.info else 0
                        ep_steps = self.episode_step_counters[env_id]
                        scene_index = self.per_env_scene_indices[env_id]
                        scene_number = scene_index + 1
                        scene_name = self._scene_name(scene_index)
                        worker_episode_no = self.per_env_episode_counts[env_id] + 1
                        scene_episode_no = self.scene_episode_counts[scene_index] + 1
                        completed_episode_no = episode_count + 1
                        assigned_episode_no = self.per_env_scene_assignment_numbers[env_id]
                        epoch_no = self.per_env_scene_epochs[env_id]
                        self.scene_episode_counts[scene_index] = scene_episode_no
                        success_flag = bool(obs.info.get("success", False)) if obs.info else False
                        success_reason = str(obs.info.get("success_reason", "")) if obs.info else ""

                        result = {
                            "completed_episode": completed_episode_no,
                            "epoch": epoch_no,
                            "assigned_episode": assigned_episode_no,
                            "env_id": env_id + 1,
                            "worker_episode": worker_episode_no,
                            "scene_index": scene_index,
                            "scene_number": scene_number,
                            "scene_name": scene_name,
                            "scene_episode": scene_episode_no,
                            "score": ep_score,
                            "coverage": ep_coverage,
                            "path_coverage": ep_path_coverage,
                            "visible_coverage": ep_visible_coverage,
                            "num_discovered_gt": ep_discovered_gt,
                            "scene_reward_gt_count": ep_total_gt,
                            "steps": ep_steps,
                            "success": success_flag,
                            "success_reason": success_reason,
                        }
                        results.append(result)
                        if csv_writer is not None:
                            csv_writer.writerow(result)
                            csv_file.flush()

                        if ep_score > max_score:
                            max_score = ep_score

                        print(
                            f"[EVAL Epoch {epoch_no}] "
                            f"Env={env_id + 1}/{self.num_workers}, "
                            f"AssignedEp={assigned_episode_no}, "
                            f"CompletedEp={completed_episode_no}, "
                            f"WorkerEp={worker_episode_no}, "
                            f"Scene={scene_number}/{self.scene_count}({scene_name}), "
                            f"SceneEp={scene_episode_no}, "
                            f"Score={ep_score:.2f}, Coverage={ep_coverage:.2f}, "
                            f"PathCov={ep_path_coverage:.2f}, VisCov={ep_visible_coverage:.2f}, "
                            f"GT={ep_discovered_gt}/{ep_total_gt}, Steps={ep_steps}, "
                            f"Success={success_flag}"
                        )

                        try:
                            reason = "success" if success_flag else "done"
                            self.env_collector.finalize_one(env_id, reason=reason)
                        except Exception as exc:
                            print(f"[WARN] Failed to finalize eval worker {env_id} episode: {exc}")

                        self.per_env_episode_counts[env_id] += 1
                        episode_count += 1
                        if pbar:
                            pbar.update(1)

                        if episode_count >= self.total_episodes:
                            continue

                        next_scene_index = self._claim_scene_for_env(env_id)
                        obs_list[env_id] = self.env_collector.reset_one(
                            env_id,
                            scene_id=str(next_scene_index + 1),
                            random_start=random_start,
                            episode_tag=self._episode_tag(env_id, next_scene_index),
                        )

                        self.per_env_discovered_objects[env_id].clear()
                        self.per_env_discovered_instances[env_id].clear()
                        self.per_env_discovered_gt_ids[env_id].clear()
                        self.per_env_prev_score[env_id] = 0.0
                        self.episode_step_counters[env_id] = 0
                        self.per_env_last_actions[env_id] = -1
                        self.per_env_forward_stuck_counts[env_id] = 0
                        self.per_env_forward_escape_cooldowns[env_id] = 0
                        self.per_env_blocked_forward_headings[env_id] = {}
                        self.per_env_no_move_counts[env_id] = 0
                        self.per_env_turn_loop_counts[env_id] = 0
                        self.per_env_hidden_states[env_id]["lssg"] = None
                        self.per_env_hidden_states[env_id]["gssg"] = None
                        self.per_env_hidden_states[env_id]["policy"] = None
                        self._reset_transformer_history(env_id)

                self.global_step += 1

        except KeyboardInterrupt:
            self.was_interrupted = True
            print("\n[INFO] Policy evaluation interrupted by user")
        finally:
            if pbar:
                pbar.close()
            if results:
                avg_score = float(np.mean([item["score"] for item in results]))
                avg_coverage = float(np.mean([item["coverage"] for item in results]))
                avg_path_coverage = float(np.mean([item["path_coverage"] for item in results]))
                avg_visible_coverage = float(np.mean([item["visible_coverage"] for item in results]))
                avg_steps = float(np.mean([item["steps"] for item in results]))
                success_rate = float(np.mean([1.0 if item["success"] else 0.0 for item in results]))
                self.writer.add_scalar("eval/avg_score", avg_score, episode_count)
                self.writer.add_scalar("eval/avg_coverage", avg_coverage, episode_count)
                self.writer.add_scalar("eval/avg_path_coverage", avg_path_coverage, episode_count)
                self.writer.add_scalar("eval/avg_visible_coverage", avg_visible_coverage, episode_count)
                self.writer.add_scalar("eval/avg_steps", avg_steps, episode_count)
                self.writer.add_scalar("eval/success_rate", success_rate, episode_count)
                self.writer.add_scalar("eval/max_score", max_score, episode_count)
                print(
                    "[EVAL SUMMARY] "
                    f"Episodes={len(results)}, Avg Score={avg_score:.3f}, "
                    f"Avg Coverage={avg_coverage:.3f}, PathCov={avg_path_coverage:.3f}, "
                    f"VisCov={avg_visible_coverage:.3f}, Avg Steps={avg_steps:.1f}, "
                    f"SuccessRate={success_rate:.3f}, Max Score={max_score:.3f}, "
                    f"TotalSteps={total_steps_collected}"
                )
            else:
                print("[EVAL SUMMARY] No completed episodes")

            if csv_file is not None:
                csv_file.close()
                print(f"[INFO] Evaluation CSV: {output_csv}")
            if self.detection_service is not None:
                try:
                    self.detection_service.close()
                except Exception:
                    pass
            self.env_collector.close()
            self.writer.close()
            print("[INFO] Policy evaluation finished")

        return results
    
    def _perform_update(self):
        """
        Perform a single policy update using data from all environments.
        """
        update_batches = []
        total_count = 0
        gamma = float(self.agent_config.get("gamma", 0.99))

        for buffer in self.per_env_buffers:
            if len(buffer.rewards) == 0:
                continue

            returns = []
            G = 0.0
            for reward, done in zip(reversed(buffer.rewards), reversed(buffer.dones)):
                if done:
                    G = 0.0
                G = float(reward) + gamma * G
                returns.insert(0, G)

            states = list(
                zip(
                    buffer.state_rgb,
                    buffer.state_lssg,
                    buffer.state_gssg,
                    buffer.state_occ,
                )
            )
            env_count = len(buffer.actions)
            total_count += env_count
            for start in range(0, env_count, self.update_batch_size):
                end = min(start + self.update_batch_size, env_count)
                update_batches.append(
                    {
                        "states": states[start:end],
                        "actions": buffer.actions[start:end],
                        "rewards": buffer.rewards[start:end],
                        "dones": buffer.dones[start:end],
                        "last_actions": buffer.last_actions[start:end],
                        "agent_pos": buffer.agent_positions[start:end],
                        "returns": returns[start:end],
                    }
                )

        if total_count == 0:
            return

        if not hasattr(self.agent, "update"):
            return

        chunk_size = min(self.update_batch_size, total_count)
        num_chunks = len(update_batches)
        print(f"[UPDATE] Running {num_chunks} mini-batch updates (batch_size={chunk_size}, transitions={total_count})")

        update_results = []
        for batch in update_batches:
            self.agent.rollout_buffers.clear()
            self.agent.rollout_buffers.add_batch(
                states=batch["states"],
                actions=batch["actions"],
                rewards=batch["rewards"],
                dones=batch["dones"],
                hiddens=[{"lssg": None, "gssg": None, "policy": None}],
                last_actions=batch["last_actions"],
                agent_pos=batch["agent_pos"],
                returns=batch["returns"],
            )
            result = self.agent.update()
            if result:
                update_results.append(result)

        if update_results:
            mean_loss = float(np.mean([item.get("loss", 0.0) for item in update_results]))
            mean_entropy = float(np.mean([item.get("entropy", 0.0) for item in update_results]))
            mean_ret_std = float(np.mean([item.get("ret_std", 0.0) for item in update_results]))
            self.writer.add_scalar("update/loss", mean_loss, self.global_step)
            self.writer.add_scalar("update/entropy", mean_entropy, self.global_step)
            self.writer.add_scalar("update/ret_std", mean_ret_std, self.global_step)
            print(
                f"[UPDATE] loss={mean_loss:.4f}, entropy={mean_entropy:.4f}, "
                f"ret_std={mean_ret_std:.4f}"
            )

        bc_result = self._perform_bc_regularization()
        if bc_result:
            self.writer.add_scalar("bc/loss", bc_result["bc_loss"], self.global_step)
            self.writer.add_scalar("bc/coef", bc_result["bc_coef"], self.global_step)
            self.writer.add_scalar("eval/argmax_expert_action_acc", bc_result["bc_acc"], self.global_step)
            print(
                f"[BC] loss={bc_result['bc_loss']:.4f}, "
                f"argmax_acc={bc_result['bc_acc']:.4f}, coef={bc_result['bc_coef']:.4f}"
            )
    
    def _get_batch_from_buffer(self, buffer: RolloutBuffer):
        """
        Convert a RolloutBuffer to a batch dict for forward_update.
        """
        batch_data = buffer.get(self.agent_config.get("gamma", 0.99))
        
        batch = {
            k: batch_data[k] for k in [
                "rgb", "lssg", "gssg", "occ", "actions", "returns", "last_actions", "agent_positions"
            ]
        }
        
        # Convert to tensors and ensure correct shapes
        for k in ["actions", "returns", "last_actions"]:
            if not isinstance(batch[k], torch.Tensor):
                batch[k] = torch.tensor(batch[k], device=self.device)
            if batch[k].dim() == 1:
                batch[k] = batch[k].unsqueeze(0)
        
        for k in ["rgb", "lssg", "gssg", "occ", "agent_positions"]:
            if isinstance(batch[k], list) and not isinstance(batch[k][0], list):
                batch[k] = [batch[k]]
        
        return batch
    
    def close(self):
        """Cleanup resources."""
        self.env_collector.close()
        self.writer.close()
