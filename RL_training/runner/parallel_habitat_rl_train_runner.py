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
from collections import deque
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from components.environments.parallel_habitat_collector import ParallelHabitatCollector
from components.perception.hm3d_labels import HM3D_REWARD_EXCLUDED_LABELS
from components.utils.observation import Observation
from components.utils.rollout_buffer import RolloutBuffer


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
            "new_cell_reward",
            "discovery_bonus_scale",
            "collision_penalty",
            "dino_max_box_area_ratio",
            "dino_max_box_aspect_ratio",
            "gt_validation_iou_threshold",
            "gt_validation_mode",
            "success_recall_threshold",
            "success_min_coverage",
            "success_reward",
            "reward_allow_semantic_iou_only",
            "reward_excluded_labels",
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
    
    def _get_batch_actions(self, obs_list: List[Observation]):
        """
        Get actions for all environments in parallel via a single batch forward.
        """
        if not obs_list:
            return [], np.array([])

        with torch.no_grad():
            batch_dict = self._build_batch_dict(obs_list)
            last_actions = torch.tensor(
                [[action] for action in self.per_env_last_actions],
                dtype=torch.long,
                device=self.device,
            )

            if self.navigation_config.get("use_transformer"):
                state_seq, _, _ = self.agent.encoder(batch_dict, last_actions)
                policy_hidden = None
                lssg_hidden_out = [None] * len(obs_list)
                gssg_hidden_out = [None] * len(obs_list)
            else:
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

            if not self.navigation_config.get("use_transformer"):
                policy_hidden_out = self._split_lstm_hidden(new_policy_hidden)
                for env_id in range(self.num_workers):
                    self.per_env_hidden_states[env_id]["lssg"] = lssg_hidden_out[env_id]
                    self.per_env_hidden_states[env_id]["gssg"] = gssg_hidden_out[env_id]
                    self.per_env_hidden_states[env_id]["policy"] = policy_hidden_out[env_id]

            logits = logits[:, -1, :]
            if value is not None:
                value = value[:, -1]

            probs = torch.softmax(logits, dim=-1)
            from torch.distributions import Categorical
            dist = Categorical(probs=probs)
            actions = dist.sample().tolist()
            values = value.tolist() if value is not None else [0.0] * len(actions)

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
        self.per_env_discovered_objects = [set() for _ in range(self.num_workers)]
        self.per_env_discovered_instances = [set() for _ in range(self.num_workers)]
        self.per_env_discovered_gt_ids = [set() for _ in range(self.num_workers)]
        self.per_env_prev_score = [0.0] * self.num_workers
        for hidden_dict in self.per_env_hidden_states:
            hidden_dict["lssg"] = None
            hidden_dict["gssg"] = None
            hidden_dict["policy"] = None
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
                    obs_list = self.env_collector.step_all(actions)
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
                    self.episode_step_counters[env_id] += 1
                    total_steps_collected += 1
                    rollout_steps_collected += 1
                    
                    # Track episode info
                    if obs.info:
                        score = obs.info.get("score", 0.0)
                        coverage = obs.info.get("coverage", 0.0)
                        self.writer.add_scalar(f"env_{env_id}/reward", reward, self.global_step)
                        self.writer.add_scalar(f"env_{env_id}/score", score, self.global_step)
                        self.writer.add_scalar(f"env_{env_id}/coverage", coverage, self.global_step)
                    
                    # Reset if done
                    if done:
                        if obs.info:
                            ep_score = obs.info.get("score", 0.0)
                            ep_coverage = obs.info.get("coverage", 0.0)
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
                        
                        # Reset hidden states
                        self.per_env_hidden_states[env_id]["lssg"] = None
                        self.per_env_hidden_states[env_id]["gssg"] = None
                        self.per_env_hidden_states[env_id]["policy"] = None
                        
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
                        avg_steps = np.mean([e["steps"] for e in self.ep_info_buffer])
                        self.writer.add_scalar("train/avg_score", avg_score, episode_count)
                        self.writer.add_scalar("train/avg_coverage", avg_coverage, episode_count)
                        self.writer.add_scalar("train/avg_steps", avg_steps, episode_count)
                        self.writer.add_scalar("train/max_score", max_score, episode_count)
                        print(
                            f"[STATS] Avg Score: {avg_score:.2f}, Avg Coverage: {avg_coverage:.2f}, "
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
