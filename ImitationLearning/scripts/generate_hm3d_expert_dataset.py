#!/usr/bin/env python3
"""Generate Habitat/HM3D expert demonstrations for behavior cloning.

The expert is an oracle over Habitat semantic observations and navmesh motion:
it favors actions that reveal new reward-eligible GT objects, then actions that
expand coverage. DINO is intentionally not used so demonstrations are not
polluted by detector noise.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from components.environments.habitat_env import HabitatEnv
from ImitationLearning.hm3d_scene_sets import DEFAULT_HM3D_TRAIN_SCENES_CSV
from train_habitat_parallel import (
    _build_habitat_dataset_config,
    _load_config_mapping,
    _resolve_habitat_scene_list,
)


def _allowed_env_kwargs(env_config: Dict) -> Dict:
    allowed = {
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
        "gt_validation_iou_threshold",
        "gt_validation_mode",
        "success_recall_threshold",
        "success_min_coverage",
        "success_reward",
        "reward_allow_semantic_iou_only",
        "reward_excluded_labels",
        "max_actions",
        "save_debug_interval",
    }
    return {k: v for k, v in env_config.items() if k in allowed and v is not None}


def _yaw_deg(env: HabitatEnv) -> float:
    state = env.sim.get_agent(0).get_state()
    return float(env._yaw_from_quat(state.rotation))


def _pose(env: HabitatEnv):
    state = env.sim.get_agent(0).get_state()
    pos = state.position
    return np.array([float(pos[0]), float(pos[1]), float(pos[2]), _yaw_deg(env)], dtype=np.float32)


def _choose_oracle_action(env: HabitatEnv, stagnation: int, rng: random.Random) -> int:
    scores = env.oracle_action_scores(candidate_actions=(0, 1, 2))

    def rank(item):
        action = int(item["action"])
        turn_balance = 0.02 if (stagnation % 2 == 0 and action == 0) or (stagnation % 2 == 1 and action == 1) else 0.0
        forward_bonus = 0.05 if action == 2 else 0.0
        collision_penalty = -1.0 if item.get("collision") else 0.0
        return (
            10.0 * float(item["score_gain"])
            + 3.0 * float(item["visible_new_gt"])
            + 1.0 * float(item["new_cells"])
            + forward_bonus
            + turn_balance
            + collision_penalty
        )

    best = max(scores, key=rank)
    if rank(best) <= 0.0:
        # If no one-step action is informative, alternate turns to scan views,
        # with occasional forward moves so the policy still learns locomotion.
        if stagnation >= 4 and rng.random() < 0.35:
            return 2
        return 0 if stagnation % 2 == 0 else 1
    return int(best["action"])


def _save_episode(path: Path, records: List[Dict], env: HabitatEnv, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgbs = np.stack([rec["rgb"] for rec in records]).astype(np.uint8)
    actions = np.asarray([rec["action"] for rec in records], dtype=np.int64)
    last_actions = np.asarray([rec["last_action"] for rec in records], dtype=np.int64)
    poses = np.stack([rec["pose"] for rec in records]).astype(np.float32)
    agent_pos = poses[:, [0, 2]].astype(np.float32)
    score = np.asarray([rec["score"] for rec in records], dtype=np.float32)
    coverage = np.asarray([rec["coverage"] for rec in records], dtype=np.float32)
    payload = {
        "rgb": rgbs,
        "actions": actions,
        "last_actions": last_actions,
        "pose": poses,
        "agent_pos": agent_pos,
        "score": score,
        "coverage": coverage,
        "num_actions": np.asarray([len(env.get_actions())], dtype=np.int64),
        "scene_index": np.asarray([metadata["scene_index"]], dtype=np.int64),
        "start_index": np.asarray([metadata["start_index"]], dtype=np.int64),
        "final_score": np.asarray([metadata["final_score"]], dtype=np.float32),
        "final_coverage": np.asarray([metadata["final_coverage"]], dtype=np.float32),
    }
    for key in ("poi_id", "poi_index"):
        if key in metadata:
            payload[key] = np.asarray([metadata[key]], dtype=np.int64)
    if "start_yaw_deg" in metadata:
        payload["start_yaw_deg"] = np.asarray([metadata["start_yaw_deg"]], dtype=np.float32)
    if "requested_start_position" in metadata:
        payload["requested_start_position"] = np.asarray(metadata["requested_start_position"], dtype=np.float32)
    if "actual_start_position" in metadata:
        payload["actual_start_position"] = np.asarray(metadata["actual_start_position"], dtype=np.float32)
    np.savez_compressed(path, **payload)


def generate(args) -> None:
    rng = random.Random(args.seed)
    env_config = _load_config_mapping(args.conf_path, "env_config.yaml", "env.json")
    scene_list = args.habitat_scenes or DEFAULT_HM3D_TRAIN_SCENES_CSV
    habitat_scene_ids, missing = _resolve_habitat_scene_list(
        args.dataset_root,
        single_scene=args.habitat_scene,
        scene_list=scene_list if args.habitat_scene is None else None,
    )
    if missing:
        print(f"[WARNING] Missing scenes: {missing}")
    if not habitat_scene_ids:
        raise RuntimeError("No valid HM3D scenes resolved.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = _build_habitat_dataset_config(
        str(Path(args.dataset_root) / "hm3d_annotated_basis.scene_dataset_config.json"),
        habitat_scene_ids,
        args.dataset_root,
        output_dir=str(output_dir),
    )

    env_kwargs = _allowed_env_kwargs(env_config)
    env_kwargs.update(
        {
            "render": False,
            "use_detector": False,
            "detector": None,
            "max_actions": int(args.max_steps),
            "save_debug_path": str(output_dir / "debug") if args.save_debug else None,
            "save_debug_interval": int(args.save_debug_interval),
        }
    )

    env = HabitatEnv(
        dataset_root=args.dataset_root,
        config_file=dataset_config,
        scene_id=habitat_scene_ids[0],
        scene_ids=habitat_scene_ids,
        gpu_device_id=int(args.gpu_id),
        **env_kwargs,
    )

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": args.dataset_root,
        "scene_count": len(habitat_scene_ids),
        "scenes": [Path(scene).parent.name for scene in habitat_scene_ids],
        "episodes_per_scene": int(args.episodes_per_scene),
        "max_steps": int(args.max_steps),
        "min_score": float(args.min_score),
        "min_coverage": float(args.min_coverage),
        "expert": "hm3d_semantic_navmesh_oracle_v1",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    saved = 0
    rejected = 0
    try:
        total = len(habitat_scene_ids) * int(args.episodes_per_scene)
        progress = tqdm(total=total, desc="Generating HM3D expert demos")
        for scene_idx in range(len(habitat_scene_ids)):
            scene_name = Path(habitat_scene_ids[scene_idx]).parent.name
            for start_idx in range(int(args.episodes_per_scene)):
                episode_tag = f"expert_scene_{scene_idx + 1:02d}_start_{start_idx:03d}"
                obs = env.reset(
                    scene_number=scene_idx + 1,
                    random_start=True,
                    episode_tag=episode_tag if args.save_debug else None,
                )
                env.observe_visible_reward_gt()
                last_action = -1
                stagnation = 0
                records = []
                last_score = float(obs.info.get("score", 0.0))
                last_coverage = float(obs.info.get("coverage", 0.0))

                for _ in range(int(args.max_steps)):
                    action = _choose_oracle_action(env, stagnation, rng)
                    records.append(
                        {
                            "rgb": np.asarray(obs.state[0], dtype=np.uint8).copy(),
                            "last_action": int(last_action),
                            "action": int(action),
                            "pose": _pose(env),
                            "score": float(obs.info.get("score", 0.0)),
                            "coverage": float(obs.info.get("coverage", 0.0)),
                        }
                    )
                    obs = env.step(action)
                    env.observe_visible_reward_gt()
                    score = env._current_discovery_score()
                    coverage = float(obs.info.get("coverage", 0.0))
                    progressed = score > last_score + 1e-6 or coverage > last_coverage + 1e-6
                    stagnation = 0 if progressed else stagnation + 1
                    last_score = score
                    last_coverage = coverage
                    last_action = int(action)

                    if (
                        score >= float(args.target_score)
                        and coverage >= float(args.target_coverage)
                    ):
                        break
                    if stagnation >= int(args.max_stagnation):
                        break

                final_score = float(env._current_discovery_score())
                final_coverage = float(last_coverage)
                quality_ok = (
                    len(records) >= int(args.min_steps)
                    and (
                        final_score >= float(args.min_score)
                        or final_coverage >= float(args.min_coverage)
                    )
                )
                if quality_ok:
                    out_path = (
                        output_dir
                        / scene_name
                        / f"{episode_tag}_score_{final_score:.3f}_cov_{final_coverage:.3f}.npz"
                    )
                    _save_episode(
                        out_path,
                        records,
                        env,
                        {
                            "scene_index": scene_idx,
                            "start_index": start_idx,
                            "final_score": final_score,
                            "final_coverage": final_coverage,
                        },
                    )
                    saved += 1
                else:
                    rejected += 1
                progress.set_postfix(saved=saved, rejected=rejected, score=f"{final_score:.2f}", cov=f"{final_coverage:.2f}")
                progress.update(1)
        progress.close()
    finally:
        env.close()

    print(f"[INFO] HM3D expert dataset complete: saved={saved}, rejected={rejected}, output={output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HM3D/Habitat expert imitation dataset.")
    parser.add_argument("--conf_path", type=str, default="/home/wgy/RL/config")
    parser.add_argument("--dataset_root", type=str, default="/home/wgy/hm3d/scene_datasets/hm3d")
    parser.add_argument("--habitat_scene", type=str, default=None)
    parser.add_argument(
        "--habitat_scenes",
        type=str,
        default=None,
        help="Comma-separated scene list. Defaults to the current 50-scene HM3D train set.",
    )
    parser.add_argument("--output_dir", type=str, default="/home/wgy/RL/components/data/hm3d_il_dataset")
    parser.add_argument("--episodes_per_scene", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--min_steps", type=int, default=30)
    parser.add_argument("--target_score", type=float, default=0.60)
    parser.add_argument("--target_coverage", type=float, default=0.20)
    parser.add_argument("--min_score", type=float, default=0.12)
    parser.add_argument("--min_coverage", type=float, default=0.08)
    parser.add_argument("--max_stagnation", type=int, default=80)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--save_debug_interval", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
