#!/usr/bin/env python3
"""Generate higher-quality HM3D expert demos with viewpoint-cover planning.

This follows the spirit of the original AI2-THOR imitation-data generator:
sample candidate viewpoints, keep viewpoints that reveal reward-eligible GT
semantic objects, greedily choose a compact cover, visit them in a short order,
and execute primitive Habitat actions to reach and face those viewpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import habitat_sim
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from components.environments.habitat_env import HabitatEnv
from habitat_sim.utils.common import quat_from_angle_axis
from ImitationLearning.hm3d_scene_sets import DEFAULT_HM3D_TRAIN_SCENES_CSV
from ImitationLearning.scripts.generate_hm3d_expert_dataset import _allowed_env_kwargs, _pose, _save_episode
from train_habitat_parallel import (
    _build_habitat_dataset_config,
    _load_config_mapping,
    _resolve_habitat_scene_list,
)


@dataclass(frozen=True)
class Viewpoint:
    index: int
    position: np.ndarray
    yaw_deg: float
    visible_ids: frozenset


def _angle_wrap_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _xz_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2])))


def _desired_yaw_to_point(source: Sequence[float], target: Sequence[float]) -> float:
    """Habitat forward at yaw=0 moves toward negative Z; left turn increases yaw."""
    dx = float(target[0] - source[0])
    dz = float(target[2] - source[2])
    return math.degrees(math.atan2(-dx, -dz))


def _set_agent_pose(env: HabitatEnv, position: Sequence[float], yaw_deg: float):
    agent = env.sim.get_agent(0)
    state = agent.get_state()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quat_from_angle_axis(math.radians(float(yaw_deg)), np.array([0.0, 1.0, 0.0]))
    agent.set_state(state)
    return env.sim.get_sensor_observations()


def _visible_reward_ids_from_pose(env: HabitatEnv, position, yaw_deg: float, min_pixels: int) -> Set[int]:
    snapshot = env.snapshot_agent_state()
    obs = _set_agent_pose(env, position, yaw_deg)
    env._process_obs(obs, is_reset=True)
    boxes = env._extract_all_visible_semantic_boxes(env._last_semantic, min_pixels=min_pixels)
    visible = set()
    for box in boxes:
        semantic_id = int(box["semantic_id"])
        if semantic_id not in env.scene_reward_gt_ids:
            continue
        canonical = box.get("canonical_label")
        if canonical in env.reward_excluded_labels:
            continue
        visible.add(semantic_id)
    env.restore_agent_state(snapshot)
    return visible


def _sample_candidate_viewpoints(
    env: HabitatEnv,
    rng: random.Random,
    sample_count: int,
    yaw_degrees: Sequence[float],
    min_pixels: int,
) -> List[Viewpoint]:
    pathfinder = env.sim.pathfinder
    largest_island_idx = env._get_largest_island_index()
    candidates: List[Viewpoint] = []
    seen_pose_keys = set()

    for _ in range(int(sample_count)):
        if largest_island_idx is None:
            point = pathfinder.get_random_navigable_point(max_tries=20)
        else:
            point = pathfinder.get_random_navigable_point(max_tries=20, island_index=largest_island_idx)
        if point is None:
            continue
        pos_key = (round(float(point[0]), 2), round(float(point[2]), 2))
        if pos_key in seen_pose_keys:
            continue
        seen_pose_keys.add(pos_key)

        yaw_order = list(yaw_degrees)
        rng.shuffle(yaw_order)
        for yaw_deg in yaw_order:
            visible = _visible_reward_ids_from_pose(env, point, yaw_deg, min_pixels=min_pixels)
            if not visible:
                continue
            candidates.append(
                Viewpoint(
                    index=len(candidates),
                    position=np.asarray(point, dtype=np.float32),
                    yaw_deg=float(yaw_deg),
                    visible_ids=frozenset(visible),
                )
            )

    return candidates


def _choose_viewpoint_cover(
    candidates: Sequence[Viewpoint],
    target_ids: Iterable[int],
    max_viewpoints: int,
) -> List[Viewpoint]:
    uncovered = set(int(item) for item in target_ids)
    remaining = list(candidates)
    selected: List[Viewpoint] = []

    while uncovered and remaining and len(selected) < int(max_viewpoints):
        best = max(
            remaining,
            key=lambda vp: (
                len(set(vp.visible_ids) & uncovered),
                len(vp.visible_ids),
                -float(np.linalg.norm(vp.position)),
            ),
        )
        gain = set(best.visible_ids) & uncovered
        if not gain:
            break
        selected.append(best)
        uncovered -= gain
        remaining = [vp for vp in remaining if vp.index != best.index]

    return selected


def _order_viewpoints(start_position: Sequence[float], viewpoints: Sequence[Viewpoint]) -> List[Viewpoint]:
    ordered: List[Viewpoint] = []
    remaining = list(viewpoints)
    current = np.asarray(start_position, dtype=np.float32)
    while remaining:
        next_vp = min(remaining, key=lambda vp: _xz_dist(current, vp.position))
        ordered.append(next_vp)
        remaining.remove(next_vp)
        current = next_vp.position
    return ordered


def _shortest_path_points(env: HabitatEnv, start: Sequence[float], goal: Sequence[float]) -> List[np.ndarray]:
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_end = np.asarray(goal, dtype=np.float32)
    found = env.sim.pathfinder.find_path(path)
    if not found:
        return []
    return [np.asarray(point, dtype=np.float32) for point in path.points]


def _next_path_waypoint(env: HabitatEnv, goal: Sequence[float], waypoint_reach_dist: float) -> np.ndarray:
    state = env.sim.get_agent(0).get_state()
    current = np.asarray(state.position, dtype=np.float32)
    points = _shortest_path_points(env, current, goal)
    if len(points) <= 1:
        return np.asarray(goal, dtype=np.float32)

    for point in points[1:]:
        if _xz_dist(current, point) > float(waypoint_reach_dist):
            return point
    return np.asarray(goal, dtype=np.float32)


def _record(records: List[Dict], obs, env: HabitatEnv, last_action: int, action: int) -> None:
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


def _best_progress_action(env: HabitatEnv, waypoint: Sequence[float], rng: random.Random) -> int:
    snapshot = env.snapshot_agent_state()
    state = env.sim.get_agent(0).get_state()
    start_pos = np.asarray(state.position, dtype=np.float32)
    target_pos = np.asarray(waypoint, dtype=np.float32)
    start_dist = _xz_dist(start_pos, target_pos)
    start_yaw = float(env._yaw_from_quat(state.rotation))
    desired_yaw = _desired_yaw_to_point(start_pos, target_pos)
    yaw_error = abs(_angle_wrap_deg(desired_yaw - start_yaw))

    candidates = []
    for action in (0, 1, 2):
        env.restore_agent_state(snapshot)
        before_ids = set(env.discovered_gt_ids)
        obs = env.step(action)
        env.observe_visible_reward_gt()
        new_ids = len(set(env.discovered_gt_ids) - before_ids)
        next_state = env.sim.get_agent(0).get_state()
        next_pos = np.asarray(next_state.position, dtype=np.float32)
        next_dist = _xz_dist(next_pos, target_pos)
        next_yaw = float(env._yaw_from_quat(next_state.rotation))
        next_yaw_error = abs(_angle_wrap_deg(desired_yaw - next_yaw))
        moved = _xz_dist(start_pos, next_pos)
        collision = int(action) == 2 and moved < 0.03
        score = (
            20.0 * new_ids
            + 8.0 * max(0.0, start_dist - next_dist)
            + 0.03 * max(0.0, yaw_error - next_yaw_error)
            + (0.20 if int(action) == 2 else 0.0)
            - (4.0 if collision else 0.0)
        )
        candidates.append((score, int(action), obs))

    env.restore_agent_state(snapshot)
    best_score = max(item[0] for item in candidates)
    best_actions = [item[1] for item in candidates if abs(item[0] - best_score) < 1e-9]
    return int(rng.choice(best_actions))


def _face_target_yaw(env: HabitatEnv, target_yaw: float) -> int:
    yaw = float(env._yaw_from_quat(env.sim.get_agent(0).get_state().rotation))
    diff = _angle_wrap_deg(float(target_yaw) - yaw)
    return 0 if diff > 0.0 else 1


def _run_episode_to_viewpoints(
    env: HabitatEnv,
    obs,
    ordered: Sequence[Viewpoint],
    rng: random.Random,
    max_steps: int,
    reach_dist: float,
    max_target_steps: int,
    min_pixels: int,
    waypoint_reach_dist: float,
) -> Tuple[List[Dict], float, float]:
    records: List[Dict] = []
    last_action = -1
    target_index = 0
    target_steps = 0

    env.observe_visible_reward_gt(min_pixels=min_pixels)
    while target_index < len(ordered) and len(records) < int(max_steps):
        target = ordered[target_index]
        state = env.sim.get_agent(0).get_state()
        dist = _xz_dist(state.position, target.position)
        yaw = float(env._yaw_from_quat(state.rotation))
        target_yaw_error = abs(_angle_wrap_deg(float(target.yaw_deg) - yaw))

        if dist <= float(reach_dist):
            if target_yaw_error <= 15.0:
                target_index += 1
                target_steps = 0
                continue
            action = _face_target_yaw(env, target.yaw_deg)
        else:
            waypoint = _next_path_waypoint(env, target.position, waypoint_reach_dist=waypoint_reach_dist)
            action = _best_progress_action(env, waypoint, rng)

        _record(records, obs, env, last_action, action)
        obs = env.step(action)
        env.observe_visible_reward_gt(min_pixels=min_pixels)
        last_action = action
        target_steps += 1

        if obs.terminated:
            break
        if target_steps >= int(max_target_steps):
            target_index += 1
            target_steps = 0

    final_score = float(env._current_discovery_score())
    final_coverage = float(obs.info.get("coverage", 0.0))
    return records, final_score, final_coverage


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
        "candidate_viewpoints": int(args.candidate_viewpoints),
        "max_cover_viewpoints": int(args.max_cover_viewpoints),
        "min_save_score": float(args.min_save_score),
        "min_save_coverage": float(args.min_save_coverage),
        "expert": "hm3d_viewpoint_cover_navmesh_oracle_v1",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    saved = 0
    rejected = 0
    no_cover = 0
    yaw_degrees = [float(item) for item in str(args.yaw_degrees).split(",") if item.strip()]
    try:
        total = len(habitat_scene_ids) * int(args.episodes_per_scene)
        progress = tqdm(total=total, desc="Generating HM3D viewpoint expert demos")
        for scene_idx in range(len(habitat_scene_ids)):
            scene_name = Path(habitat_scene_ids[scene_idx]).parent.name
            for start_idx in range(int(args.episodes_per_scene)):
                episode_tag = f"vp_scene_{scene_idx + 1:02d}_start_{start_idx:03d}"
                obs = env.reset(
                    scene_number=scene_idx + 1,
                    random_start=True,
                    episode_tag=episode_tag if args.save_debug else None,
                )
                start_position = np.asarray(env.sim.get_agent(0).get_state().position, dtype=np.float32)

                candidates = _sample_candidate_viewpoints(
                    env,
                    rng,
                    sample_count=int(args.candidate_viewpoints),
                    yaw_degrees=yaw_degrees,
                    min_pixels=int(args.min_visible_pixels),
                )
                cover = _choose_viewpoint_cover(
                    candidates,
                    target_ids=env.scene_reward_gt_ids,
                    max_viewpoints=int(args.max_cover_viewpoints),
                )
                ordered = _order_viewpoints(start_position, cover)
                if not ordered:
                    no_cover += 1
                    rejected += 1
                    progress.set_postfix(saved=saved, rejected=rejected, no_cover=no_cover, score="0.00", cov="0.00")
                    progress.update(1)
                    continue

                records, final_score, final_coverage = _run_episode_to_viewpoints(
                    env,
                    obs,
                    ordered,
                    rng,
                    max_steps=int(args.max_steps),
                    reach_dist=float(args.reach_dist),
                    max_target_steps=int(args.max_target_steps),
                    min_pixels=int(args.min_visible_pixels),
                    waypoint_reach_dist=float(args.waypoint_reach_dist),
                )
                quality_ok = (
                    len(records) >= int(args.min_steps)
                    and final_score >= float(args.min_save_score)
                    and final_coverage >= float(args.min_save_coverage)
                )
                if quality_ok:
                    out_path = (
                        output_dir
                        / scene_name
                        / f"{episode_tag}_score_{final_score:.3f}_cov_{final_coverage:.3f}_vp_{len(ordered):02d}.npz"
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
                progress.set_postfix(
                    saved=saved,
                    rejected=rejected,
                    no_cover=no_cover,
                    score=f"{final_score:.2f}",
                    cov=f"{final_coverage:.2f}",
                    vp=len(ordered),
                )
                progress.update(1)
        progress.close()
    finally:
        env.close()

    print(
        "[INFO] HM3D viewpoint expert dataset complete: "
        f"saved={saved}, rejected={rejected}, no_cover={no_cover}, output={output_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HM3D viewpoint-cover expert imitation dataset.")
    parser.add_argument("--conf_path", type=str, default="/home/wgy/RL/config")
    parser.add_argument("--dataset_root", type=str, default="/home/wgy/hm3d/scene_datasets/hm3d")
    parser.add_argument("--habitat_scene", type=str, default=None)
    parser.add_argument("--habitat_scenes", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/home/wgy/RL/components/data/hm3d_il_dataset")
    parser.add_argument("--episodes_per_scene", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=900)
    parser.add_argument("--min_steps", type=int, default=80)
    parser.add_argument("--candidate_viewpoints", type=int, default=180)
    parser.add_argument("--max_cover_viewpoints", type=int, default=28)
    parser.add_argument("--yaw_degrees", type=str, default="0,60,120,180,240,300")
    parser.add_argument("--min_visible_pixels", type=int, default=80)
    parser.add_argument("--reach_dist", type=float, default=0.45)
    parser.add_argument("--waypoint_reach_dist", type=float, default=0.30)
    parser.add_argument("--max_target_steps", type=int, default=80)
    parser.add_argument("--min_save_score", type=float, default=0.45)
    parser.add_argument("--min_save_coverage", type=float, default=0.02)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--save_debug_interval", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
