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
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


@dataclass(frozen=True)
class PoiStart:
    index: int
    poi_id: int
    position: np.ndarray


def _angle_wrap_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _xz_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2])))


def _add_timing(timing: Optional[Dict[str, float]], key: str, start_time: float) -> None:
    if timing is not None:
        timing[key] = float(timing.get(key, 0.0)) + (time.perf_counter() - start_time)


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


def _load_scene_pois(scene_id: str, poi_dir: Path) -> List[PoiStart]:
    scene_hash = scene_id.split("-")[0].split("/")[-1] if "/" in scene_id else scene_id.split("-")[0]
    poi_file = poi_dir / f"{scene_hash}_poi.json"
    if not poi_file.exists():
        return []
    try:
        payload = json.loads(poi_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARNING] Failed to read POI file {poi_file}: {exc}")
        return []

    starts: List[PoiStart] = []
    for idx, item in enumerate(payload.get("poi", [])):
        try:
            position = np.asarray(item["position"], dtype=np.float32)
        except Exception:
            continue
        if position.shape[0] < 3:
            continue
        try:
            poi_id = int(item.get("id", idx))
        except Exception:
            poi_id = int(idx)
        starts.append(PoiStart(index=idx, poi_id=poi_id, position=position[:3]))
    return starts


def _fallback_scene_starts(env: HabitatEnv, count: int) -> List[PoiStart]:
    starts: List[PoiStart] = []
    largest_island_idx = env._get_largest_island_index()
    for idx in range(max(int(count), 1)):
        if largest_island_idx is None:
            point = env.sim.pathfinder.get_random_navigable_point(max_tries=20)
        else:
            point = env.sim.pathfinder.get_random_navigable_point(
                max_tries=20,
                island_index=largest_island_idx,
            )
        if point is None:
            continue
        starts.append(PoiStart(index=idx, poi_id=idx, position=np.asarray(point, dtype=np.float32)))
    return starts


def _reset_to_start(env: HabitatEnv, scene_number: int, start: PoiStart, yaw_deg: float, episode_tag: str, save_debug: bool):
    return env.reset(
        scene_number=scene_number,
        random_start=False,
        start_position=start.position,
        start_rotation=math.radians(float(yaw_deg)),
        episode_tag=episode_tag if save_debug else None,
    )


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


def _best_progress_action(
    env: HabitatEnv,
    waypoint: Sequence[float],
    rng: random.Random,
    timing: Optional[Dict[str, float]] = None,
) -> int:
    t0 = time.perf_counter()
    snapshot = env.snapshot_agent_state()
    _add_timing(timing, "oracle_snapshot", t0)
    state = env.sim.get_agent(0).get_state()
    start_pos = np.asarray(state.position, dtype=np.float32)
    target_pos = np.asarray(waypoint, dtype=np.float32)
    start_dist = _xz_dist(start_pos, target_pos)
    start_yaw = float(env._yaw_from_quat(state.rotation))
    desired_yaw = _desired_yaw_to_point(start_pos, target_pos)
    yaw_error = abs(_angle_wrap_deg(desired_yaw - start_yaw))

    candidates = []
    for action in (0, 1, 2):
        t0 = time.perf_counter()
        env.restore_agent_state(snapshot)
        _add_timing(timing, "oracle_restore", t0)
        before_ids = set(env.discovered_gt_ids)
        t0 = time.perf_counter()
        obs = env.step(action)
        _add_timing(timing, "oracle_step", t0)
        t0 = time.perf_counter()
        env.observe_visible_reward_gt()
        _add_timing(timing, "oracle_observe", t0)
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

    t0 = time.perf_counter()
    env.restore_agent_state(snapshot)
    _add_timing(timing, "oracle_restore", t0)
    best_score = max(item[0] for item in candidates)
    best_actions = [item[1] for item in candidates if abs(item[0] - best_score) < 1e-9]
    return int(rng.choice(best_actions))


def _face_target_yaw(env: HabitatEnv, target_yaw: float) -> int:
    yaw = float(env._yaw_from_quat(env.sim.get_agent(0).get_state().rotation))
    diff = _angle_wrap_deg(float(target_yaw) - yaw)
    return 0 if diff > 0.0 else 1


def _geometric_progress_action(
    env: HabitatEnv,
    waypoint: Sequence[float],
    turn_threshold_deg: float,
    timing: Optional[Dict[str, float]] = None,
) -> int:
    t0 = time.perf_counter()
    state = env.sim.get_agent(0).get_state()
    current = np.asarray(state.position, dtype=np.float32)
    desired_yaw = _desired_yaw_to_point(current, waypoint)
    yaw = float(env._yaw_from_quat(state.rotation))
    diff = _angle_wrap_deg(desired_yaw - yaw)
    _add_timing(timing, "geometric_action", t0)
    if abs(diff) > float(turn_threshold_deg):
        return 0 if diff > 0.0 else 1
    return 2


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
    action_policy: str,
    turn_threshold_deg: float,
) -> Tuple[List[Dict], float, float, Dict[str, float]]:
    run_start = time.perf_counter()
    timing: Dict[str, float] = {}
    records: List[Dict] = []
    last_action = -1
    target_index = 0
    target_steps = 0

    t0 = time.perf_counter()
    env.observe_visible_reward_gt(min_pixels=min_pixels)
    _add_timing(timing, "initial_observe", t0)
    while target_index < len(ordered) and len(records) < int(max_steps):
        loop_start = time.perf_counter()
        target = ordered[target_index]
        state = env.sim.get_agent(0).get_state()
        dist = _xz_dist(state.position, target.position)
        yaw = float(env._yaw_from_quat(state.rotation))
        target_yaw_error = abs(_angle_wrap_deg(float(target.yaw_deg) - yaw))
        _add_timing(timing, "loop_state", loop_start)

        if dist <= float(reach_dist):
            if target_yaw_error <= 15.0:
                target_index += 1
                target_steps = 0
                continue
            t0 = time.perf_counter()
            action = _face_target_yaw(env, target.yaw_deg)
            _add_timing(timing, "face_action", t0)
        else:
            t0 = time.perf_counter()
            waypoint = _next_path_waypoint(env, target.position, waypoint_reach_dist=waypoint_reach_dist)
            _add_timing(timing, "path_plan", t0)
            if action_policy == "oracle":
                t0 = time.perf_counter()
                action = _best_progress_action(env, waypoint, rng, timing=timing)
                _add_timing(timing, "best_action_total", t0)
            elif action_policy == "geometric":
                action = _geometric_progress_action(
                    env,
                    waypoint,
                    turn_threshold_deg=turn_threshold_deg,
                    timing=timing,
                )
            else:
                raise ValueError(f"Unsupported action_policy={action_policy!r}")

        t0 = time.perf_counter()
        _record(records, obs, env, last_action, action)
        _add_timing(timing, "record", t0)
        t0 = time.perf_counter()
        obs = env.step(action)
        _add_timing(timing, "real_step", t0)
        t0 = time.perf_counter()
        env.observe_visible_reward_gt(min_pixels=min_pixels)
        _add_timing(timing, "real_observe", t0)
        last_action = action
        target_steps += 1
        timing["steps"] = float(timing.get("steps", 0.0)) + 1.0

        if obs.terminated:
            break
        if target_steps >= int(max_target_steps):
            target_index += 1
            target_steps = 0

    t0 = time.perf_counter()
    final_score = float(env._current_discovery_score())
    final_coverage = float(obs.info.get("coverage", 0.0))
    _add_timing(timing, "final_metrics", t0)
    timing["run_total"] = time.perf_counter() - run_start
    return records, final_score, final_coverage, timing


def _format_timing(timing: Dict[str, float]) -> str:
    steps = max(float(timing.get("steps", 0.0)), 1.0)
    keys = [
        "run_total",
        "path_plan",
        "best_action_total",
        "oracle_step",
        "oracle_observe",
        "real_step",
        "real_observe",
        "record",
        "initial_observe",
        "final_metrics",
    ]
    parts = [f"steps={int(timing.get('steps', 0.0))}"]
    for key in keys:
        if key in timing:
            parts.append(f"{key}={timing[key]:.2f}s")
    if "run_total" in timing:
        parts.append(f"per_step={timing['run_total'] / steps:.3f}s")
    return " ".join(parts)


def _assigned_task_count(offset: int, count: int, num_shards: int, shard_index: int) -> int:
    return sum(
        1
        for local_idx in range(int(count))
        if (int(offset) + local_idx) % int(num_shards) == int(shard_index)
    )


def generate(args) -> None:
    rng = random.Random(args.seed)
    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

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

    start_yaw_degrees = [float(item) for item in str(args.start_yaw_degrees).split(",") if item.strip()]
    if not start_yaw_degrees:
        raise ValueError("--start_yaw_degrees must contain at least one angle")
    viewpoint_yaw_degrees = [float(item) for item in str(args.viewpoint_yaw_degrees).split(",") if item.strip()]
    if not viewpoint_yaw_degrees:
        raise ValueError("--viewpoint_yaw_degrees must contain at least one angle")
    poi_dir = Path(args.poi_dir).expanduser().resolve()

    scene_plan_counts = []
    for scene_id in habitat_scene_ids:
        pois = _load_scene_pois(Path(scene_id).parent.name, poi_dir)
        if pois:
            scene_plan_counts.append(len(pois) * len(start_yaw_degrees))
        else:
            scene_plan_counts.append(int(args.episodes_per_scene) * len(start_yaw_degrees))
    scene_task_offsets = []
    running_total = 0
    for count in scene_plan_counts:
        scene_task_offsets.append(running_total)
        running_total += int(count)
    shard_scene_plan_counts = [
        _assigned_task_count(offset, count, num_shards, shard_index)
        for offset, count in zip(scene_task_offsets, scene_plan_counts)
    ]

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": args.dataset_root,
        "scene_count": len(habitat_scene_ids),
        "scenes": [Path(scene).parent.name for scene in habitat_scene_ids],
        "episodes_per_scene": int(args.episodes_per_scene),
        "start_policy": "all_pois_fixed_yaw",
        "poi_dir": str(poi_dir),
        "start_yaw_degrees": start_yaw_degrees,
        "viewpoint_yaw_degrees": viewpoint_yaw_degrees,
        "planned_episodes_per_scene": scene_plan_counts,
        "planned_total_episodes_full": int(sum(scene_plan_counts)),
        "planned_episodes_per_scene_shard": shard_scene_plan_counts,
        "planned_total_episodes": int(sum(shard_scene_plan_counts)),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "max_steps": int(args.max_steps),
        "candidate_viewpoints": int(args.candidate_viewpoints),
        "max_cover_viewpoints": int(args.max_cover_viewpoints),
        "min_save_score": float(args.min_save_score),
        "min_save_coverage": float(args.min_save_coverage),
        "action_policy": str(args.action_policy),
        "turn_threshold_deg": float(args.turn_threshold_deg),
        "expert": "hm3d_viewpoint_cover_navmesh_oracle_v1",
    }
    metadata_path = output_dir / "metadata.json"
    if num_shards > 1:
        metadata_path = output_dir / f"metadata_shard_{shard_index:02d}_of_{num_shards:02d}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if num_shards == 1 or shard_index == 0:
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    saved = 0
    rejected = 0
    no_cover = 0
    skipped_existing = 0
    try:
        total = int(sum(shard_scene_plan_counts))
        progress = tqdm(total=total, desc="Generating HM3D viewpoint expert demos")
        for scene_idx in range(len(habitat_scene_ids)):
            assigned_scene_count = int(shard_scene_plan_counts[scene_idx])
            if assigned_scene_count <= 0:
                if args.profile_timing:
                    scene_name = Path(habitat_scene_ids[scene_idx]).parent.name
                    print(
                        "[TIMING][scene_skip_shard] "
                        f"scene={scene_idx + 1}/{len(habitat_scene_ids)} name={scene_name} "
                        f"shard={shard_index}/{num_shards}",
                        flush=True,
                    )
                continue
            scene_total_start = time.perf_counter()
            scene_name = Path(habitat_scene_ids[scene_idx]).parent.name
            t0 = time.perf_counter()
            env.reset(scene_number=scene_idx + 1, random_start=False)
            scene_reset_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            starts = _load_scene_pois(scene_name, poi_dir)
            poi_load_seconds = time.perf_counter() - t0
            if not starts:
                print(
                    f"[WARNING] No POIs for scene {scene_name}; falling back to "
                    f"{args.episodes_per_scene} largest-island starts"
                )
                t0 = time.perf_counter()
                starts = _fallback_scene_starts(env, int(args.episodes_per_scene))
                poi_load_seconds += time.perf_counter() - t0

            t0 = time.perf_counter()
            candidates = _sample_candidate_viewpoints(
                env,
                rng,
                sample_count=int(args.candidate_viewpoints),
                yaw_degrees=viewpoint_yaw_degrees,
                min_pixels=int(args.min_visible_pixels),
            )
            candidate_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            cover = _choose_viewpoint_cover(
                candidates,
                target_ids=env.scene_reward_gt_ids,
                max_viewpoints=int(args.max_cover_viewpoints),
            )
            cover_seconds = time.perf_counter() - t0
            if args.profile_timing:
                print(
                    "[TIMING][scene] "
                    f"scene={scene_idx + 1}/{len(habitat_scene_ids)} name={scene_name} "
                    f"reset={scene_reset_seconds:.2f}s poi_load={poi_load_seconds:.2f}s "
                    f"candidate_sample={candidate_seconds:.2f}s cover={cover_seconds:.2f}s "
                    f"candidates={len(candidates)} cover_vp={len(cover)} starts={len(starts)}",
                    flush=True,
                )

            if not cover:
                scene_rejected = assigned_scene_count
                no_cover += scene_rejected
                rejected += scene_rejected
                progress.set_postfix(saved=saved, rejected=rejected, no_cover=no_cover, score="0.00", cov="0.00")
                progress.update(scene_rejected)
                continue

            for start_idx, start in enumerate(starts):
                for yaw_idx, start_yaw in enumerate(start_yaw_degrees):
                    local_task_idx = start_idx * len(start_yaw_degrees) + yaw_idx
                    global_task_idx = int(scene_task_offsets[scene_idx]) + local_task_idx
                    if global_task_idx % num_shards != shard_index:
                        continue
                    episode_total_start = time.perf_counter()
                    episode_tag = (
                        f"vp_scene_{scene_idx + 1:02d}_poi_{start.poi_id:04d}_"
                        f"yaw_{int(round(start_yaw)) % 360:03d}"
                    )
                    if args.skip_existing and any((output_dir / scene_name).glob(f"{episode_tag}_score_*.npz")):
                        skipped_existing += 1
                        if args.profile_timing:
                            print(
                                "[TIMING][skip_existing] "
                                f"scene={scene_name} poi={start.poi_id} "
                                f"yaw={int(round(start_yaw)) % 360} "
                                f"shard={shard_index}/{num_shards}",
                                flush=True,
                            )
                        progress.set_postfix(
                            saved=saved,
                            skipped=skipped_existing,
                            rejected=rejected,
                            no_cover=no_cover,
                        )
                        progress.update(1)
                        continue
                    t0 = time.perf_counter()
                    obs = _reset_to_start(
                        env,
                        scene_number=scene_idx + 1,
                        start=start,
                        yaw_deg=start_yaw,
                        episode_tag=episode_tag,
                        save_debug=bool(args.save_debug),
                    )
                    episode_reset_seconds = time.perf_counter() - t0
                    start_position = np.asarray(env.sim.get_agent(0).get_state().position, dtype=np.float32)
                    t0 = time.perf_counter()
                    ordered = _order_viewpoints(start_position, cover)
                    order_seconds = time.perf_counter() - t0
                    if not ordered:
                        no_cover += 1
                        rejected += 1
                        progress.set_postfix(saved=saved, rejected=rejected, no_cover=no_cover, score="0.00", cov="0.00")
                        progress.update(1)
                        continue

                    records, final_score, final_coverage, run_timing = _run_episode_to_viewpoints(
                        env,
                        obs,
                        ordered,
                        rng,
                        max_steps=int(args.max_steps),
                        reach_dist=float(args.reach_dist),
                        max_target_steps=int(args.max_target_steps),
                        min_pixels=int(args.min_visible_pixels),
                        waypoint_reach_dist=float(args.waypoint_reach_dist),
                        action_policy=str(args.action_policy),
                        turn_threshold_deg=float(args.turn_threshold_deg),
                    )
                    quality_ok = (
                        len(records) >= int(args.min_steps)
                        and final_score >= float(args.min_save_score)
                        and final_coverage >= float(args.min_save_coverage)
                    )
                    save_seconds = 0.0
                    if quality_ok:
                        out_path = (
                            output_dir
                            / scene_name
                            / (
                                f"{episode_tag}_score_{final_score:.3f}_cov_{final_coverage:.3f}_"
                                f"vp_{len(ordered):02d}.npz"
                            )
                        )
                        t0 = time.perf_counter()
                        _save_episode(
                            out_path,
                            records,
                            env,
                            {
                                "scene_index": scene_idx,
                                "start_index": start_idx,
                                "poi_index": start.index,
                                "poi_id": start.poi_id,
                                "start_yaw_deg": float(start_yaw),
                                "requested_start_position": start.position,
                                "actual_start_position": start_position,
                                "final_score": final_score,
                                "final_coverage": final_coverage,
                            },
                        )
                        save_seconds = time.perf_counter() - t0
                        saved += 1
                    else:
                        rejected += 1
                    episode_total_seconds = time.perf_counter() - episode_total_start
                    if args.profile_timing:
                        print(
                            "[TIMING][episode] "
                            f"scene={scene_name} poi={start.poi_id} yaw={int(round(start_yaw)) % 360} "
                            f"task={global_task_idx} shard={shard_index}/{num_shards} "
                            f"accepted={int(quality_ok)} score={final_score:.3f} cov={final_coverage:.3f} "
                            f"total={episode_total_seconds:.2f}s reset={episode_reset_seconds:.2f}s "
                            f"order={order_seconds:.2f}s save={save_seconds:.2f}s "
                            f"{_format_timing(run_timing)}",
                            flush=True,
                        )
                    progress.set_postfix(
                        saved=saved,
                        rejected=rejected,
                        no_cover=no_cover,
                        score=f"{final_score:.2f}",
                        cov=f"{final_coverage:.2f}",
                        vp=len(ordered),
                        poi=start.poi_id,
                        yaw=int(round(start_yaw)) % 360,
                    )
                    progress.update(1)
            if args.profile_timing:
                print(
                    "[TIMING][scene_done] "
                    f"scene={scene_idx + 1}/{len(habitat_scene_ids)} name={scene_name} "
                    f"total={time.perf_counter() - scene_total_start:.2f}s saved={saved} "
                    f"skipped={skipped_existing} rejected={rejected} no_cover={no_cover} "
                    f"shard={shard_index}/{num_shards}",
                    flush=True,
                )
        progress.close()
    finally:
        env.close()

    print(
        "[INFO] HM3D viewpoint expert dataset complete: "
        f"saved={saved}, skipped_existing={skipped_existing}, rejected={rejected}, "
        f"no_cover={no_cover}, shard={shard_index}/{num_shards}, output={output_dir}"
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
    parser.add_argument("--poi_dir", type=str, default="/home/wgy/RL/pois")
    parser.add_argument("--start_yaw_degrees", type=str, default="0,180")
    parser.add_argument("--viewpoint_yaw_degrees", type=str, default="0,60,120,180,240,300")
    parser.add_argument(
        "--yaw_degrees",
        type=str,
        default=None,
        help="Deprecated alias for --viewpoint_yaw_degrees.",
    )
    parser.add_argument("--min_visible_pixels", type=int, default=80)
    parser.add_argument("--reach_dist", type=float, default=0.45)
    parser.add_argument("--waypoint_reach_dist", type=float, default=0.30)
    parser.add_argument("--max_target_steps", type=int, default=80)
    parser.add_argument("--min_save_score", type=float, default=0.45)
    parser.add_argument("--min_save_coverage", type=float, default=0.02)
    parser.add_argument(
        "--action_policy",
        type=str,
        default="oracle",
        choices=("oracle", "geometric"),
        help="oracle tries candidate actions with GT visibility; geometric follows the path without oracle probes.",
    )
    parser.add_argument(
        "--turn_threshold_deg",
        type=float,
        default=15.0,
        help="For geometric action policy, turn until yaw error is within this threshold before moving forward.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split all scene/POI/yaw tasks into this many shards.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Run only tasks whose global task index belongs to this shard.",
    )
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--save_debug_interval", type=int, default=100)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip episodes whose output .npz already exists in the output directory.",
    )
    parser.add_argument(
        "--profile_timing",
        action="store_true",
        help="Print per-scene and per-episode timing breakdowns for bottleneck diagnosis.",
    )
    args = parser.parse_args()
    if args.yaw_degrees is not None:
        args.viewpoint_yaw_degrees = args.yaw_degrees
    return args


if __name__ == "__main__":
    generate(parse_args())
