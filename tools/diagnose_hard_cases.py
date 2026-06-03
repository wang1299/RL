#!/usr/bin/env python3
"""Compare expert demos and RL rollouts for selected HM3D hard scenes."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


HARD_SCENES = {
    37: "00598-mt9H8KcxRKD",
    45: "00712-HZ2iMMBsBQ9",
    50: "00758-HfMobPm86Xn",
    32: "00466-xAHnY3QzFUN",
    46: "00733-GtM3JtRvvvR",
}


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def scalar(array: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(array)
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def circular_yaw_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def path_distance(points: np.ndarray, n: int | None = None) -> float:
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0
    if n is not None and n > 0:
        points = points[: min(n, points.shape[0])]
    diffs = np.diff(points.astype(np.float64), axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def summarize(values: list[float]) -> str:
    if not values:
        return "-"
    arr = np.asarray(values, dtype=np.float64)
    return f"{arr.mean():.3f}/{np.median(arr):.3f}/{arr.min():.3f}-{arr.max():.3f}"


def action_fractions(actions: list[int]) -> tuple[float, float, float]:
    if not actions:
        return 0.0, 0.0, 0.0
    total = len(actions)
    counts = Counter(actions)
    return counts[0] / total, counts[1] / total, counts[2] / total


def load_experts(expert_dir: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    demos: dict[int, list[dict[str, Any]]] = defaultdict(list)
    starts: dict[int, list[dict[str, Any]]] = defaultdict(list)

    scene_lookup = {scene_id: idx for idx, scene_id in HARD_SCENES.items()}
    for path in expert_dir.rglob("*.npz"):
        path_text = str(path)
        scene_idx = None
        for scene_id, idx in scene_lookup.items():
            if scene_id in path_text:
                scene_idx = idx
                break
        if scene_idx is None:
            continue
        try:
            data = np.load(path, allow_pickle=False)
        except Exception:
            continue

        num_actions = int(scalar(data.get("num_actions"), 0))
        actions_arr = np.asarray(data.get("actions", []), dtype=np.int64)
        if num_actions <= 0:
            num_actions = int(actions_arr.shape[0])
        actions = actions_arr[:num_actions].astype(int).tolist()

        pos = np.asarray(data.get("agent_pos", []), dtype=np.float32)
        start_pos = np.asarray(data.get("actual_start_position", []), dtype=np.float32)
        start_yaw = scalar(data.get("start_yaw_deg"), 0.0) % 360.0
        poi_id = int(scalar(data.get("poi_id"), -1))

        demo = {
            "path": path,
            "score": scalar(data.get("final_score")),
            "coverage": scalar(data.get("final_coverage")),
            "steps": num_actions,
            "move": path_distance(pos, num_actions),
            "actions": actions,
            "poi_id": poi_id,
            "yaw": start_yaw,
            "start_pos": start_pos,
        }
        demos[scene_idx].append(demo)
        if start_pos.shape[0] >= 3:
            starts[scene_idx].append(
                {
                    "x": float(start_pos[0]),
                    "z": float(start_pos[2]),
                    "yaw": start_yaw,
                    "poi_id": poi_id,
                    "path": path,
                }
            )
    return demos, starts


def nearest_expert_start(
    starts: list[dict[str, Any]],
    x: float,
    z: float,
    yaw: float,
) -> tuple[float | None, float | None, int | None]:
    if not starts:
        return None, None, None
    best = None
    for start in starts:
        dist = math.hypot(x - start["x"], z - start["z"])
        yaw_diff = circular_yaw_diff(yaw % 360.0, start["yaw"] % 360.0)
        key = (dist, yaw_diff)
        if best is None or key < best[0]:
            best = (key, start)
    assert best is not None
    return best[0][0], best[0][1], int(best[1]["poi_id"])


def load_rl_rollouts(rl_dir: Path, starts: dict[int, list[dict[str, Any]]]) -> dict[int, list[dict[str, Any]]]:
    rollouts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    scene_re = re.compile(r"scene_(\d+)_([A-Za-z0-9-]+)")

    for csv_path in rl_dir.rglob("trajectory.csv"):
        match = scene_re.search(str(csv_path.parent))
        if not match:
            continue
        scene_idx = int(match.group(1))
        if scene_idx not in HARD_SCENES:
            continue
        rows: list[dict[str, str]] = []
        try:
            with csv_path.open("r", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not rows:
            continue

        first = rows[0]
        last = rows[-1]
        actions = [int(fnum(row.get("action"), -1)) for row in rows if fnum(row.get("action"), -1) >= 0]
        moved = [fnum(row.get("moved_distance"), 0.0) for row in rows]
        zero_forward = 0
        forward_count = 0
        for row in rows:
            action = int(fnum(row.get("action"), -1))
            if action == 2:
                forward_count += 1
                if fnum(row.get("moved_distance"), 0.0) < 1e-4:
                    zero_forward += 1
        nearest_dist, yaw_diff, poi_id = nearest_expert_start(
            starts.get(scene_idx, []),
            fnum(first.get("x")),
            fnum(first.get("z")),
            fnum(first.get("yaw_deg")),
        )
        rollouts[scene_idx].append(
            {
                "path": csv_path,
                "score": fnum(last.get("score")),
                "coverage": fnum(last.get("coverage")),
                "path_coverage": fnum(last.get("path_coverage")),
                "visible_coverage": fnum(last.get("visible_coverage")),
                "steps": len(rows),
                "move": sum(moved),
                "actions": actions,
                "start_x": fnum(first.get("x")),
                "start_z": fnum(first.get("z")),
                "start_yaw": fnum(first.get("yaw_deg")) % 360.0,
                "nearest_expert_dist": nearest_dist,
                "nearest_expert_yaw_diff": yaw_diff,
                "nearest_poi_id": poi_id,
                "forward_zero_frac": zero_forward / forward_count if forward_count else 0.0,
                "override_frac": mean([fnum(row.get("action_overridden"), 0.0) for row in rows]),
                "blocked_frac": mean([fnum(row.get("forward_heading_blocked"), 0.0) for row in rows]),
                "entropy": mean([fnum(row.get("entropy"), 0.0) for row in rows]),
            }
        )
    return rollouts


def print_scene_report(
    scene_idx: int,
    experts: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
) -> None:
    scene_id = HARD_SCENES[scene_idx]
    print(f"\nScene {scene_idx:02d} {scene_id}")
    print("-" * 92)
    if experts:
        expert_actions = [a for demo in experts for a in demo["actions"]]
        e_left, e_right, e_forward = action_fractions(expert_actions)
        yaw_counts = Counter(int(round(demo["yaw"])) for demo in experts)
        poi_count = len({demo["poi_id"] for demo in experts})
        print(
            "expert: "
            f"n={len(experts)} poi={poi_count} "
            f"score mean/med/min-max={summarize([d['score'] for d in experts])} "
            f"cov(old)={summarize([d['coverage'] for d in experts])} "
            f"move={summarize([d['move'] for d in experts])} "
            f"actions L/R/F={e_left:.2f}/{e_right:.2f}/{e_forward:.2f} "
            f"yaw_counts={dict(sorted(yaw_counts.items()))}"
        )
    else:
        print("expert: no demos found")

    if rollouts:
        rl_actions = [a for rollout in rollouts for a in rollout["actions"]]
        r_left, r_right, r_forward = action_fractions(rl_actions)
        start_dists = [r["nearest_expert_dist"] for r in rollouts if r["nearest_expert_dist"] is not None]
        yaw_diffs = [r["nearest_expert_yaw_diff"] for r in rollouts if r["nearest_expert_yaw_diff"] is not None]
        print(
            "rl:     "
            f"n={len(rollouts)} "
            f"score mean/med/min-max={summarize([r['score'] for r in rollouts])} "
            f"cov={summarize([r['coverage'] for r in rollouts])} "
            f"path_cov={summarize([r['path_coverage'] for r in rollouts])} "
            f"vis_cov={summarize([r['visible_coverage'] for r in rollouts])} "
            f"move={summarize([r['move'] for r in rollouts])} "
            f"actions L/R/F={r_left:.2f}/{r_right:.2f}/{r_forward:.2f}"
        )
        if start_dists:
            close = sum(1 for d in start_dists if d <= 0.25)
            yaw_ok = sum(1 for d in yaw_diffs if d <= 5.0)
            print(
                "start:  "
                f"nearest_expert_dist mean/max={mean(start_dists):.3f}/{max(start_dists):.3f}m "
                f"close<=0.25m={close}/{len(start_dists)} "
                f"yaw_diff mean/max={mean(yaw_diffs):.1f}/{max(yaw_diffs):.1f}deg "
                f"yaw_ok<=5deg={yaw_ok}/{len(yaw_diffs)}"
            )

        worst = sorted(rollouts, key=lambda r: r["score"])[:3]
        print("worst rl:")
        for r in worst:
            l_frac, rr_frac, f_frac = action_fractions(r["actions"])
            rel = r["path"].parent.relative_to(r["path"].parents[3])
            print(
                f"  score={r['score']:.3f} cov={r['coverage']:.3f} "
                f"path={r['path_coverage']:.3f} vis={r['visible_coverage']:.3f} "
                f"move={r['move']:.1f} L/R/F={l_frac:.2f}/{rr_frac:.2f}/{f_frac:.2f} "
                f"fwd_zero={r['forward_zero_frac']:.2f} blocked={r['blocked_frac']:.2f} "
                f"override={r['override_frac']:.2f} "
                f"start_match={r['nearest_expert_dist'] if r['nearest_expert_dist'] is not None else -1:.2f}m/"
                f"{r['nearest_expert_yaw_diff'] if r['nearest_expert_yaw_diff'] is not None else -1:.0f}deg "
                f"{rel}"
            )
    else:
        print("rl:     no rollouts found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expert-dir",
        type=Path,
        default=Path("/root/RL/components/data/hm3d_viewpoint_il_dataset_poi4yaw_geometric_20260524_172033_features"),
    )
    parser.add_argument(
        "--rl-dir",
        type=Path,
        default=Path("/root/RL/train_png/parallel_train_20260531_195606"),
    )
    args = parser.parse_args()

    experts, starts = load_experts(args.expert_dir)
    rollouts = load_rl_rollouts(args.rl_dir, starts)

    print(f"expert_dir={args.expert_dir}")
    print(f"rl_dir={args.rl_dir}")
    for scene_idx in [37, 45, 50, 32, 46]:
        print_scene_report(scene_idx, experts.get(scene_idx, []), rollouts.get(scene_idx, []))


if __name__ == "__main__":
    main()
