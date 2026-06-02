#!/usr/bin/env python3
"""Summarize IL/RL trajectory diagnostics from saved trajectory.csv files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _to_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def summarize(root: Path) -> dict:
    files = sorted(root.glob("worker_*/epoch_*/trajectory.csv"))
    scenes = []
    total_counts = {"0": 0, "1": 0, "2": 0}
    total_actions = 0
    total_move = 0.0
    total_forward = 0
    forward_zero = 0
    total_overridden = 0
    total_heading_blocked = 0
    entropy_sum = 0.0

    for path in files:
        with path.open("r", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        action_rows = [row for row in rows if row.get("action") not in (None, "")]
        if not action_rows:
            continue

        counts = {"0": 0, "1": 0, "2": 0}
        move = 0.0
        fwd = 0
        fwd_zero = 0
        overridden = 0
        heading_blocked = 0
        ent = 0.0
        last_move_step = 0
        for row in action_rows:
            action = str(row.get("action", ""))
            counts[action] = counts.get(action, 0) + 1
            total_counts[action] = total_counts.get(action, 0) + 1
            total_actions += 1
            step = int(_to_float(row.get("step")))
            moved = _to_float(row.get("moved_distance"))
            move += moved
            total_move += moved
            ent += _to_float(row.get("entropy"))
            entropy_sum += _to_float(row.get("entropy"))
            if int(_to_float(row.get("action_overridden"))) > 0:
                overridden += 1
                total_overridden += 1
            if int(_to_float(row.get("forward_heading_blocked"))) > 0:
                heading_blocked += 1
                total_heading_blocked += 1
            if moved > 1e-3:
                last_move_step = step
            if action == "2":
                fwd += 1
                total_forward += 1
                if moved <= 1e-3:
                    fwd_zero += 1
                    forward_zero += 1

        last = rows[-1]
        scene_dir = path.parent.name
        scenes.append(
            {
                "scene_dir": scene_dir,
                "steps": int(_to_float(last.get("step"))),
                "score": _to_float(last.get("score")),
                "coverage": _to_float(last.get("coverage")),
                "path_coverage": _to_float(last.get("path_coverage")),
                "visible_coverage": _to_float(last.get("visible_coverage")),
                "actions": counts,
                "move_distance": move,
                "forward_zero_frac": fwd_zero / max(fwd, 1),
                "override_frac": overridden / max(len(action_rows), 1),
                "heading_blocked_frac": heading_blocked / max(len(action_rows), 1),
                "avg_entropy": ent / max(len(action_rows), 1),
                "last_move_step": last_move_step,
            }
        )

    summary = {
        "root": str(root),
        "num_trajectories": len(scenes),
        "avg_score": sum(s["score"] for s in scenes) / max(len(scenes), 1),
        "avg_coverage": sum(s["coverage"] for s in scenes) / max(len(scenes), 1),
        "avg_path_coverage": sum(s["path_coverage"] for s in scenes) / max(len(scenes), 1),
        "avg_visible_coverage": sum(s["visible_coverage"] for s in scenes) / max(len(scenes), 1),
        "avg_move_distance": sum(s["move_distance"] for s in scenes) / max(len(scenes), 1),
        "action_counts": total_counts,
        "action_fracs": {key: value / max(total_actions, 1) for key, value in total_counts.items()},
        "forward_zero_frac": forward_zero / max(total_forward, 1),
        "override_frac": total_overridden / max(total_actions, 1),
        "heading_blocked_frac": total_heading_blocked / max(total_actions, 1),
        "avg_entropy": entropy_sum / max(total_actions, 1),
        "scenes": scenes,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Evaluation visualization directory")
    parser.add_argument("--top", type=int, default=12, help="Number of worst scenes to print")
    parser.add_argument("--json", action="store_true", help="Print full JSON summary")
    args = parser.parse_args()

    summary = summarize(Path(args.root).expanduser().resolve())
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    print(f"root: {summary['root']}")
    print(
        "trajectories={num_trajectories} avg_score={avg_score:.3f} "
        "avg_coverage={avg_coverage:.3f} path={avg_path_coverage:.3f} "
        "visible={avg_visible_coverage:.3f} avg_move={avg_move_distance:.2f}".format(**summary)
    )
    print(
        "actions={action_counts} fracs={fracs} forward_zero_frac={fzero:.3f} "
        "override_frac={override:.3f} heading_blocked_frac={blocked:.3f} "
        "avg_entropy={entropy:.3f}".format(
            action_counts=summary["action_counts"],
            fracs={k: round(v, 3) for k, v in summary["action_fracs"].items()},
            fzero=summary["forward_zero_frac"],
            override=summary["override_frac"],
            blocked=summary["heading_blocked_frac"],
            entropy=summary["avg_entropy"],
        )
    )
    print("worst scenes:")
    for scene in sorted(summary["scenes"], key=lambda item: item["score"])[: max(int(args.top), 0)]:
        print(
            "{scene_dir} step={steps} score={score:.3f} cov={coverage:.3f} "
            "path={path_coverage:.3f} vis={visible_coverage:.3f} move={move_distance:.2f} "
            "actions={actions} fwd_zero={forward_zero_frac:.3f} override={override_frac:.3f} "
            "blocked={heading_blocked_frac:.3f} last_move={last_move_step}".format(**scene)
        )


if __name__ == "__main__":
    main()
