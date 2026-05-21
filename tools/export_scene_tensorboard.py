#!/usr/bin/env python3
"""Export per-scene training metrics from a parallel Habitat log to TensorBoard."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from torch.utils.tensorboard import SummaryWriter


LINE_RE = re.compile(
    r"\[Epoch\s+(?P<epoch>\d+)\].*?"
    r"AssignedEp=(?P<assigned_episode>\d+),\s+"
    r"CompletedEp=(?P<completed_episode>\d+),\s+"
    r"WorkerEp=(?P<worker_episode>\d+),\s+"
    r"Scene=(?P<scene_index>\d+)/(?P<scene_count>\d+)\((?P<scene_name>[^)]*)\),\s+"
    r"SceneEp=(?P<scene_episode>\d+),\s+"
    r"Score=(?P<score>-?\d+(?:\.\d+)?),\s+"
    r"Coverage=(?P<coverage>-?\d+(?:\.\d+)?),\s+"
    r"GT=(?P<gt_found>\d+)/(?P<gt_total>\d+),\s+"
    r"Steps=(?P<steps>\d+)"
)


@dataclass(frozen=True)
class SceneMetric:
    epoch: int
    assigned_episode: int
    completed_episode: int
    worker_episode: int
    scene_index: int
    scene_count: int
    scene_name: str
    scene_episode: int
    score: float
    coverage: float
    gt_found: int
    gt_total: int
    steps: int

    @property
    def key(self) -> str:
        return f"{self.epoch}:{self.scene_index}"

    @property
    def scene_tag(self) -> str:
        return f"scene_{self.scene_index:02d}_{self.scene_name}"

    @property
    def gt_recall(self) -> float:
        if self.gt_total <= 0:
            return 0.0
        return min(max(self.gt_found / self.gt_total, 0.0), 1.0)


def parse_metric(line: str) -> Optional[SceneMetric]:
    match = LINE_RE.search(line)
    if match is None:
        return None
    group = match.groupdict()
    return SceneMetric(
        epoch=int(group["epoch"]),
        assigned_episode=int(group["assigned_episode"]),
        completed_episode=int(group["completed_episode"]),
        worker_episode=int(group["worker_episode"]),
        scene_index=int(group["scene_index"]),
        scene_count=int(group["scene_count"]),
        scene_name=group["scene_name"],
        scene_episode=int(group["scene_episode"]),
        score=float(group["score"]),
        coverage=float(group["coverage"]),
        gt_found=int(group["gt_found"]),
        gt_total=int(group["gt_total"]),
        steps=int(group["steps"]),
    )


def default_output_dir(log_file: Path) -> Path:
    repo_root = log_file.resolve().parents[1]
    return repo_root / "RL_training" / "runs" / "scene_tensorboard" / log_file.stem


def load_state(state_file: Path, log_file: Path) -> Tuple[int, set]:
    if not state_file.exists():
        return 0, set()
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return 0, set()
    if state.get("log_file") != str(log_file.resolve()):
        return 0, set()
    offset = int(state.get("offset", 0))
    seen = set(str(k) for k in state.get("seen_keys", []))
    try:
        if log_file.stat().st_size < offset:
            return 0, set()
    except OSError:
        return 0, set()
    return offset, seen


def save_state(state_file: Path, log_file: Path, offset: int, seen: Iterable[str]) -> None:
    payload = {
        "log_file": str(log_file.resolve()),
        "offset": int(offset),
        "seen_keys": sorted(seen),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    state_file.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def read_new_metrics(log_file: Path, offset: int) -> Tuple[List[SceneMetric], int]:
    metrics: List[SceneMetric] = []
    with log_file.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(offset)
        for line in handle:
            metric = parse_metric(line)
            if metric is not None:
                metrics.append(metric)
        return metrics, handle.tell()


def write_metrics(writer: SummaryWriter, metrics: List[SceneMetric], seen_keys: set) -> int:
    written = 0
    new_by_epoch: Dict[int, List[SceneMetric]] = {}

    for metric in metrics:
        if metric.key in seen_keys:
            continue
        seen_keys.add(metric.key)
        new_by_epoch.setdefault(metric.epoch, []).append(metric)

        tag = metric.scene_tag
        writer.add_scalar(f"scene_score/{tag}", metric.score, metric.epoch)
        writer.add_scalar(f"scene_coverage/{tag}", metric.coverage, metric.epoch)
        writer.add_scalar(f"scene_steps/{tag}", metric.steps, metric.epoch)
        writer.add_scalar(f"scene_gt_recall/{tag}", metric.gt_recall, metric.epoch)
        writer.add_scalar(f"scene_gt_found/{tag}", metric.gt_found, metric.epoch)
        writer.add_scalar(f"scene_gt_total/{tag}", metric.gt_total, metric.epoch)
        writer.add_text(
            f"scene_meta/{tag}",
            json.dumps(asdict(metric), ensure_ascii=False, indent=2),
            metric.epoch,
        )
        written += 1

    for epoch, epoch_metrics in sorted(new_by_epoch.items()):
        writer.add_scalar(
            "scene_epoch_summary/avg_score",
            sum(m.score for m in epoch_metrics) / len(epoch_metrics),
            epoch,
        )
        writer.add_scalar(
            "scene_epoch_summary/avg_coverage",
            sum(m.coverage for m in epoch_metrics) / len(epoch_metrics),
            epoch,
        )
        writer.add_scalar("scene_epoch_summary/completed_scenes", len(epoch_metrics), epoch)

    writer.flush()
    return written


def run_once(log_file: Path, output_dir: Path) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "scene_tensorboard_state.json"
    offset, seen_keys = load_state(state_file, log_file)
    metrics, new_offset = read_new_metrics(log_file, offset)
    with SummaryWriter(str(output_dir)) as writer:
        written = write_metrics(writer, metrics, seen_keys)
    save_state(state_file, log_file, new_offset, seen_keys)
    return written, len(seen_keys)


def follow(log_file: Path, output_dir: Path, poll_interval: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "scene_tensorboard_state.json"
    offset, seen_keys = load_state(state_file, log_file)
    with SummaryWriter(str(output_dir)) as writer:
        while True:
            metrics, offset = read_new_metrics(log_file, offset)
            written = write_metrics(writer, metrics, seen_keys)
            if written:
                print(
                    f"[scene-tb] wrote {written} scene metrics "
                    f"(total={len(seen_keys)}, offset={offset})",
                    flush=True,
                )
                save_state(state_file, log_file, offset, seen_keys)
            time.sleep(poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse parallel Habitat training logs and export per-scene score, "
            "coverage, step, and GT recall curves to TensorBoard."
        )
    )
    parser.add_argument(
        "--log-file",
        required=True,
        type=Path,
        help="Path to parallel_train_*.log.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="TensorBoard output directory. Defaults to RL_training/runs/scene_tensorboard/<log_stem>.",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep watching the log and append new scene metrics as training continues.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between log checks when --follow is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = args.log_file.expanduser().resolve()
    if not log_file.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_file}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else default_output_dir(log_file)

    if args.follow:
        print(f"[scene-tb] following {log_file}")
        print(f"[scene-tb] writing TensorBoard events to {output_dir}")
        follow(log_file, output_dir, args.poll_interval)
    else:
        written, total = run_once(log_file, output_dir)
        print(f"[scene-tb] wrote {written} new scene metrics; total tracked scene-epochs={total}")
        print(f"[scene-tb] TensorBoard dir: {output_dir}")


if __name__ == "__main__":
    main()
