#!/usr/bin/env python3
"""Low-frequency watchdog for a Habitat RL training log.

The watchdog waits for a target number of [STATS] entries. If the selected
average score is not above the threshold, it terminates the current training
process with SIGTERM, writes a short diagnosis, and restarts the same command
with a fresh run/log name.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


STATS_RE = re.compile(
    r"\[STATS\]\s+Avg Score:\s*([0-9.]+),\s*"
    r"Avg Coverage:\s*([0-9.]+),\s*"
    r"PathCov:\s*([0-9.]+),\s*"
    r"VisCov:\s*([0-9.]+),\s*"
    r"Avg Steps:\s*([0-9.]+),\s*"
    r"Max Score:\s*([0-9.]+)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def read_tail(path: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes), os.SEEK_SET)
        return f.read().decode("utf-8", errors="replace")


def parse_stats(text: str) -> List[Dict[str, float]]:
    stats: List[Dict[str, float]] = []
    for match in STATS_RE.finditer(text):
        stats.append(
            {
                "avg_score": float(match.group(1)),
                "avg_coverage": float(match.group(2)),
                "path_cov": float(match.group(3)),
                "vis_cov": float(match.group(4)),
                "avg_steps": float(match.group(5)),
                "max_score": float(match.group(6)),
            }
        )
    return stats


def read_proc_cmd(pid: int) -> List[str]:
    data = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [part.decode("utf-8", errors="replace") for part in data.split(b"\0") if part]


def read_proc_env(pid: int) -> Dict[str, str]:
    data = Path(f"/proc/{pid}/environ").read_bytes()
    env: Dict[str, str] = {}
    for part in data.split(b"\0"):
        if not part or b"=" not in part:
            continue
        key, value = part.split(b"=", 1)
        env[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")
    return env


def read_proc_cwd(pid: int, fallback: Path) -> Path:
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return fallback


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def replace_arg(cmd: List[str], name: str, value: str) -> List[str]:
    updated = list(cmd)
    if name in updated:
        idx = updated.index(name)
        if idx + 1 < len(updated):
            updated[idx + 1] = value
        else:
            updated.append(value)
    else:
        updated.extend([name, value])
    return updated


def build_restart_cmd(cmd: List[str], repo_root: Path, run_name: str) -> tuple[List[str], Path, Path]:
    frames_dir = repo_root / "train_png" / run_name
    log_path = repo_root / "train_log" / f"{run_name}.log"
    updated = replace_arg(cmd, "--save_frames_to", str(frames_dir))
    return updated, frames_dir, log_path


def write_diagnosis(
    path: Path,
    stats: List[Dict[str, float]],
    threshold: float,
    source_log: Path,
    old_pid: int,
    new_pid: Optional[int],
    new_log: Optional[Path],
) -> None:
    second = stats[1] if len(stats) > 1 else None
    lines = [
        f"time: {utc_now()}",
        f"source_log: {source_log}",
        f"old_pid: {old_pid}",
        f"threshold: avg_score > {threshold}",
        f"stats_count: {len(stats)}",
        f"second_stats: {json.dumps(second, sort_keys=True)}",
        "",
        "diagnosis:",
        "- The run did not clear the configured two-round score gate.",
        "- The final success threshold is stricter than the previous run, so terminal success reward is sparse.",
        "- This run was started from the IL checkpoint, not from the previous interrupted RL checkpoint.",
        "- Several early episodes reached old-threshold score ranges but ran to max steps under the stricter target.",
        "",
        f"new_pid: {new_pid if new_pid is not None else 'not_started'}",
        f"new_log: {new_log if new_log is not None else 'not_started'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def terminate_process(pid: int, wait_seconds: int) -> bool:
    if not process_alive(pid):
        log(f"pid {pid} is already stopped")
        return True
    log(f"sending SIGTERM to pid {pid}")
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not process_alive(pid):
            log(f"pid {pid} exited after SIGTERM")
            return True
        time.sleep(5)
    log(f"pid {pid} still alive after {wait_seconds}s; leaving it for manual review")
    return False


def start_training(cmd: List[str], env: Dict[str, str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()
    return int(proc.pid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("/root/RL"))
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-stats", type=int, default=2)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--initial-delay-seconds", type=int, default=0)
    parser.add_argument("--term-wait-seconds", type=int, default=900)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--once", action="store_true", help="Run one low-frequency check and exit.")
    args = parser.parse_args()

    if args.min_stats < 1:
        raise ValueError("--min-stats must be >= 1")

    source_cmd = read_proc_cmd(args.pid)
    source_env = read_proc_env(args.pid)
    source_cwd = read_proc_cwd(args.pid, args.repo_root)
    if not source_cmd:
        raise RuntimeError(f"Could not read command line for pid {args.pid}")

    state_file = args.state_file or (args.repo_root / "train_log" / "rl_score_watchdog_state.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    if state_file.exists():
        try:
            previous = json.loads(state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        if previous.get("status") in {"passed", "restarted", "term_timeout", "source_stopped"}:
            log(f"final state already recorded: {previous.get('status')}")
            return 0

    log(
        "watchdog started "
        f"pid={args.pid} log={args.log} threshold={args.threshold} "
        f"min_stats={args.min_stats} interval={args.interval_seconds}s once={args.once}"
    )
    if args.initial_delay_seconds > 0:
        log(f"initial delay {args.initial_delay_seconds}s")
        time.sleep(args.initial_delay_seconds)

    while True:
        if not args.log.exists():
            log(f"log not found: {args.log}")
            time.sleep(args.interval_seconds)
            continue

        stats = parse_stats(read_tail(args.log))
        latest = stats[-1] if stats else None
        state = {
            "time": utc_now(),
            "source_pid": args.pid,
            "source_log": str(args.log),
            "stats_count": len(stats),
            "latest_stats": latest,
            "threshold": args.threshold,
            "status": "waiting",
        }
        if not process_alive(args.pid) and len(stats) < args.min_stats:
            state["status"] = "source_stopped"
            state_file.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            log(f"source pid {args.pid} stopped before {args.min_stats} stats")
            return 3
        state_file.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        log(f"stats_count={len(stats)} latest={latest}")
        if len(stats) < args.min_stats:
            if args.once:
                return 0
            time.sleep(args.interval_seconds)
            continue

        selected = stats[args.min_stats - 1]
        selected_score = float(selected["avg_score"])
        if selected_score > args.threshold:
            state["status"] = "passed"
            state["selected_stats"] = selected
            state_file.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            log(f"score gate passed: {selected_score:.3f} > {args.threshold:.3f}")
            return 0

        log(f"score gate failed: {selected_score:.3f} <= {args.threshold:.3f}")
        stopped = terminate_process(args.pid, args.term_wait_seconds)
        if not stopped and process_alive(args.pid):
            state["status"] = "term_timeout"
            state["selected_stats"] = selected
            state_file.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 2

        run_name = f"parallel_train_{stamp()}_watchdog_restart"
        restart_cmd, frames_dir, new_log = build_restart_cmd(source_cmd, args.repo_root, run_name)
        frames_dir.mkdir(parents=True, exist_ok=True)
        diagnosis_path = args.repo_root / "train_log" / f"{run_name}_diagnosis.txt"
        new_pid = start_training(restart_cmd, source_env, source_cwd, new_log)
        write_diagnosis(diagnosis_path, stats, args.threshold, args.log, args.pid, new_pid, new_log)

        state.update(
            {
                "status": "restarted",
                "selected_stats": selected,
                "new_pid": new_pid,
                "new_log": str(new_log),
                "diagnosis": str(diagnosis_path),
            }
        )
        state_file.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log(f"restarted training pid={new_pid} log={new_log}")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"watchdog failed: {exc}")
        raise
