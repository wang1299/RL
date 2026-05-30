#!/usr/bin/env python3
"""Precompute HM3D expert RGB encoder features for faster imitation training."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from components.agents.imitation_agent import ImitationAgent
from ImitationLearning.dataset.hm3d_il_dataset import list_episode_files
from train_habitat_parallel import _load_config_mapping


def _resolve_encoder_path(conf_path: str, explicit_path: str | None, use_transformer: bool) -> Path | None:
    encoder_path = explicit_path
    if encoder_path is None:
        encoder_path = _load_config_mapping(conf_path, "agent_config.yaml", "agent.json").get("encoder_path")
    if not encoder_path:
        return None

    path = Path(encoder_path).expanduser()
    candidate = path.parent / f"{path.stem}_{use_transformer}{path.suffix}"
    if candidate.exists():
        return candidate
    if path.exists():
        return path
    return path


def _load_agent(args, num_actions: int, device: torch.device) -> ImitationAgent:
    navigation_config = _load_config_mapping(args.conf_path, "navigation_config.yaml", "navigation.json")
    navigation_config["use_transformer"] = False
    print("[INFO] Navigation config:")
    print(json.dumps(navigation_config, indent=2))

    agent = ImitationAgent(
        navigation_config=navigation_config,
        num_actions=int(num_actions),
        device=device,
    )

    encoder_path = _resolve_encoder_path(args.conf_path, args.encoder_path, navigation_config["use_transformer"])
    if encoder_path is not None:
        if encoder_path.exists():
            print(f"[INFO] Loading encoder weights from {encoder_path}")
            agent.encoder.load_weights(str(encoder_path), device=str(device))
        else:
            print(f"[WARNING] Encoder checkpoint not found: {encoder_path}")

    if args.checkpoint_path_load:
        checkpoint = Path(args.checkpoint_path_load).expanduser()
        if checkpoint.exists():
            print(f"[INFO] Loading full checkpoint before caching from {checkpoint}")
            payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
            state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
            missing, unexpected = agent.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[WARNING] Missing keys while loading checkpoint: {missing}")
            if unexpected:
                print(f"[WARNING] Unexpected keys while loading checkpoint: {unexpected}")
        else:
            print(f"[WARNING] Checkpoint not found: {checkpoint}")

    agent.eval()
    for param in agent.parameters():
        param.requires_grad_(False)
    return agent


def _copy_metadata(data_dir: Path, output_dir: Path) -> None:
    for name in ("metadata.json",):
        src = data_dir / name
        dst = output_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def _first_num_actions(files: list[Path]) -> int:
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if "num_actions" in data:
                return int(data["num_actions"][0])
    return 3


def _save_npz(path: Path, compress: bool, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def _cache_one(path: Path, output_path: Path, agent: ImitationAgent, device: torch.device, args) -> dict:
    t0 = time.perf_counter()
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        rgb = data["rgb"]
        payload = {key: data[key] for key in keys if key != "rgb"}
    read_seconds = time.perf_counter() - t0

    features = []
    encode_start = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rgb), int(args.batch_size)):
            batch = rgb[start : start + int(args.batch_size)]
            rgb_tensor = agent.encoder.preprocess_rgb(list(batch)).to(device, non_blocking=True)
            feat = agent.encoder.rgb_encoder(rgb_tensor)
            features.append(feat.detach().cpu())
    encode_seconds = time.perf_counter() - encode_start

    feature_tensor = torch.cat(features, dim=0).numpy()
    if args.dtype == "float16":
        feature_array = feature_tensor.astype(np.float16)
    else:
        feature_array = feature_tensor.astype(np.float32)

    payload["rgb_features"] = feature_array
    payload["feature_dim"] = np.asarray([feature_array.shape[-1]], dtype=np.int64)
    payload["feature_dtype"] = np.asarray([0 if args.dtype == "float16" else 1], dtype=np.int64)

    save_start = time.perf_counter()
    _save_npz(output_path, bool(args.compress), payload)
    save_seconds = time.perf_counter() - save_start
    total_seconds = time.perf_counter() - t0
    return {
        "frames": int(len(rgb)),
        "read_seconds": read_seconds,
        "encode_seconds": encode_seconds,
        "save_seconds": save_seconds,
        "total_seconds": total_seconds,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Cache HM3D expert RGB features.")
    parser.add_argument("--conf_path", type=str, default="/home/wgy/RL/config")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--checkpoint_path_load", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_episode_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    if int(args.num_shards) <= 0:
        raise ValueError("--num_shards must be positive")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("--shard_index must be in [0, num_shards)")
    files = [path for idx, path in enumerate(files) if idx % int(args.num_shards) == int(args.shard_index)]
    if int(args.limit) > 0:
        files = files[: int(args.limit)]

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu_id)}")
    else:
        device = torch.device("cpu")

    print("[INFO] Starting HM3D RGB feature cache")
    print(f"[INFO] data_dir: {data_dir}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] shard: {args.shard_index}/{args.num_shards}")
    print(f"[INFO] files: {len(files)}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] batch_size: {args.batch_size}")
    print(f"[INFO] dtype: {args.dtype}")

    agent = _load_agent(args, _first_num_actions(files), device)
    _copy_metadata(data_dir, output_dir)

    processed = 0
    skipped = 0
    failed = 0
    frames = 0
    timings = {"read_seconds": 0.0, "encode_seconds": 0.0, "save_seconds": 0.0, "total_seconds": 0.0}

    progress = tqdm(files, desc="Caching HM3D RGB features")
    for path in progress:
        rel = path.relative_to(data_dir)
        output_path = output_dir / rel
        if output_path.exists() and not args.overwrite:
            skipped += 1
            progress.set_postfix(processed=processed, skipped=skipped, failed=failed)
            continue

        try:
            stats = _cache_one(path, output_path, agent, device, args)
        except Exception as exc:
            failed += 1
            print(f"[ERROR] Failed to cache {path}: {exc}")
            progress.set_postfix(processed=processed, skipped=skipped, failed=failed)
            continue

        processed += 1
        frames += int(stats["frames"])
        for key in timings:
            timings[key] += float(stats[key])
        progress.set_postfix(
            processed=processed,
            skipped=skipped,
            failed=failed,
            frames=frames,
            fps=f"{frames / max(timings['total_seconds'], 1e-6):.1f}",
        )

    summary = {
        "source_data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "processed_files": int(processed),
        "skipped_files": int(skipped),
        "failed_files": int(failed),
        "processed_frames": int(frames),
        "feature_key": "rgb_features",
        "feature_dtype": args.dtype,
        "compress": bool(args.compress),
        **timings,
    }
    summary_path = output_dir / f"feature_cache_summary_shard_{int(args.shard_index):02d}_of_{int(args.num_shards):02d}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] HM3D RGB feature cache complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
