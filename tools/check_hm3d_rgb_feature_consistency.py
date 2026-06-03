#!/usr/bin/env python3
"""Compare cached HM3D RGB features against a live encoder output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from components.agents.imitation_agent import ImitationAgent
from train_habitat_parallel import _load_config_mapping


def _load_agent(
    conf_path: str,
    checkpoint_path: str | None,
    encoder_path: str | None,
    device: torch.device,
) -> ImitationAgent:
    navigation_config = _load_config_mapping(conf_path, "navigation_config.yaml", "navigation.json")
    navigation_config["use_transformer"] = True

    agent = ImitationAgent(
        navigation_config=navigation_config,
        num_actions=3,
        device=device,
    )
    if encoder_path:
        agent.encoder.load_weights(encoder_path, device=str(device))
    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        missing, unexpected = agent.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys while loading checkpoint: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys while loading checkpoint: {unexpected}")
    agent.to(device)
    agent.eval()
    return agent


def _choose_pairs(raw_dir: Path, feature_dir: Path, max_files: int):
    feature_files = sorted(feature_dir.glob("**/*.npz"))
    pairs = []
    for feature_path in feature_files:
        try:
            rel = feature_path.relative_to(feature_dir)
        except ValueError:
            continue
        raw_path = raw_dir / rel
        if raw_path.exists():
            pairs.append((raw_path, feature_path))
        if len(pairs) >= max_files:
            break
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--encoder-path", default=None)
    parser.add_argument("--conf-path", default="/root/RL/config")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--frames-per-file", type=int, default=4)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    feature_dir = Path(args.feature_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    encoder_path = Path(args.encoder_path).expanduser().resolve() if args.encoder_path else None
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    if checkpoint is None and encoder_path is None:
        raise ValueError("Pass --checkpoint, --encoder-path, or both.")

    pairs = _choose_pairs(raw_dir, feature_dir, max(int(args.max_files), 1))
    if not pairs:
        raise FileNotFoundError(f"No matching raw/feature npz pairs under {raw_dir} and {feature_dir}")

    agent = _load_agent(
        args.conf_path,
        str(checkpoint) if checkpoint is not None else None,
        str(encoder_path) if encoder_path is not None else None,
        device,
    )

    cosine_values = []
    l2_values = []
    max_abs_values = []
    checked = 0
    examples = []

    with torch.inference_mode():
        for raw_path, feature_path in pairs:
            with np.load(raw_path, allow_pickle=False) as raw, np.load(feature_path, allow_pickle=False) as feat:
                rgb = raw["rgb"]
                cached = feat["rgb_features"].astype(np.float32)
                n = min(len(rgb), len(cached))
                if n <= 0:
                    continue
                frames = max(int(args.frames_per_file), 1)
                indices = np.linspace(0, n - 1, num=min(frames, n), dtype=np.int64)
                rgb_tensor = agent.encoder.preprocess_rgb([rgb[int(i)] for i in indices]).to(device)
                live = agent.encoder.rgb_encoder(rgb_tensor).detach().cpu().float()
                cached_t = torch.as_tensor(cached[indices], dtype=torch.float32)

            cos = F.cosine_similarity(live, cached_t, dim=-1)
            l2 = torch.linalg.vector_norm(live - cached_t, dim=-1)
            max_abs = torch.max(torch.abs(live - cached_t), dim=-1).values
            cosine_values.extend(float(v) for v in cos)
            l2_values.extend(float(v) for v in l2)
            max_abs_values.extend(float(v) for v in max_abs)
            checked += int(len(indices))
            if len(examples) < 3:
                examples.append(
                    {
                        "file": str(feature_path.relative_to(feature_dir)),
                        "indices": [int(i) for i in indices.tolist()],
                        "cosine": [round(float(v), 6) for v in cos.tolist()],
                        "l2": [round(float(v), 6) for v in l2.tolist()],
                    }
                )

    summary = {
        "checked_frames": checked,
        "checked_files": len(pairs),
        "cosine_mean": float(np.mean(cosine_values)) if cosine_values else None,
        "cosine_min": float(np.min(cosine_values)) if cosine_values else None,
        "l2_mean": float(np.mean(l2_values)) if l2_values else None,
        "l2_max": float(np.max(l2_values)) if l2_values else None,
        "max_abs_mean": float(np.mean(max_abs_values)) if max_abs_values else None,
        "max_abs_max": float(np.max(max_abs_values)) if max_abs_values else None,
        "examples": examples,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
