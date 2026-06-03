#!/usr/bin/env python3
"""Train behavior cloning on Habitat/HM3D expert demonstrations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from components.agents.imitation_agent import ImitationAgent
from ImitationLearning.dataset.hm3d_il_dataset import HM3DImitationLearningDataset
from ImitationLearning.runner.hm3d_il_train_runner import HM3DILTrainRunner
from train_habitat_parallel import _load_config_mapping


def parse_args():
    parser = argparse.ArgumentParser(description="Train HM3D/Habitat imitation policy.")
    parser.add_argument("--conf_path", type=str, default="/root/RL/config")
    parser.add_argument("--data_dir", type=str, default="/root/RL/components/data/hm3d_il_dataset")
    parser.add_argument("--save_dir", type=str, default="/root/RL/components/data/model_weights/hm3d_imitation")
    parser.add_argument("--checkpoint_path_load", type=str, default=None)
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_epoch_samples", type=int, default=0)
    parser.add_argument("--val_epoch_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--val_split_mode", choices=["file", "scene", "window"], default="file")
    parser.add_argument("--train_sampling_mode", choices=["scene_sqrt", "none"], default="scene_sqrt")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    navigation_config = _load_config_mapping(args.conf_path, "navigation_config.yaml", "navigation.json")
    navigation_config["use_transformer"] = False

    dataset = HM3DImitationLearningDataset(args.data_dir, seq_len=args.seq_len)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu_id)}")
    else:
        device = torch.device("cpu")
    print(f"[INFO] HM3D IL dataset windows={len(dataset)} files={len(dataset.files)} actions={dataset.num_actions}")
    print(f"[INFO] Using device: {device}")
    print("[INFO] Navigation config:")
    print(json.dumps(navigation_config, indent=2))

    agent = ImitationAgent(
        navigation_config=navigation_config,
        num_actions=int(dataset.num_actions),
        device=device,
    )

    encoder_path = args.encoder_path
    if encoder_path is None:
        encoder_path = _load_config_mapping(args.conf_path, "agent_config.yaml", "agent.json").get("encoder_path")
    if encoder_path:
        encoder_path = Path(encoder_path).expanduser()
        candidate = encoder_path.parent / f"{encoder_path.stem}_{navigation_config['use_transformer']}{encoder_path.suffix}"
        if candidate.exists():
            print(f"[INFO] Loading encoder weights from {candidate}")
            agent.encoder.load_weights(str(candidate), device=str(device))
        elif encoder_path.exists():
            print(f"[INFO] Loading encoder weights from {encoder_path}")
            agent.encoder.load_weights(str(encoder_path), device=str(device))
        else:
            print(f"[WARNING] Encoder checkpoint not found: {encoder_path}")

    if args.checkpoint_path_load:
        checkpoint = Path(args.checkpoint_path_load).expanduser()
        if checkpoint.exists():
            print(f"[INFO] Loading HM3D IL checkpoint from {checkpoint}")
            payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
            state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
            agent.load_state_dict(state_dict)
        else:
            print(f"[WARNING] Checkpoint not found: {checkpoint}")

    runner = HM3DILTrainRunner(
        agent=agent,
        dataset=dataset,
        device=device,
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        val_split=float(args.val_split),
        freeze_encoder=bool(args.freeze_encoder),
        label_smoothing=float(args.label_smoothing),
        early_stopping_patience=int(args.early_stopping_patience),
        split_seed=int(args.split_seed),
        val_split_mode=args.val_split_mode,
        train_sampling_mode=args.train_sampling_mode,
        train_epoch_samples=int(args.train_epoch_samples) if int(args.train_epoch_samples) > 0 else None,
        val_epoch_samples=int(args.val_epoch_samples) if int(args.val_epoch_samples) > 0 else None,
        num_workers=int(args.num_workers),
    )
    runner.run(num_epochs=int(args.epochs), save_folder=args.save_dir)


if __name__ == "__main__":
    main()
