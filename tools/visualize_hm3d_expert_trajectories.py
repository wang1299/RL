#!/usr/bin/env python3
"""Visualize HM3D imitation expert trajectories saved as NPZ episodes."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from components.environments.habitat_env import HabitatEnv
from ImitationLearning.scripts.generate_hm3d_expert_dataset import _allowed_env_kwargs
from train_habitat_parallel import (
    _build_habitat_dataset_config,
    _load_config_mapping,
    _resolve_habitat_scene_list,
)


DEFAULT_DATA_DIR = (
    "/root/RL/components/data/"
    "hm3d_viewpoint_il_dataset_poi4yaw_geometric_20260524_172033"
)
DEFAULT_OUT_DIR = "/root/RL/train_png/expert_trajectory_visualization_20260524_172033"


def _scene_from_path(path: Path) -> str:
    for part in path.parts:
        if re.match(r"^\d{5}-", part):
            return part
    return "unknown_scene"


def _yaw_arrow(yaw_deg: float, length: float = 0.45):
    rad = math.radians(float(yaw_deg))
    # Habitat yaw=0 faces -Z, left turn increases yaw.
    return -math.sin(rad) * length, -math.cos(rad) * length


def _world_to_topdown_px(x: float, z: float, shape, bounds):
    min_b, max_b = bounds
    min_x, max_x = float(min_b[0]), float(max_b[0])
    min_z, max_z = float(min_b[2]), float(max_b[2])
    h, w = int(shape[0]), int(shape[1])
    if max_x <= min_x or max_z <= min_z:
        return None
    col = int(np.clip((float(x) - min_x) / (max_x - min_x) * w, 0, w - 1))
    row = int(np.clip((float(z) - min_z) / (max_z - min_z) * h, 0, h - 1))
    return col, row


def _load_topdown_for_scene(
    scene: str,
    *,
    dataset_root: str,
    conf_path: str,
    meters_per_pixel: float | None,
    gpu_id: int,
):
    scene_ids, missing = _resolve_habitat_scene_list(dataset_root, single_scene=scene, scene_list=None)
    if missing or not scene_ids:
        raise RuntimeError(f"Could not resolve scene {scene!r}; missing={missing}")

    env_config = _load_config_mapping(conf_path, "env_config.yaml", "env.json")
    if meters_per_pixel is not None:
        env_config["topdown_meters_per_pixel"] = float(meters_per_pixel)
    with tempfile.TemporaryDirectory(prefix="hm3d_expert_topdown_") as tmpdir:
        dataset_config = _build_habitat_dataset_config(
            str(Path(dataset_root) / "hm3d_annotated_basis.scene_dataset_config.json"),
            scene_ids,
            dataset_root,
            output_dir=tmpdir,
        )
        env_kwargs = _allowed_env_kwargs(env_config)
        env_kwargs.update(
            {
                "render": False,
                "use_detector": False,
                "detector": None,
                "save_debug_path": None,
            }
        )
        env = HabitatEnv(
            dataset_root=dataset_root,
            config_file=dataset_config,
            scene_id=scene_ids[0],
            scene_ids=scene_ids,
            gpu_device_id=int(gpu_id),
            **env_kwargs,
        )
        try:
            env.reset(scene_number=1, random_start=False)
            env._prepare_topdown_base_map()
            if env.topdown_base_img is None:
                raise RuntimeError(f"Failed to build topdown map for {scene}")
            base = env.topdown_base_img.copy()
            shape = tuple(env.topdown_shape)
            bounds = env.topdown_bounds
        finally:
            try:
                env.close()
            except Exception:
                pass
    return base, shape, bounds


def _thin_indices(n: int, max_items: int) -> np.ndarray:
    if n <= max_items:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_items).astype(np.int64))


def _load_scalar(data, key: str, default=float("nan")) -> float:
    if key not in data.files:
        return float(default)
    arr = np.asarray(data[key])
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def plot_episode(path: Path, out_path: Path) -> None:
    data = np.load(path, allow_pickle=True)
    pose = np.asarray(data["pose"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    score = np.asarray(data["score"], dtype=np.float32)
    coverage = np.asarray(data["coverage"], dtype=np.float32)

    x = pose[:, 0]
    z = pose[:, 2]
    yaw = pose[:, 3]
    final_score = _load_scalar(data, "final_score")
    final_cov = _load_scalar(data, "final_coverage")
    poi_id = int(_load_scalar(data, "poi_id", -1))
    start_yaw = _load_scalar(data, "start_yaw_deg")
    scene = _scene_from_path(path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 6.5), dpi=160)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.0, 1.0], height_ratios=[1, 1])
    ax_path = fig.add_subplot(grid[:, 0])
    ax_score = fig.add_subplot(grid[0, 1:])
    ax_actions = fig.add_subplot(grid[1, 1:])

    ax_path.plot(x, z, color="#1f77b4", linewidth=1.8, alpha=0.9)
    idx = _thin_indices(len(x), 36)
    if len(idx):
        dx, dz = np.vectorize(_yaw_arrow)(yaw[idx], 0.28)
        ax_path.quiver(
            x[idx],
            z[idx],
            dx,
            dz,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
            color="#2ca02c",
            alpha=0.75,
        )
    ax_path.scatter([x[0]], [z[0]], s=90, c="#2ca02c", marker="o", label="start", zorder=5)
    ax_path.scatter([x[-1]], [z[-1]], s=90, c="#d62728", marker="X", label="end", zorder=5)
    ax_path.set_aspect("equal", adjustable="datalim")
    ax_path.grid(True, linewidth=0.4, alpha=0.35)
    ax_path.set_xlabel("x")
    ax_path.set_ylabel("z")
    ax_path.legend(loc="best")
    ax_path.set_title("Trajectory on X-Z Plane")

    t = np.arange(len(score))
    ax_score.plot(t, score, label="score", color="#1f77b4", linewidth=1.5)
    ax_score.plot(t, coverage, label="coverage", color="#ff7f0e", linewidth=1.5)
    ax_score.set_ylim(-0.03, 1.03)
    ax_score.grid(True, linewidth=0.4, alpha=0.35)
    ax_score.legend(loc="lower right")
    ax_score.set_title("Score / Coverage Over Steps")

    action_names = {0: "left", 1: "right", 2: "forward"}
    colors = {0: "#9467bd", 1: "#8c564b", 2: "#17becf"}
    counts = [(action_names.get(a, str(a)), int((actions == a).sum()), colors.get(a, "#7f7f7f")) for a in sorted(set(actions.tolist()))]
    ax_actions.bar([c[0] for c in counts], [c[1] for c in counts], color=[c[2] for c in counts])
    ax_actions.set_ylabel("count")
    ax_actions.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax_actions.set_title("Action Counts")

    fig.suptitle(
        f"{scene} | poi={poi_id} start_yaw={start_yaw:.0f} | "
        f"steps={len(actions)} score={final_score:.3f} cov={final_cov:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def plot_episode_on_topdown(path: Path, out_path: Path, *, topdown_cache: dict, args) -> None:
    data = np.load(path, allow_pickle=True)
    pose = np.asarray(data["pose"], dtype=np.float32)
    score = np.asarray(data["score"], dtype=np.float32)
    coverage = np.asarray(data["coverage"], dtype=np.float32)
    scene = _scene_from_path(path)
    if scene not in topdown_cache:
        topdown_cache[scene] = _load_topdown_for_scene(
            scene,
            dataset_root=args.dataset_root,
            conf_path=args.conf_path,
            meters_per_pixel=args.meters_per_pixel,
            gpu_id=args.gpu_id,
        )
    base, shape, bounds = topdown_cache[scene]
    img = Image.fromarray(base.copy(), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    pts = []
    for x, _y, z, _yaw in pose:
        px = _world_to_topdown_px(float(x), float(z), shape, bounds)
        if px is not None:
            pts.append(px)

    if len(pts) >= 2:
        draw.line(pts, fill=(80, 230, 80, 255), width=max(2, int(args.line_width)))
        # Draw a few direction arrows along the path.
        idx = _thin_indices(len(pts), int(args.num_arrows))
        for i in idx:
            if i <= 0 or i >= len(pts):
                continue
            x0, y0 = pts[int(i) - 1]
            x1, y1 = pts[int(i)]
            dx, dy = x1 - x0, y1 - y0
            norm = math.hypot(dx, dy)
            if norm < 2:
                continue
            ux, uy = dx / norm, dy / norm
            size = 5
            left = (x1 - ux * size - uy * size * 0.55, y1 - uy * size + ux * size * 0.55)
            right = (x1 - ux * size + uy * size * 0.55, y1 - uy * size - ux * size * 0.55)
            draw.polygon([(x1, y1), left, right], fill=(80, 230, 80, 230))

    if pts:
        s = pts[0]
        e = pts[-1]
        rs = max(5, int(args.marker_radius))
        re = max(6, int(args.marker_radius) + 1)
        # Match HabitatEnv's convention: start is blue, end is red.
        draw.ellipse((s[0] - rs, s[1] - rs, s[0] + rs, s[1] + rs), fill=(80, 140, 255, 255), outline=(255, 255, 255, 255), width=1)
        draw.ellipse((e[0] - re, e[1] - re, e[0] + re, e[1] + re), fill=(255, 60, 60, 255), outline=(255, 255, 255, 255), width=1)

    if args.annotate:
        final_score = _load_scalar(data, "final_score")
        final_cov = _load_scalar(data, "final_coverage")
        poi_id = int(_load_scalar(data, "poi_id", -1))
        yaw = _load_scalar(data, "start_yaw_deg")
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        text = (
            f"{scene}  poi={poi_id} yaw={yaw:.0f}  steps={len(pose)}  "
            f"score={final_score:.3f} cov={final_cov:.3f}"
        )
        bbox = draw.textbbox((0, 0), text, font=font)
        pad = 5
        draw.rectangle((4, 4, bbox[2] + 2 * pad + 4, bbox[3] + 2 * pad + 4), fill=(0, 0, 0, 150))
        draw.text((4 + pad, 4 + pad), text, fill=(255, 255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)


def make_contact_sheet(image_paths: List[Path], out_path: Path, thumb_w: int = 480) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        ratio = thumb_w / float(img.width)
        thumb_h = int(img.height * ratio)
        img = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h + 34), "white")
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb_h + 8), path.name[:72], fill=(20, 20, 20))
        thumbs.append(canvas)

    cols = min(2, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    w = cols * thumb_w
    h = rows * thumbs[0].height
    sheet = Image.new("RGB", (w, h), "white")
    for i, img in enumerate(thumbs):
        sheet.paste(img, ((i % cols) * thumb_w, (i // cols) * img.height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _select_files(files: List[Path], count: int, mode: str) -> List[Path]:
    def final_score(path: Path) -> float:
        match = re.search(r"_score_([0-9.]+)_cov_", path.name)
        return float(match.group(1)) if match else -1.0

    if mode == "highest":
        return sorted(files, key=final_score, reverse=True)[:count]
    if mode == "lowest":
        return sorted(files, key=final_score)[:count]
    if mode == "spread":
        ordered = sorted(files, key=final_score)
        idx = _thin_indices(len(ordered), count)
        return [ordered[int(i)] for i in idx]
    return files[:count]


def _filter_split(
    files: List[Path],
    *,
    split: str,
    val_split: float,
    split_seed: int,
    split_mode: str,
) -> List[Path]:
    split = str(split).lower()
    if split in {"all", "none"}:
        return files
    if split == "test":
        split = "val"
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split={split!r}; use all, train, val, or test")

    if not files or float(val_split) <= 0.0:
        return files if split == "train" else []

    mode = str(split_mode).lower()
    if mode not in {"file", "scene"}:
        raise ValueError(f"Unsupported split_mode={mode!r}; use file or scene for trajectory visualization")

    grouped: Dict[str, List[Path]] = {}
    for path in files:
        key = str(path) if mode == "file" else path.parent.name
        grouped.setdefault(key, []).append(path)

    keys = sorted(grouped)
    if len(keys) <= 1:
        return files

    val_groups = max(1, int(len(keys) * float(val_split)))
    val_groups = min(val_groups, len(keys) - 1)
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(len(keys)).tolist()
    val_keys = {keys[i] for i in perm[:val_groups]}

    selected: List[Path] = []
    for key in keys:
        in_val = key in val_keys
        if (split == "val" and in_val) or (split == "train" and not in_val):
            selected.extend(grouped[key])
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize saved HM3D expert NPZ trajectories.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--mode", choices=["highest", "lowest", "spread", "first"], default="spread")
    parser.add_argument("--scene", default=None, help="Optional scene id substring, e.g. 00016 or qk9ee.")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["file", "scene"], default="file")
    parser.add_argument("--topdown", action="store_true", help="Draw trajectories on Habitat topdown navmesh map.")
    parser.add_argument("--dataset-root", default="/root/hm3d/scene_datasets/hm3d")
    parser.add_argument("--conf-path", default="/root/RL/config")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--meters-per-pixel", type=float, default=None)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--marker-radius", type=int, default=5)
    parser.add_argument("--num-arrows", type=int, default=28)
    parser.add_argument("--annotate", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    files = sorted(data_dir.rglob("*.npz"))
    if args.scene:
        files = [p for p in files if args.scene in str(p)]
    files = _filter_split(
        files,
        split=args.split,
        val_split=args.val_split,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
    )
    selected = _select_files(files, int(args.count), args.mode)
    if not selected:
        raise SystemExit(f"No NPZ files found under {data_dir}")

    print(
        f"selected_split={args.split} split_mode={args.split_mode} "
        f"val_split={args.val_split} split_seed={args.split_seed} "
        f"available_files={len(files)} drawing={len(selected)}"
    )

    written: List[Path] = []
    topdown_cache = {}
    for i, path in enumerate(selected):
        scene = _scene_from_path(path)
        out_name = f"{i:02d}_{scene}_{path.stem}.png"
        out_path = out_dir / out_name
        if args.topdown:
            plot_episode_on_topdown(path, out_path, topdown_cache=topdown_cache, args=args)
        else:
            plot_episode(path, out_path)
        written.append(out_path)
        print(out_path)

    sheet = out_dir / "contact_sheet.png"
    make_contact_sheet(written, sheet)
    print(f"contact_sheet={sheet}")


if __name__ == "__main__":
    main()
