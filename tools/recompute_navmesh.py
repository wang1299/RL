#!/usr/bin/env python3
"""Recompute Habitat navmesh files for one scene or a directory of scenes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import habitat_sim


def _default_navmesh_path(scene_path: Path, output_dir: Path | None = None) -> Path:
    out_name = scene_path.with_suffix(".navmesh").name
    if output_dir is not None:
        return output_dir / out_name
    return scene_path.with_suffix(".navmesh")


def _is_scene_glb(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and name.endswith(".glb") and ".semantic." not in name


def iter_scene_glbs(root: Path, pattern: str) -> Iterable[Path]:
    if root.is_file():
        if _is_scene_glb(root):
            yield root
        return
    for path in sorted(root.rglob(pattern)):
        if _is_scene_glb(path):
            yield path


def recompute_and_save_navmesh(
    scene_path: str | Path,
    out_path: str | Path | None = None,
    *,
    agent_radius: float = 0.17,
    agent_height: float = 1.5,
    agent_max_climb: float = 0.1,
    agent_max_slope: float | None = None,
    cell_height: float = 0.05,
    cell_size: float = 0.03,
    include_static_objects: bool = True,
    overwrite: bool = False,
) -> bool:
    scene_path = Path(scene_path)
    out_path = Path(out_path) if out_path is not None else _default_navmesh_path(scene_path)

    if out_path.exists() and not overwrite:
        print(f"[SKIP] exists: {out_path}")
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_path)
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    with habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg])) as sim:
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = float(agent_radius)
        navmesh_settings.agent_height = float(agent_height)
        navmesh_settings.agent_max_climb = float(agent_max_climb)
        if agent_max_slope is not None:
            navmesh_settings.agent_max_slope = float(agent_max_slope)
        navmesh_settings.cell_height = float(cell_height)
        navmesh_settings.cell_size = float(cell_size)

        try:
            success = sim.recompute_navmesh(
                sim.pathfinder,
                navmesh_settings,
                include_static_objects=bool(include_static_objects),
            )
        except TypeError:
            success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        if not success:
            print(f"[FAIL] {scene_path}")
            return False

        sim.pathfinder.save_nav_mesh(str(out_path))
        print(f"[OK] {scene_path} -> {out_path}")
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        help="Single .glb scene file or a directory containing .glb scenes.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Root directory to scan recursively for .glb scenes.",
    )
    parser.add_argument("--pattern", default="*.glb")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for generated .navmesh files. Defaults next to each .glb.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--agent-radius", type=float, default=0.17)
    parser.add_argument("--agent-height", type=float, default=1.5)
    parser.add_argument("--agent-max-climb", type=float, default=0.1)
    parser.add_argument("--agent-max-slope", type=float, default=None)
    parser.add_argument("--cell-height", type=float, default=0.05)
    parser.add_argument("--cell-size", type=float, default=0.03)
    parser.add_argument(
        "--no-static-objects",
        action="store_true",
        help="Do not include static objects in navmesh recomputation.",
    )
    args = parser.parse_args()
    if args.scene is None and args.root is None:
        parser.error("Provide --scene or --root")
    return args


def main() -> int:
    args = parse_args()
    scan_root = args.scene or args.root
    scenes = list(iter_scene_glbs(scan_root, args.pattern))
    if args.limit > 0:
        scenes = scenes[: args.limit]

    if not scenes:
        print(f"[ERROR] No .glb scenes found under {scan_root}")
        return 1

    print(f"[INFO] Found {len(scenes)} scene(s)")
    failures = 0
    for scene_path in scenes:
        out_path = _default_navmesh_path(scene_path, args.output_dir)
        if args.dry_run:
            print(f"[DRY] {scene_path} -> {out_path}")
            continue
        ok = recompute_and_save_navmesh(
            scene_path,
            out_path,
            agent_radius=args.agent_radius,
            agent_height=args.agent_height,
            agent_max_climb=args.agent_max_climb,
            agent_max_slope=args.agent_max_slope,
            cell_height=args.cell_height,
            cell_size=args.cell_size,
            include_static_objects=not args.no_static_objects,
            overwrite=args.overwrite,
        )
        failures += 0 if ok else 1

    if failures:
        print(f"[DONE] failures={failures}/{len(scenes)}")
        return 2
    print("[DONE] all navmeshes ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
