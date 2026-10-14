"""Visualize center-and-scale aligned, unrefined rotation seeds for every cached scene."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .sim3_registration import ROOT, load_observations, project, ranking, search_rotations


GRID_COLUMNS = 8
THUMBNAIL = (200, 112)
LABEL_HEIGHT = 20
ALL_FRAMES = tuple(range(32))
PIPER_INITIAL_ROTATIONS = 128


def center_and_scale(model: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if len(model) < 3 or len(observed) < 3:
        raise ValueError("at least three model and observed points are required")
    model_center = model.mean(axis=0)
    observed_center = observed.mean(axis=0)
    model_extent = np.linalg.norm(np.quantile(model, .95, axis=0) -
                                  np.quantile(model, .05, axis=0))
    observed_extent = np.linalg.norm(np.quantile(observed, .95, axis=0) -
                                     np.quantile(observed, .05, axis=0))
    if model_extent <= 0 or observed_extent <= 0:
        raise ValueError("degenerate model or observed point cloud")
    return model_center, observed_center, float(observed_extent / model_extent)


def visual_tile(image: np.ndarray, mask: np.ndarray, footprint: np.ndarray,
                index: int, overlap: float) -> np.ndarray:
    overlay = image.copy()
    overlay[mask] = (overlay[mask] * .5 + np.array([0, 180, 0]) * .5).astype(np.uint8)
    overlay[footprint] = (overlay[footprint] * .35 + np.array([0, 0, 255]) * .65).astype(np.uint8)
    thumbnail = cv2.resize(overlay, THUMBNAIL, interpolation=cv2.INTER_AREA)
    tile = np.full((THUMBNAIL[1] + LABEL_HEIGHT, THUMBNAIL[0], 3), 32, np.uint8)
    tile[LABEL_HEIGHT:] = thumbnail
    cv2.putText(tile, f"frame {index:02d}  IoU {overlap:.3f}", (6, 15),
                cv2.FONT_HERSHEY_SIMPLEX, .41, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def ranked_sheet(directory: Path, candidates: list[dict], model: np.ndarray,
                 cameras: dict, masks: dict, target: Path,
                 ranks: list[int] | None = None,
                 labels: list[str] | None = None) -> None:
    tiles = {}
    for index in ALL_FRAMES:
        image = cv2.imread(str(directory / "frames" / f"{index:02d}.jpg"))
        if image is None or image.shape[:2] != masks[index].shape:
            raise ValueError(f"missing or mismatched image for frame {index}")
        tiles[index] = image
    blocks = []
    for rank, candidate in enumerate(candidates, 1):
        title = np.full((37, GRID_COLUMNS * THUMBNAIL[0], 3), 24, np.uint8)
        displayed_rank = ranks[rank - 1] if ranks is not None else rank
        prefix = labels[rank - 1] if labels is not None else f"#{displayed_rank}"
        cv2.putText(title, f"{prefix}   rotation R{candidate['index']:02d}   "
                    f"loss {candidate['loss']:.4f}   mean IoU "
                    f"{candidate['metrics']['scored_views']['mean_iou']:.3f}   "
                    f"scale {candidate['scale_vggt_units_per_meter']:.4f} VGGT/m",
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .68, (255, 255, 255), 2, cv2.LINE_AA)
        pose = (np.asarray(candidate["rotation"]),
                np.asarray(candidate["translation_vggt_world"]), candidate["scale_vggt_units_per_meter"])
        rows = []
        per_frame = {frame["frame"]: frame for frame in candidate["metrics"]["per_frame"]}
        for first in range(0, 32, GRID_COLUMNS):
            previews = []
            for index in range(first, first + GRID_COLUMNS):
                footprint, _ = project(model, pose, cameras, index, masks[index].shape)
                previews.append(visual_tile(tiles[index], masks[index], footprint, index,
                                            per_frame[index]["iou"]))
            rows.append(np.concatenate(previews, axis=1))
        blocks.append(np.concatenate([title, *rows], axis=0))
    if not cv2.imwrite(str(target), np.concatenate(blocks, axis=0)):
        raise OSError(f"failed to write {target}")


def visualize_scene(directory: Path, output: Path, rotations: int = 48) -> dict:
    started = time.monotonic()
    manifest, cameras, depths, masks, observed, counts = load_observations(directory, held_out=())
    robot = manifest["robot"]
    model_directory = ROOT / "asset/robot_surfaces" / robot
    with np.load(model_directory / "surface_40000.npz") as data:
        full_model = data["points"].astype(float)
    model_center, observed_center, scale = center_and_scale(full_model, observed)
    indices = np.random.default_rng(2027).permutation(len(full_model))[:20000]
    visual_model = full_model[indices]
    orientations = search_rotations(rotations, include_identity=False)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "observation.npz", points=observed)
    trimesh.points.PointCloud(observed).export(output / "observation.ply")

    candidates = []
    for identifier, rotation in enumerate(orientations):
        translation = observed_center - scale * rotation @ model_center
        pose = (rotation, translation, scale)
        loss, quality = ranking(visual_model, observed, pose, cameras, depths, masks, ALL_FRAMES)
        candidates.append({"index": identifier, "rotation": rotation.tolist(),
                           "translation_vggt_world": translation.tolist(),
                           "scale_vggt_units_per_meter": scale,
                           "loss": loss, "metrics": quality})
    top_four = sorted(candidates, key=lambda item: (item["loss"], item["index"]))[:4]
    ranked_sheet(directory, top_four, visual_model, cameras, masks, output / "top4.png")
    for index in (0, 10, 21, 31):
        old_sheet = output / f"frame_{index:02d}_all_rotations.png"
        if old_sheet.is_file():
            old_sheet.unlink()

    report = {"status": "initialization_only", "accepted": False,
              "source": str(directory), "robot": robot,
              "video_sha256": manifest["video_sha256"],
              "sam3_masks_sha256": manifest.get("sam3_masks_sha256"),
              "robot_surface_sha256": hashlib.sha256((model_directory / "surface_40000.npz").read_bytes()).hexdigest(),
              "depth_unit": "unverified VGGT units", "model_unit": "meters",
              "model_center_base_m": model_center.tolist(),
              "observed_center_vggt_world": observed_center.tolist(),
              "model_p05_p95_extent_m": float(np.linalg.norm(np.quantile(full_model, .95, axis=0) -
                                                            np.quantile(full_model, .05, axis=0))),
              "observed_p05_p95_extent_vggt": float(np.linalg.norm(np.quantile(observed, .95, axis=0) -
                                                                    np.quantile(observed, .05, axis=0))),
              "scale_vggt_units_per_meter": scale,
              "scale_method": "ratio of P05-P95 axis-aligned bounding-box diagonals",
              "translation_method": "observation_center - scale * rotation * model_center",
              "rotation_method": "deterministic quasi-uniform quaternion samples over SO(3)",
              "rotation_count": len(orientations), "model_center_source_points": len(full_model),
              "scoring_and_visualization_model_points": len(visual_model),
              "model_subsample_seed": 2027,
              "observed_points": len(observed), "point_counts_by_training_frame": counts,
              "pointcloud_frames": list(ALL_FRAMES), "scoring_frames": list(ALL_FRAMES),
              "visualized_frames": list(ALL_FRAMES),
              "note": "Only the four highest-scoring seeds are selected for display. No ICP or pose/scale refinement is run.",
              "top_four_indices": [item["index"] for item in top_four],
              "top_four": top_four, "candidates": candidates,
              "elapsed_seconds": time.monotonic() - started}
    (output / "initialization.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def write_index(results: list[dict], output: Path) -> None:
    rows = []
    for result in results:
        if result["status"] == "failed":
            rows.append(f"<li>{html.escape(result['robot'])}/{html.escape(result['scene'])}: failed</li>")
            continue
        target = f"{result['robot']}/{result['scene']}"
        label = html.escape(target)
        rows.append(f'<li><h2>{label} | scale {result["scale_vggt_units_per_meter"]:.4f} VGGT/m</h2>'
                    f'<a href="{label}/top4.png"><img src="{label}/top4.png" '
                    f'alt="{label} 前四名（32 帧）" loading="lazy" style="max-width:100%"></a></li>')
    (output / "index.html").write_text(
        "<!doctype html><html lang=zh><meta charset=utf-8><title>初始位姿总览</title>"
        "<h1>点云中心＋单一尺度＋旋转候选（无 ICP）</h1><ul>" + "\n".join(rows) + "</ul></html>\n",
        encoding="utf-8")


def batch(processed: Path = ROOT / "data/processed", output: Path = ROOT / "runs/initial_poses",
          rotations: int = 48, progress=print, robot: str | None = None,
          active_only: bool = False) -> list[dict]:
    if rotations < 1:
        raise ValueError("rotations must be positive")
    scenes = [(name, scene) for name in ((robot,) if robot else ("franka", "piper"))
              for scene in sorted((processed / name / "stage1").iterdir()) if scene.is_dir()
              and not scene.name.startswith(".")]
    if active_only:
        from .pipeline import sha256
        scenes = [(name, scene) for name, scene in scenes
                  if (processed / name / "stage1" / scene.name / "manifest.json").is_file()
                  and (ROOT / "data/raw" / ("frank" if name == "franka" else name) /
                       f"{scene.name}.mp4").is_file()
                  and sha256(ROOT / "data/raw" / ("frank" if name == "franka" else name) /
                             f"{scene.name}.mp4") == json.loads((scene / "manifest.json").read_text())["video_sha256"]]
    if not scenes:
        raise ValueError(f"no processed scenes in {processed}")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for number, (robot, scene) in enumerate(scenes, 1):
        target = output / robot / scene.name
        progress(f"[{number}/{len(scenes)}] {robot}/{scene.name}")
        try:
            report = visualize_scene(scene, target, rotations)
            result = {"robot": robot, "scene": scene.name,
                      "status": report["status"], "output": str(target),
                      "scale_vggt_units_per_meter": report["scale_vggt_units_per_meter"],
                      "seconds": report["elapsed_seconds"]}
        except Exception as exc:
            target.mkdir(parents=True, exist_ok=True)
            (target / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            result = {"robot": robot, "scene": scene.name, "status": "failed",
                      "output": str(target), "error": str(exc)}
        results.append(result)
        progress(f"    {result['status']}" +
                 (f" scale={result['scale_vggt_units_per_meter']:.4f} time={result['seconds']:.1f}s"
                  if result["status"] != "failed" else f" error={result['error']}"))
        (output / "index.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    write_index(results, output)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize unrefined initial poses for all cached scenes")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rotations", type=int)
    parser.add_argument("--robot", choices=("franka", "piper"))
    parser.add_argument("--active-only", action="store_true", help="Skip obsolete scenes with no matching raw video")
    arguments = parser.parse_args()
    rotations = arguments.rotations if arguments.rotations is not None else (
        PIPER_INITIAL_ROTATIONS if arguments.robot == "piper" else 48)
    output = arguments.output or (ROOT / "runs/piper_initial_128_review" if arguments.robot == "piper"
                                  and rotations == PIPER_INITIAL_ROTATIONS else ROOT / "runs/initial_poses")
    result = batch(arguments.processed, output, rotations,
                   robot=arguments.robot, active_only=arguments.active_only)
    if any(item["status"] == "failed" for item in result):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
