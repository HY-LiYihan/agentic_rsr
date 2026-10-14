"""Refine the best of 48 cached Franka rotation seeds for every task."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from .initial_pose_batch import ALL_FRAMES, ranked_sheet
from .metric_calibration import export_metric_scene, metric_method
from .rotation_search import MIN_LOSS_IMPROVEMENT, ROTATION_STEPS_DEGREES, refine_rotation
from .sim3_registration import ROOT, load_observations, ranking
from .topk_refinement import MODEL_STAGES, optimize_seed


METHOD = "franka_48_top1_sim3_rotation_v1"


def select_seed(initialization: dict, robot: str = "franka", seed_rank: int = 1) -> dict:
    if initialization.get("robot") != robot or initialization.get("status") != "initialization_only":
        raise ValueError(f"expected a completed {robot} initialization")
    rotation_count = 128 if robot == "piper" else 48
    if initialization.get("rotation_count") != rotation_count or len(initialization.get("candidates", [])) != rotation_count:
        raise ValueError(f"expected exactly {rotation_count} rotation candidates")
    for field in ("pointcloud_frames", "scoring_frames"):
        if initialization.get(field) != list(ALL_FRAMES):
            raise ValueError(f"expected all 32 {field}")
    candidates = initialization["candidates"]
    if len({candidate["index"] for candidate in candidates}) != rotation_count:
        raise ValueError("duplicate rotation candidates")
    if seed_rank < 1 or seed_rank > rotation_count:
        raise ValueError(f"seed rank must be between 1 and {rotation_count}")
    return sorted(candidates, key=lambda candidate: (candidate["loss"], candidate["index"]))[seed_rank - 1]


def refine_scene(scene: str, processed: Path, initializations: Path,
                 output: Path, models: dict, full_model: np.ndarray,
                 passes: int = 2, *, robot: str = "franka", seed_rank: int = 1) -> dict:
    started = time.monotonic()
    initialization_file = initializations / scene / "initialization.json"
    contents = initialization_file.read_bytes()
    initialization = json.loads(contents)
    seed = select_seed(initialization, robot, seed_rank)
    source = processed / robot / "stage1" / scene
    manifest, cameras, depths, masks, observed, counts = load_observations(source, held_out=())
    if manifest.get("robot") != robot or [entry["frame"] for entry in counts] != list(ALL_FRAMES):
        raise ValueError(f"expected 32 valid {robot} observation frames")
    if robot == "piper":
        from .pipeline import sha256
        surface = ROOT / "asset/robot_surfaces" / robot / "surface_40000.npz"
        if (initialization.get("video_sha256") != manifest.get("video_sha256") or
                initialization.get("robot_surface_sha256") != sha256(surface) or
                not manifest.get("sam3_masks_sha256") or
                initialization.get("sam3_masks_sha256") != manifest["sam3_masks_sha256"]):
            raise ValueError("Piper initialization used stale video, masks or robot surface")
    center = full_model.mean(axis=0)
    radius = float(np.linalg.norm(np.quantile(observed, .95, axis=0) -
                                  np.quantile(observed, .05, axis=0)))
    if radius <= 0:
        raise ValueError("degenerate observation point cloud")
    initial_scale = float(seed["scale_vggt_units_per_meter"])
    bounds = (.5 * initial_scale, 1.6 * initial_scale)
    baseline = optimize_seed(seed, models, observed, cameras, depths, masks, center,
                             bounds, radius, passes)
    baseline_pose = (np.asarray(baseline["rotation"]),
                     np.asarray(baseline["translation_vggt_world"]),
                     float(baseline["scale_vggt_units_per_meter"]))

    def score(candidate: tuple) -> tuple[float, dict]:
        return ranking(models[20000], observed, candidate, cameras, depths, masks, ALL_FRAMES)

    final_pose, loss, metrics, search_history = refine_rotation(
        baseline_pose, center, radius, bounds, score,
        initial_score=(baseline["loss"], baseline["metrics"]))
    rotation, translation, scale = final_pose
    similarity = np.eye(4)
    similarity[:3, :3] = scale * rotation
    similarity[:3, 3] = translation
    difference = rotation @ baseline_pose[0].T
    rotation_change = math.degrees(math.acos(float(np.clip((np.trace(difference) - 1) / 2, -1, 1))))
    result = {**baseline, "loss": loss, "metrics": metrics,
              "rotation": rotation.tolist(), "translation_vggt_world": translation.tolist(),
              "scale_vggt_units_per_meter": scale,
              "similarity_world_from_base": similarity.tolist(),
              "baseline": baseline, "baseline_loss": baseline["loss"],
              "rotation_change_from_baseline_degrees": rotation_change,
              "rotation_steps_degrees": list(ROTATION_STEPS_DEGREES),
              "minimum_loss_improvement": MIN_LOSS_IMPROVEMENT,
              "search_history": search_history}
    result.update({"status": "candidate_unverified", "accepted": False,
                   "reason": f"Metric scale anchored by {robot} URDF; independent registration validation pending",
                   "scene": scene, "robot": robot, "source": str(source),
                   "method": metric_method(robot),
                   "initialization_sha256": hashlib.sha256(contents).hexdigest(),
                   "sam3_masks_sha256": manifest.get("sam3_masks_sha256"),
                   "selected_seed_rank": seed_rank, "rotation_count": initialization["rotation_count"],
                   "pointcloud_frames": list(ALL_FRAMES), "scoring_frames": list(ALL_FRAMES),
                   "observation_counts": counts, "scale_bounds_vggt_per_meter": list(bounds),
                   "model_stages": [{"points": size, "iterations": iterations}
                                    for size, iterations in MODEL_STAGES],
                   "local_search_passes": passes,
                   "elapsed_seconds": time.monotonic() - started})
    output.mkdir(parents=True, exist_ok=True)
    ranked_sheet(source, [baseline, result], models[20000], cameras, masks,
                 output / "before_after_32.png", labels=["BEFORE", "AFTER"])
    ranked_sheet(source, [result], models[20000], cameras, masks, output / "overlay_32.png")
    result["metric_calibration"] = export_metric_scene(source, output, result)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def write_gallery(results: list[dict], output: Path, robot: str = "franka") -> None:
    rows = []
    for result in results:
        name = html.escape(result["scene"])
        if result["status"] == "failed":
            rows.append(f"<li>{name}: failed; see error.log</li>")
            continue
        rows.append(f'<li><h2>{name} | R{result["index"]:02d} | '
                    f'loss {result["baseline_loss"]:.4f} → {result["loss"]:.4f} | '
                    f'scale {result["scale_vggt_units_per_meter"]:.4f} VGGT/m</h2>'
                    f'<a href="{name}/before_after_32.png"><img loading="lazy" '
                    f'src="{name}/before_after_32.png" alt="{name} 优化前后 32 帧叠图" '
                    'style="max-width:100%"></a></li>')
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        f'<title>{robot} 各任务最终候选</title>'
        f'<h1>{robot}：48 旋转取第一名，多尺度 ICP 后扩大旋转搜索</h1>'
        f'<p>全部 32 帧参与初值及优化；米制尺度由 {robot} URDF 锚定，但尚无独立配准验收。</p>'
        '<ol>' + "\n".join(rows) + '</ol></html>\n', encoding="utf-8")


def batch(initializations: Path = ROOT / "runs/initial_poses/franka",
          processed: Path = ROOT / "data/processed",
          output: Path = ROOT / "runs/franka_stage1_final_48", passes: int = 2,
          only: str | None = None, *, robot: str = "franka", active_only: bool = False,
          progress=print) -> list[dict]:
    if passes < 0:
        raise ValueError("passes must not be negative")
    scenes = sorted(path.name for path in initializations.iterdir()
                    if path.is_dir() and (path / "initialization.json").is_file())
    if active_only:
        from .pipeline import sha256
        raw = ROOT / "data/raw" / ("frank" if robot == "franka" else robot)
        scenes = [scene for scene in scenes if (raw / f"{scene}.mp4").is_file()
                  and (processed / robot / "stage1" / scene / "manifest.json").is_file()
                  and sha256(raw / f"{scene}.mp4") == json.loads(
                      (processed / robot / "stage1" / scene / "manifest.json").read_text())["video_sha256"]]
    if only is not None:
        if only not in scenes:
            raise ValueError(f"missing initialized {robot} scene: {only}")
        scenes = [only]
    if not scenes:
        raise ValueError(f"no {robot} initialization reports in {initializations}")
    model_root = ROOT / "asset/robot_surfaces" / robot
    models = {}
    for size in (5000, 10000, 40000):
        with np.load(model_root / f"surface_{size}.npz") as data:
            models[size] = data["points"].astype(float)
    full_model = models.pop(40000)
    models[20000] = full_model[np.random.default_rng(2027).permutation(len(full_model))[:20000]]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for number, scene in enumerate(scenes, 1):
        progress(f"[{number}/{len(scenes)}] {robot}/{scene}")
        try:
            result = refine_scene(scene, processed, initializations, output / scene,
                                  models, full_model, passes, robot=robot)
        except Exception as exc:
            target = output / scene
            target.mkdir(parents=True, exist_ok=True)
            (target / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            result = {"scene": scene, "status": "failed", "error": str(exc)}
        results.append(result)
        progress(f"    {result['status']}" +
                 (f" R{result['index']:02d} {result['baseline_loss']:.4f}"
                  f" -> {result['loss']:.4f}" if result["status"] != "failed"
                  else f" error={result['error']}"))
        (output / "index.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        write_gallery(results, output, robot)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine the top-1 Franka seed from 48 rotations per task")
    parser.add_argument("--initializations", type=Path, default=ROOT / "runs/initial_poses/franka")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/franka_stage1_final_48")
    parser.add_argument("--local-passes", type=int, default=2)
    parser.add_argument("--only", help="process a single initialized Franka task")
    args = parser.parse_args()
    if args.local_passes < 0:
        parser.error("--local-passes must not be negative")
    results = batch(args.initializations, args.processed, args.output, args.local_passes,
                    only=args.only)
    if any(result["status"] == "failed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
