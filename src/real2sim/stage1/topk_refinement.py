"""Refine every ranked initialization seed without pruning branches."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .initial_pose_batch import ALL_FRAMES, ranked_sheet
from .sim3_registration import ROOT, load_observations, ranking, refine


REQUIRED_SEEDS = (221, 42)
MODEL_STAGES = ((5000, 8), (10000, 7), (20000, 6))


def minimum_k(initializations: Path, required: tuple[int, ...] = REQUIRED_SEEDS) -> tuple[int, dict]:
    ranks = {}
    for path in sorted(initializations.glob("*/initialization.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["robot"] != "piper" or report["status"] != "initialization_only":
            raise ValueError(f"unexpected initialization: {path}")
        ordered = sorted(report["candidates"], key=lambda item: (item["loss"], item["index"]))
        if len({item["index"] for item in ordered}) != len(ordered):
            raise ValueError(f"duplicate rotation indices: {path}")
        positions = {item["index"]: position for position, item in enumerate(ordered, 1)}
        if any(index not in positions for index in required):
            raise ValueError(f"missing required rotation in {path}")
        ranks[path.parent.name] = {str(index): positions[index] for index in required}
    if not ranks:
        raise ValueError(f"no initialization reports in {initializations}")
    return max(rank for scene in ranks.values() for rank in scene.values()), ranks


def local_search(pose: tuple, model_center: np.ndarray, bounds: tuple[float, float],
                 radius: float, score, passes: int = 2) -> tuple[tuple, float, dict, int]:
    best_loss, best_metrics = score(pose)
    accepted = 0
    for level in range(passes):
        rotation_step = .10 / (2 ** level)
        translation_step = .025 * radius / (2 ** level)
        scale_step = .05 / (2 ** level)
        for axis in range(7):
            for direction in (1, -1):
                rotation, translation, scale = pose
                if axis < 3:
                    offset = np.zeros(3)
                    offset[axis] = direction * translation_step
                    proposal = (rotation, translation + offset, scale)
                elif axis < 6:
                    offset = np.zeros(3)
                    offset[axis - 3] = direction * rotation_step
                    next_rotation = cv2.Rodrigues(offset)[0] @ rotation
                    next_translation = translation + scale * (rotation - next_rotation) @ model_center
                    proposal = (next_rotation, next_translation, scale)
                else:
                    next_scale = scale * np.exp(direction * scale_step)
                    if not bounds[0] <= next_scale <= bounds[1]:
                        continue
                    next_translation = translation + (scale - next_scale) * rotation @ model_center
                    proposal = (rotation, next_translation, float(next_scale))
                loss, metrics = score(proposal)
                if loss < best_loss - 1e-6:
                    pose, best_loss, best_metrics = proposal, loss, metrics
                    accepted += 1
    return pose, best_loss, best_metrics, accepted


def optimize_seed(seed: dict, models: dict, observed: np.ndarray, cameras: dict,
                  depths: dict, masks: dict, model_center: np.ndarray,
                  bounds: tuple[float, float], radius: float, passes: int = 2) -> dict:
    started = time.monotonic()
    pose = (np.asarray(seed["rotation"], dtype=float),
            np.asarray(seed["translation_vggt_world"], dtype=float),
            float(seed["scale_vggt_units_per_meter"]))

    def score(candidate: tuple) -> tuple[float, dict]:
        return ranking(models[20000], observed, candidate, cameras, depths, masks, ALL_FRAMES)

    best_loss, best_metrics = score(pose)
    initial_loss = best_loss
    stage_history = []
    for count, iterations in MODEL_STAGES:
        proposed = refine(models[count], observed, pose, bounds, iterations)
        loss, metrics = score(proposed)
        improved = loss < best_loss - 1e-6
        if improved:
            pose, best_loss, best_metrics = proposed, loss, metrics
        stage_history.append({"model_points": count, "icp_iterations": iterations,
                              "proposed_loss": loss, "accepted": improved,
                              "best_loss": best_loss})
    pose, best_loss, best_metrics, local_updates = local_search(
        pose, model_center, bounds, radius, score, passes)
    rotation, translation, scale = pose
    similarity = np.eye(4)
    similarity[:3, :3] = scale * rotation
    similarity[:3, 3] = translation
    return {"index": seed["index"], "initial_loss_saved": seed["loss"],
            "initial_loss_recomputed": initial_loss, "loss": best_loss,
            "rotation": rotation.tolist(), "translation_vggt_world": translation.tolist(),
            "scale_vggt_units_per_meter": scale, "similarity_world_from_base": similarity.tolist(),
            "metrics": best_metrics, "stages": stage_history,
            "local_search_updates": local_updates, "elapsed_seconds": time.monotonic() - started}


def write_gallery(directory: Path, branches: list[dict], scene: str, k: int) -> None:
    rows = []
    for final_rank, branch in enumerate(branches, 1):
        number = branch["index"]
        image = f"branches/R{number:03d}.png"
        rows.append(f'<article id="R{number:03d}"><h2>#{final_rank} R{number:03d} '
                    f'(initial #{branch["initial_rank"]}) | loss '
                    f'{branch["initial_loss_recomputed"]:.4f} → {branch["loss"]:.4f}</h2>'
                    f'<a href="{image}"><img loading="lazy" src="{image}" '
                    f'alt="R{number:03d} 32 frame overlay" style="max-width:100%"></a></article>')
    (directory / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        f'<title>{html.escape(scene)} K={k} refined</title>'
        f'<h1>Piper {k} 分支 Sim(3) 局部优化：32 帧评分，无米制验收</h1>'
        '<p><a href="top4_refined.png">最终前四</a> | '
        '<a href="compare_221_42.png">R221 / R42 对比</a> | '
        '<a href="#R221">R221</a> | <a href="#R042">R42</a></p>'
        + "\n".join(rows) + "</html>\n", encoding="utf-8")


def run_scene(scene: str, initializations: Path, processed: Path, output: Path,
              required: tuple[int, ...] = REQUIRED_SEEDS, passes: int = 2,
              progress=print) -> dict:
    k, scene_ranks = minimum_k(initializations, required)
    source = processed / "piper" / "stage1" / scene
    initialization_file = initializations / scene / "initialization.json"
    source_bytes = initialization_file.read_bytes()
    initial = json.loads(source_bytes)
    if initial["robot"] != "piper" or initial["pointcloud_frames"] != list(ALL_FRAMES) \
            or initial["scoring_frames"] != list(ALL_FRAMES) \
            or initial["rotation_count"] != 256:
        raise ValueError("expected a Piper 256-rotation, 32-frame initialization report")
    seeds = sorted(initial["candidates"], key=lambda item: (item["loss"], item["index"]))[:k]
    if not set(required).issubset({item["index"] for item in seeds}):
        raise ValueError("the top K does not include all required rotations")
    manifest, cameras, depths, masks, observed, counts = load_observations(source, held_out=())
    if manifest["robot"] != "piper" or [entry["frame"] for entry in counts] != list(ALL_FRAMES):
        raise ValueError("expected all 32 Piper observation frames")
    model_root = ROOT / "asset/robot_surfaces/piper"
    models = {}
    for size in (5000, 10000, 40000):
        with np.load(model_root / f"surface_{size}.npz") as data:
            models[size] = data["points"].astype(float)
    full_model = models.pop(40000)
    models[20000] = full_model[np.random.default_rng(2027).permutation(len(full_model))[:20000]]
    model_center = full_model.mean(axis=0)
    radius = float(np.linalg.norm(np.quantile(observed, .95, axis=0) -
                                  np.quantile(observed, .05, axis=0)))
    if radius <= 0:
        raise ValueError("degenerate observation point cloud")
    start_scale = float(initial["scale_vggt_units_per_meter"])
    bounds = (.5 * start_scale, 1.6 * start_scale)
    output.mkdir(parents=True, exist_ok=True)
    (output / "branches").mkdir(exist_ok=True)
    branches = []
    started = time.monotonic()
    for rank, seed in enumerate(seeds, 1):
        progress(f"[{rank}/{k}] R{seed['index']:03d} initial={seed['loss']:.4f}")
        branch = optimize_seed(seed, models, observed, cameras, depths, masks,
                               model_center, bounds, radius, passes)
        branch["initial_rank"] = rank
        (output / "branches" / f"R{seed['index']:03d}.json").write_text(
            json.dumps(branch, indent=2) + "\n", encoding="utf-8")
        branches.append(branch)
        progress(f"    final={branch['loss']:.4f} scale={branch['scale_vggt_units_per_meter']:.4f} "
                 f"time={branch['elapsed_seconds']:.1f}s")
    branches.sort(key=lambda item: (item["loss"], item["index"]))
    for final_rank, branch in enumerate(branches, 1):
        branch["final_rank"] = final_rank
        ranked_sheet(source, [branch], models[20000], cameras, masks,
                     output / "branches" / f"R{branch['index']:03d}.png", ranks=[final_rank])
    ranked_sheet(source, branches[:4], models[20000], cameras, masks,
                 output / "top4_refined.png")
    highlighted = [next(branch for branch in branches if branch["index"] == index)
                   for index in required]
    ranked_sheet(source, highlighted, models[20000], cameras, masks,
                 output / "compare_221_42.png",
                 ranks=[branch["final_rank"] for branch in highlighted])
    write_gallery(output, branches, scene, k)
    report = {"status": "candidate_unverified", "accepted": False,
              "reason": "32 frames used for selection and optimization; no held-out metric or metric depth calibration",
              "scene": scene, "source": str(source), "initialization_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "k": k, "required_rotations": list(required), "required_ranks_by_scene": scene_ranks,
              "initialization_model_points": 20000, "scoring_model_points": 20000,
              "scoring_frames": list(ALL_FRAMES), "pointcloud_frames": list(ALL_FRAMES),
              "observation_counts": counts, "scale_bounds_vggt_per_meter": list(bounds),
              "model_stages": [{"points": size, "iterations": iterations} for size, iterations in MODEL_STAGES],
              "local_search_passes": passes, "final_order": [branch["index"] for branch in branches],
              "branches": branches, "elapsed_seconds": time.monotonic() - started}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine the Piper top K Sim(3) seeds using all 32 frames")
    parser.add_argument("--scene", default="open_drawer")
    parser.add_argument("--initializations", type=Path, default=ROOT / "runs/initial_poses_256/piper")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-passes", type=int, default=2)
    args = parser.parse_args()
    if args.local_passes < 0:
        parser.error("--local-passes must not be negative")
    output = args.output or ROOT / "runs/topk_refinement/piper" / args.scene
    report = run_scene(args.scene, args.initializations, args.processed, output,
                       passes=args.local_passes)
    print(json.dumps({key: report[key] for key in ("status", "scene", "k", "final_order", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
