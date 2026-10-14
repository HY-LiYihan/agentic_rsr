"""Compare wider rotation refinement against a saved Franka single-seed result."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .franka_final import select_seed
from .initial_pose_batch import ALL_FRAMES, ranked_sheet
from .rotation_search import (MIN_LOSS_IMPROVEMENT, ROTATION_STEPS_DEGREES,
                              adjust_translation_scale, refine_rotation)
from .sim3_registration import ROOT, load_observations, ranking


def run(scene: str, baseline_file: Path, initial_file: Path,
        processed: Path, output: Path) -> dict:
    baseline_bytes = baseline_file.read_bytes()
    baseline = json.loads(baseline_bytes)
    initial = json.loads(initial_file.read_text(encoding="utf-8"))
    seed = select_seed(initial)
    if baseline.get("robot") != "franka" or baseline.get("scene") != scene \
            or baseline.get("index") != seed["index"] or baseline.get("selected_seed_rank") != 1 \
            or baseline.get("status") != "candidate_unverified" \
            or baseline.get("scoring_frames") != list(ALL_FRAMES):
        raise ValueError("baseline must be the same scene's top-1 Franka 48-rotation result")
    source = processed / "franka" / "stage1" / scene
    manifest, cameras, depths, masks, observed, counts = load_observations(source, held_out=())
    if manifest.get("robot") != "franka" or [entry["frame"] for entry in counts] != list(ALL_FRAMES):
        raise ValueError("expected all 32 Franka observation frames")
    with np.load(ROOT / "asset/robot_surfaces/franka/surface_40000.npz") as data:
        full_model = data["points"].astype(float)
    center = full_model.mean(axis=0)
    model = full_model[np.random.default_rng(2027).permutation(len(full_model))[:20000]]
    radius = float(np.linalg.norm(np.quantile(observed, .95, axis=0) -
                                  np.quantile(observed, .05, axis=0)))
    if radius <= 0:
        raise ValueError("degenerate observation point cloud")
    bounds = tuple(baseline["scale_bounds_vggt_per_meter"])
    pose = (np.asarray(baseline["rotation"], dtype=float),
            np.asarray(baseline["translation_vggt_world"], dtype=float),
            float(baseline["scale_vggt_units_per_meter"]))

    def score(candidate: tuple) -> tuple[float, dict]:
        return ranking(model, observed, candidate, cameras, depths, masks, ALL_FRAMES)

    baseline_loss, baseline_metrics = score(pose)
    if abs(baseline_loss - baseline["loss"]) > 1e-3:
        raise ValueError("baseline loss changed; cannot compare against saved result")
    started = time.monotonic()
    final_pose, loss, metrics, history = refine_rotation(
        pose, center, radius, bounds, score,
        initial_score=(baseline_loss, baseline_metrics))
    rotation, translation, scale = final_pose
    relative = rotation @ pose[0].T
    angle = math.degrees(math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    similarity = np.eye(4)
    similarity[:3, :3] = scale * rotation
    similarity[:3, 3] = translation
    result = {"status": "candidate_unverified", "accepted": False, "robot": "franka",
              "scene": scene, "source": str(source), "index": seed["index"],
              "selected_seed_rank": 1, "rotation_count": 48,
              "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
              "initialization_sha256": baseline["initialization_sha256"],
              "baseline_loss": baseline_loss, "loss": loss,
              "rotation_change_from_baseline_degrees": angle,
              "rotation_steps_degrees": ROTATION_STEPS_DEGREES, "search_history": history,
              "minimum_loss_improvement": MIN_LOSS_IMPROVEMENT,
              "rotation": rotation.tolist(), "translation_vggt_world": translation.tolist(),
              "scale_vggt_units_per_meter": scale,
              "similarity_world_from_base": similarity.tolist(), "metrics": metrics,
              "pointcloud_frames": list(ALL_FRAMES), "scoring_frames": list(ALL_FRAMES),
              "observation_counts": counts, "scale_bounds_vggt_per_meter": list(bounds),
              "elapsed_seconds": time.monotonic() - started,
              "reason": "Same 32 frames used for optimization and evaluation; scale not independently metric calibrated"}
    output.mkdir(parents=True, exist_ok=True)
    before = {**baseline, "loss": baseline_loss}
    ranked_sheet(source, [before, result], model, cameras, masks,
                 output / "before_after_32.png", labels=["BEFORE", "AFTER"])
    ranked_sheet(source, [result], model, cameras, masks,
                 output / "refined_32.png")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine rotation for one saved Franka task")
    parser.add_argument("--scene", default="close_drawer")
    parser.add_argument("--baselines", type=Path, default=ROOT / "runs/franka_final_48")
    parser.add_argument("--initializations", type=Path, default=ROOT / "runs/initial_poses/franka")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "runs/franka_rotation_refine_48" / args.scene
    result = run(args.scene, args.baselines / args.scene / "result.json",
                 args.initializations / args.scene / "initialization.json",
                 args.processed, output)
    print(json.dumps({key: result[key] for key in
                      ("scene", "index", "baseline_loss", "loss",
                       "rotation_change_from_baseline_degrees", "search_history",
                       "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
