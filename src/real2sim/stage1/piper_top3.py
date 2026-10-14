"""Refine three automatically ranked Piper seeds and retain the best final loss."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .initial_pose_batch import ALL_FRAMES
from .metric_calibration import PIPER_TOP3_METHOD, metric_method
from .piper_final import refine_task, write_gallery
from .pipeline import sha256
from .sim3_registration import ROOT


def ranked_seeds(initialization: dict) -> list[dict]:
    if (initialization.get("robot") != "piper" or initialization.get("status") != "initialization_only"
            or initialization.get("rotation_count") != 128
            or len(initialization.get("candidates", [])) != 128
            or initialization.get("pointcloud_frames") != list(ALL_FRAMES)
            or initialization.get("scoring_frames") != list(ALL_FRAMES)):
        raise ValueError("requires the complete Piper 128-seed, 32-frame initialization")
    candidates = initialization["candidates"]
    if len({candidate["index"] for candidate in candidates}) != 128:
        raise ValueError("duplicate Piper seed indices")
    return sorted(candidates, key=lambda item: (item["loss"], item["index"]))[:3]


def scene_inputs(scene: str, initializations: Path, processed: Path) -> tuple[Path, dict, dict, list[dict]]:
    path = initializations / scene / "initialization.json"
    initialization = json.loads(path.read_text(encoding="utf-8"))
    seeds = ranked_seeds(initialization)
    manifest = json.loads((processed / "piper/stage1" / scene / "manifest.json").read_text(encoding="utf-8"))
    video = ROOT / "data/raw/piper" / f"{scene}.mp4"
    model = ROOT / "asset/robot_surfaces/piper/surface_40000.npz"
    if (not video.is_file() or manifest.get("status") != "inference_complete"
            or initialization.get("video_sha256") != manifest.get("video_sha256")
            or sha256(video) != manifest["video_sha256"]
            or not manifest.get("sam3_masks_sha256")
            or initialization.get("sam3_masks_sha256") != manifest["sam3_masks_sha256"]
            or initialization.get("robot_surface_sha256") != sha256(model)):
        raise ValueError(f"stale Piper video, masks, or D435 robot template: {scene}")
    return path, initialization, manifest, seeds


def cached_branch(folder: Path, scene: str, rank: int, seed: dict,
                  initialization_path: Path, manifest: dict, *,
                  source: Path | None = None) -> dict | None:
    source = source if source is not None else ROOT / "data/processed/piper/stage1" / scene
    report_path = folder / "result.json"
    if not report_path.is_file():
        return None
    try:
        result = json.loads(report_path.read_text(encoding="utf-8"))
        if (result.get("robot") != "piper" or result.get("scene") != scene
                or result.get("source") != str(source)
                or result.get("method") != metric_method("piper")
                or result.get("initialization_sha256") != sha256(initialization_path)
                or result.get("sam3_masks_sha256") != manifest["sam3_masks_sha256"]
                or result.get("selected_seed_rank") != rank or result.get("index") != seed["index"]
                or result.get("pointcloud_frames") != list(ALL_FRAMES)
                or result.get("scoring_frames") != list(ALL_FRAMES)
                or result.get("status") != "candidate_unverified" or result.get("accepted") is not False
                or not all((folder / artifact).is_file() for artifact in (
                    "camera_metric.json", "before_after_32.png", "overlay_32.png"))
                or not all((folder / "depth_metric" / f"depth_{index:02d}.npy").is_file()
                           for index in ALL_FRAMES)):
            return None
        return result
    except (OSError, ValueError, KeyError):
        return None


def publish_winner(scene: str, branches: list[tuple[dict, Path]], destination: Path,
                   initialization_path: Path, manifest: dict) -> dict:
    winner, folder = min(branches, key=lambda entry: (entry[0]["loss"], entry[0]["selected_seed_rank"]))
    result = {**winner, "method": PIPER_TOP3_METHOD,
              "selection_policy": "refine initial top 3 independently; select lowest 32-frame final loss",
              "evaluated_seed_ranks": [1, 2, 3],
              "branch_results": [{"seed_rank": branch["selected_seed_rank"], "index": branch["index"],
                                  "initial_loss": branch["initial_loss_saved"], "loss": branch["loss"],
                                  "mean_iou": branch["metrics"]["scored_views"]["mean_iou"]}
                                 for branch, _ in branches]}
    if result["initialization_sha256"] != sha256(initialization_path) or result["sam3_masks_sha256"] != manifest["sam3_masks_sha256"]:
        raise ValueError("winner source changed before publishing")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("camera_metric.json", "before_after_32.png", "overlay_32.png"):
        shutil.copyfile(folder / name, destination / name)
    metric_depth = destination / "depth_metric"
    metric_depth.mkdir(exist_ok=True)
    for index in ALL_FRAMES:
        source = folder / "depth_metric" / f"depth_{index:02d}.npy"
        shutil.copyfile(source, metric_depth / source.name)
    staged = destination / ".result-pending.json"
    staged.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(staged, destination / "result.json")
    return result


def batch(initializations: Path = ROOT / "runs/piper_initial_128_review/piper",
          processed: Path = ROOT / "data/processed", output: Path = ROOT / "runs/piper_stage1_top3",
          top1: Path = ROOT / "runs/piper_stage1_final_128", workers: int = 4,
          only: str | None = None, progress=print) -> list[dict]:
    scenes = sorted(path.stem for path in (ROOT / "data/raw/piper").glob("*.mp4"))
    if only:
        if only not in scenes:
            raise ValueError(f"not an active Piper scene: {only}")
        scenes = [only]
    if workers < 1 or not scenes:
        raise ValueError("requires active Piper scenes and at least one worker")
    sources = {scene: scene_inputs(scene, initializations, processed) for scene in scenes}
    output.mkdir(parents=True, exist_ok=True)
    branches = {scene: {} for scene in scenes}
    pending = []
    for scene in scenes:
        path, _, manifest, seeds = sources[scene]
        for rank, seed in enumerate(seeds, 1):
            folder = (top1 / scene if rank == 1 else output / scene / "branches" / f"rank{rank:02d}")
            cached = cached_branch(folder, scene, rank, seed, path, manifest,
                                   source=processed / "piper/stage1" / scene)
            if cached is not None:
                branches[scene][rank] = (cached, folder)
                progress(f"{scene} rank {rank} R{seed['index']:03d}: reuse validated result")
            elif rank == 1:
                pending.append((scene, rank, seed, output / scene / "branches" / "rank01"))
            else:
                pending.append((scene, rank, seed, folder))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(refine_task, scene, initializations, processed,
                                   output, rank, folder): (scene, rank, seed, folder)
                   for scene, rank, seed, folder in pending}
        for future in as_completed(futures):
            scene, rank, seed, folder = futures[future]
            try:
                future.result()
                path, _, manifest, _ = sources[scene]
                result = cached_branch(folder, scene, rank, seed, path, manifest,
                                       source=processed / "piper/stage1" / scene)
                if result is None:
                    raise ValueError("optimized branch failed provenance/artifact validation")
                branches[scene][rank] = (result, folder)
                progress(f"{scene} rank {rank} R{seed['index']:03d}: final loss {result['loss']:.4f}")
            except Exception as exc:
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                progress(f"{scene} rank {rank}: FAILED {exc}")
    completed = []
    for scene in scenes:
        if len(branches[scene]) != 3:
            continue
        path, _, manifest, _ = sources[scene]
        ordered = [branches[scene][rank] for rank in (1, 2, 3)]
        result = publish_winner(scene, ordered, output / scene, path, manifest)
        progress(f"{scene}: selected rank {result['selected_seed_rank']} R{result['index']:03d}, loss {result['loss']:.4f}")
        completed.append(result)
        (output / "index.json").write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
        write_gallery(completed, output, "piper top3")
    if len(completed) != len(scenes):
        raise RuntimeError(f"only {len(completed)}/{len(scenes)} scenes completed all three branches")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine automatically ranked Piper top-3 seeds and choose best")
    parser.add_argument("--initializations", type=Path, default=ROOT / "runs/piper_initial_128_review/piper")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/piper_stage1_top3")
    parser.add_argument("--top1", type=Path, default=ROOT / "runs/piper_stage1_final_128")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only")
    args = parser.parse_args()
    batch(args.initializations, args.processed, args.output, args.top1, args.workers, args.only)


if __name__ == "__main__":
    main()
