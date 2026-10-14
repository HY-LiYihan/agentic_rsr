"""Experimental multiscale Sim(3) registration from cached SAM3 and VGGT outputs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .registration import REFERENCE, camera_world


ROOT = Path(__file__).resolve().parents[3]
TRAIN_VIEWS = (5, 12, 19, 26)
SIZES = (5000, 10000, 40000)


def apply_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
               scale: float) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def similarity_fit(source: np.ndarray, target: np.ndarray, bounds: tuple[float, float]):
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    left, singular, right = np.linalg.svd((source - source_center).T @ (target - target_center))
    diagonal = np.ones(3)
    diagonal[-1] = np.linalg.det(right.T @ left.T)
    rotation = right.T @ np.diag(diagonal) @ left.T
    scale = np.clip(np.dot(singular, diagonal) / np.sum((source - source_center) ** 2), *bounds)
    translation = target_center - scale * rotation @ source_center
    return rotation, translation, float(scale)


def search_rotations(count: int, *, include_identity: bool = True) -> list[np.ndarray]:
    rotations = [np.eye(3)] if include_identity else []
    for index in range(count):
        first = (index + 0.5) / count
        second = (index * 0.618033988749895) % 1
        third = (index * 0.754877666246693) % 1
        quaternion = np.array([math.sqrt(1 - first) * math.sin(2 * math.pi * second),
                               math.sqrt(1 - first) * math.cos(2 * math.pi * second),
                               math.sqrt(first) * math.sin(2 * math.pi * third),
                               math.sqrt(first) * math.cos(2 * math.pi * third)])
        vector = quaternion[:3]
        scalar = quaternion[3]
        cross = np.array([[0, -vector[2], vector[1]],
                          [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])
        rotations.append(np.eye(3) + 2 * scalar * cross + 2 * cross @ cross)
    return rotations


def match_nearest(source: np.ndarray, target: np.ndarray, index=None):
    if index is None:
        index = cv2.flann_Index(np.ascontiguousarray(target, dtype=np.float32),
                               {"algorithm": 1, "trees": 4})
    matches, distances = index.knnSearch(np.ascontiguousarray(source, dtype=np.float32), 1,
                                         params={"checks": 64})
    return matches[:, 0], np.sqrt(distances[:, 0])


def refine(model: np.ndarray, observed: np.ndarray, initial: tuple,
           bounds: tuple[float, float], iterations: int) -> tuple:
    rotation, translation, scale = initial
    target_index = cv2.flann_Index(np.ascontiguousarray(observed, dtype=np.float32),
                                   {"algorithm": 1, "trees": 4})
    for _ in range(iterations):
        predicted = apply_pose(model, rotation, translation, scale)
        matches, distances = match_nearest(predicted, observed, target_index)
        cutoff = np.quantile(distances, 0.65)
        keep = distances <= cutoff
        next_rotation, next_translation, next_scale = similarity_fit(
            model[keep], observed[matches[keep]], bounds)
        change = np.linalg.norm(next_translation - translation) + abs(next_scale - scale)
        rotation, translation, scale = next_rotation, next_translation, next_scale
        if change < 1e-5:
            break
    return rotation, translation, scale


def project(model: np.ndarray, pose: tuple, camera: dict, index: int,
            shape: tuple[int, int], *, world_points: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    points = world_points if world_points is not None else apply_pose(model, *pose)
    extrinsic = camera["extrinsics"][index]
    points = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    points = points[points[:, 2] > 0]
    height, width = shape
    predicted = np.full(shape, np.inf, dtype=np.float32)
    if len(points):
        intrinsic = camera["intrinsics"][index]
        horizontal = np.rint(points[:, 0] * intrinsic[0, 0] / points[:, 2] + intrinsic[0, 2]).astype(int)
        vertical = np.rint(points[:, 1] * intrinsic[1, 1] / points[:, 2] + intrinsic[1, 2]).astype(int)
        inside = (horizontal >= 0) & (horizontal < width) & (vertical >= 0) & (vertical < height)
        np.minimum.at(predicted, (vertical[inside], horizontal[inside]), points[inside, 2])
    footprint = cv2.dilate(np.isfinite(predicted).astype(np.uint8),
                           np.ones((5, 5), np.uint8)).astype(bool)
    return footprint, predicted


def score_views(model: np.ndarray, pose: tuple, camera: dict, depths: dict,
                masks: dict, indices: tuple[int, ...],
                world_points: np.ndarray | None = None) -> tuple[dict, list]:
    per_frame = []
    for index in indices:
        mask = masks[index]
        footprint, predicted = project(model, pose, camera, index, mask.shape,
                                       world_points=world_points)
        intersection = footprint & mask
        valid = np.isfinite(predicted) & mask & np.isfinite(depths[index]) & (depths[index] > 0)
        residual = np.abs(predicted[valid] - depths[index][valid])
        per_frame.append({"frame": index,
                          "iou": float(intersection.sum() / max((footprint | mask).sum(), 1)),
                          "recall": float(intersection.sum() / max(mask.sum(), 1)),
                          "projected_pixels": int(footprint.sum()),
                          "valid_depth_pixels": int(valid.sum()),
                          "median_depth_residual_vggt": float(np.median(residual)) if len(residual) else None})
    median = [item["median_depth_residual_vggt"] for item in per_frame
              if item["median_depth_residual_vggt"] is not None]
    summary = {"mean_iou": float(np.mean([item["iou"] for item in per_frame])),
               "mean_recall": float(np.mean([item["recall"] for item in per_frame])),
               "median_depth_residual_vggt": float(np.median(median)) if median else None,
               "frames_with_depth": len(median)}
    return summary, per_frame


def ranking(model: np.ndarray, observed: np.ndarray, pose: tuple, camera: dict,
            depths: dict, masks: dict, indices: tuple[int, ...] = TRAIN_VIEWS) -> tuple[float, dict]:
    predicted = apply_pose(model, *pose)
    views, per_frame = score_views(model, pose, camera, depths, masks, indices, predicted)
    _, distances = match_nearest(predicted, observed)
    reverse_points = observed[::max(1, len(observed) // 8000)]
    _, reverse_distances = match_nearest(reverse_points, predicted)
    radius = max(float(np.linalg.norm(np.quantile(observed, 0.95, axis=0) -
                                      np.quantile(observed, 0.05, axis=0))), 1e-6)
    trimmed = float(np.quantile(distances, 0.60) / radius)
    reverse_trimmed = float(np.quantile(reverse_distances, 0.60) / radius)
    overlap = float((distances < 0.05 * radius).mean())
    reverse_overlap = float((reverse_distances < 0.05 * radius).mean())
    depth = views["median_depth_residual_vggt"]
    depth_penalty = min(depth / radius, 1.0) if depth is not None else 1.0
    loss = 0.50 * (1 - views["mean_iou"]) + 0.12 * (1 - views["mean_recall"])
    loss += 0.14 * min(trimmed * 5, 1.0) + 0.08 * min(reverse_trimmed * 5, 1.0)
    loss += 0.08 * (1 - min(overlap, reverse_overlap)) + 0.08 * depth_penalty
    return float(loss), {"scored_views": views, "per_frame": per_frame,
                         "model_to_observed_over_radius": trimmed,
                         "observed_to_model_over_radius": reverse_trimmed,
                         "model_to_observed_overlap": overlap,
                         "observed_to_model_overlap": reverse_overlap}


def load_observations(directory: Path, *, held_out: tuple[int, ...] = REFERENCE):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "inference_complete":
        raise ValueError(f"inference is not complete: {directory}")
    camera = json.loads((directory / "camera.json").read_text())
    if camera.get("extrinsics_direction") != "world_to_camera":
        raise ValueError("expected world_to_camera extrinsics")
    camera = {key: np.asarray(camera[key], dtype=float) for key in ("intrinsics", "extrinsics")}
    depths, masks, clouds, counts = {}, {}, [], []
    for index in range(32):
        depths[index] = np.load(directory / "depth" / f"depth_{index:02}.npy")
        masks[index] = np.load(directory / "sam3_masks" / f"mask_{index:02}.npy")
        if depths[index].shape != masks[index].shape:
            raise ValueError(f"depth/mask size mismatch for frame {index}")
        if index in held_out:
            continue
        depth, mask = depths[index], masks[index]
        vertical, horizontal = np.mgrid[0:depth.shape[0]:4, 0:depth.shape[1]:4]
        value = depth[::4, ::4]
        selected = mask[::4, ::4] & np.isfinite(value) & (value > 0)
        intrinsic = camera["intrinsics"][index]
        z = value[selected]
        points = np.column_stack(((horizontal[selected] - intrinsic[0, 2]) * z / intrinsic[0, 0],
                                  (vertical[selected] - intrinsic[1, 2]) * z / intrinsic[1, 1], z))
        transform = camera_world(camera["extrinsics"][index])
        world = points @ transform[:3, :3].T + transform[:3, 3]
        if len(world):
            clouds.append(world.astype(np.float32))
        counts.append({"frame": index, "points": len(world)})
    if len(clouds) < 16:
        raise ValueError("fewer than 16 usable training frames")
    observed = np.concatenate(clouds)
    if len(observed) > 30000:
        observed = observed[np.linspace(0, len(observed) - 1, 30000, dtype=int)]
    return manifest, camera, depths, masks, observed, counts


def register_scene(directory: Path, output: Path, rotations: int = 48) -> dict:
    start = time.monotonic()
    manifest, camera, depths, masks, observed, counts = load_observations(directory)
    robot = manifest["robot"]
    if robot not in ("piper", "franka"):
        raise ValueError(f"unsupported robot: {robot}")
    model_directory = ROOT / "asset/robot_surfaces" / robot
    models = {}
    for size in SIZES:
        with np.load(model_directory / f"surface_{size}.npz") as data:
            models[size] = data["points"].astype(float)
    center = observed.mean(axis=0)
    model_center = models[5000].mean(axis=0)
    model_size = np.linalg.norm(np.quantile(models[5000], .95, axis=0) -
                                np.quantile(models[5000], .05, axis=0))
    cloud_size = np.linalg.norm(np.quantile(observed, .95, axis=0) -
                                np.quantile(observed, .05, axis=0))
    reference_scale = float(cloud_size / model_size)
    bounds = (0.35 * reference_scale, 2.5 * reference_scale)
    # Search rotations and scale around the measured cloud centers, without an RANSAC axis.
    coarse = []
    for rotation in search_rotations(rotations):
        for factor in (0.7, 1.0, 1.4):
            scale = reference_scale * factor
            translation = center - scale * rotation @ model_center
            pose = (rotation, translation, scale)
            views, _ = score_views(models[5000], pose, camera, depths, masks, TRAIN_VIEWS)
            coarse.append((1 - views["mean_iou"], pose))
    coarse.sort(key=lambda item: item[0])
    candidates = []
    for _, pose in coarse[:12]:
        fitted = refine(models[5000], observed, pose, bounds, iterations=7)
        score, metrics = ranking(models[5000], observed, fitted, camera, depths, masks)
        candidates.append((score, fitted, metrics))
    candidates.sort(key=lambda item: item[0])
    for size, limit, iterations in ((10000, 4, 8), (40000, 2, 10)):
        refined = []
        for _, pose, _ in candidates[:limit]:
            fitted = refine(models[size], observed, pose, bounds, iterations)
            score, metrics = ranking(models[size], observed, fitted, camera, depths, masks)
            refined.append((score, fitted, metrics))
        candidates = sorted(refined, key=lambda item: item[0])
    loss, pose, metrics = candidates[0]
    held_out, frames = score_views(models[40000], pose, camera, depths, masks, REFERENCE)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "observation.npz", points=observed)
    trimesh.points.PointCloud(observed).export(output / "observation.ply")
    np.savez_compressed(output / "aligned_model.npz", points=apply_pose(models[40000], *pose))
    for index in REFERENCE:
        image = cv2.imread(str(directory / "frames" / f"{index:02}.jpg"))
        footprint, _ = project(models[40000], pose, camera, index, masks[index].shape)
        image[masks[index]] = (image[masks[index]] * .5 + np.array([0, 180, 0]) * .5).astype(np.uint8)
        image[footprint] = (image[footprint] * .35 + np.array([0, 0, 255]) * .65).astype(np.uint8)
        cv2.imwrite(str(output / f"frame_{index:02}_overlay.png"), image)
    rotation, translation, scale = pose
    result = {"status": "candidate_unverified", "accepted": False,
              "reason": "VGGT scale is not independently metric calibrated; rigid pose and joint-state agreement are unverified",
              "source": str(directory), "robot": robot, "model": str(model_directory),
              "scale_vggt_units_per_meter": scale, "rotation_world_from_base": rotation.tolist(),
              "translation_vggt_world": translation.tolist(),
              "similarity_world_from_base": np.block([[scale * rotation, translation[:, None]],
                                                       [np.zeros((1, 3)), np.ones((1, 1))]]).tolist(),
              "coarse_scale_guess": reference_scale, "scale_search_bounds": bounds,
              "initialization": "cloud center + deterministic SO(3) and scale grid",
              "rotation_seeds": rotations + 1, "coarse_candidates": len(coarse),
              "training_frames": [i for i in range(32) if i not in REFERENCE],
              "ranking_frames": TRAIN_VIEWS, "held_out_frames": REFERENCE,
              "training_point_counts": counts, "observed_points": len(observed),
              "training_loss": loss, "training_metrics": metrics,
              "held_out": held_out, "held_out_per_frame": frames,
              "elapsed_seconds": time.monotonic() - start}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental fixed-arm multiscale similarity registration")
    parser.add_argument("--input", type=Path, required=True, help="cached processed/<robot>/stage1/<video>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotations", type=int, default=48)
    args = parser.parse_args()
    if args.rotations < 1:
        parser.error("--rotations must be positive")
    result = register_scene(args.input, args.output, rotations=args.rotations)
    print(json.dumps({key: result[key] for key in ("status", "robot", "training_loss", "held_out", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
