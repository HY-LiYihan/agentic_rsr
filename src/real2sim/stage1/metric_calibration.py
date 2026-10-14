"""Anchor VGGT reconstruction units to URDF meters using a fitted robot Sim(3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .sim3_registration import ROOT


FRAMES = tuple(range(32))
METHOD = "franka_48_top1_sim3_rotation_v1"


def metric_method(robot: str) -> str:
    if robot == "franka":
        return METHOD
    if robot == "piper":
        return "piper_128_top1_sim3_rotation_v1"
    raise ValueError(f"unsupported robot: {robot}")


PIPER_TOP3_METHOD = "piper_128_top3_sim3_rotation_v1"


def valid_metric_method(result: dict) -> bool:
    robot = result.get("robot")
    return robot in ("franka", "piper") and result.get("method") in (
        (metric_method(robot), PIPER_TOP3_METHOD) if robot == "piper" else (metric_method(robot),))


def metric_transforms(result: dict, camera: dict) -> tuple[dict, dict]:
    robot = result.get("robot")
    if not valid_metric_method(result):
        raise ValueError("expected a fixed-seed registration result")
    scale = float(result["scale_vggt_units_per_meter"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("VGGT units per meter must be positive and finite")
    rotation = np.asarray(result["rotation"], dtype=float)
    translation = np.asarray(result["translation_vggt_world"], dtype=float)
    intrinsics = np.asarray(camera["intrinsics"], dtype=float)
    extrinsics = np.asarray(camera["extrinsics"], dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,) \
            or intrinsics.shape != (32, 3, 3) or extrinsics.shape != (32, 3, 4) \
            or camera.get("extrinsics_direction") != "world_to_camera" \
            or not all(np.isfinite(array).all() for array in (rotation, translation, intrinsics, extrinsics)):
        raise ValueError("invalid pose or 32-frame world-to-camera calibration")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4) \
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4):
        raise ValueError("robot pose must be a proper rigid rotation")

    meters_per_unit = 1. / scale
    robot_pose = np.eye(4)
    robot_pose[:3, :3] = rotation
    robot_pose[:3, 3] = translation * meters_per_unit
    metric_extrinsics = extrinsics.copy()
    metric_extrinsics[:, :, 3] *= meters_per_unit
    scale_source = f"fitted {'Franka' if robot == 'franka' else robot} URDF surface (meters)"
    calibrated_camera = {"intrinsics": intrinsics.tolist(),
                         "extrinsics": metric_extrinsics.tolist(),
                         "extrinsics_direction": "world_to_camera",
                         "camera_coordinate_system": camera.get("camera_coordinate_system"),
                         "source_frame_indices": camera.get("source_frame_indices"),
                         "world_and_camera_translation_unit": "m",
                         "depth_unit": "m",
                         "scale_source": scale_source}
    calibration = {"status": "urdf_anchored_metric_scale", "unit": "m",
                   "scale_source": scale_source,
                   "scale_vggt_units_per_meter": scale,
                   "meters_per_vggt_unit": meters_per_unit,
                   "metric_world_definition": "VGGT world axes and origin, all positions divided by scale",
                   "robot_pose_metric_world_from_base": robot_pose.tolist(),
                   "camera_file": "camera_metric.json",
                   "depth_directory": "depth_metric",
                   "depth_frames": list(FRAMES),
                   "registration_validation": "not independently verified"}
    return calibration, calibrated_camera


def rebase_to_robot_base(calibration: dict, camera: dict) -> tuple[dict, dict]:
    world_from_base = np.asarray(calibration["robot_pose_metric_world_from_base"], dtype=float)
    extrinsics = np.asarray(camera["extrinsics"], dtype=float)
    if world_from_base.shape != (4, 4) or extrinsics.shape != (32, 3, 4):
        raise ValueError("invalid metric robot pose or camera extrinsics")
    camera_from_base = np.empty_like(extrinsics)
    camera_from_base[:, :, :3] = extrinsics[:, :, :3] @ world_from_base[:3, :3]
    camera_from_base[:, :, 3] = (extrinsics[:, :, :3] @ world_from_base[:3, 3]
                               + extrinsics[:, :, 3])
    rebased_camera = {**camera, "extrinsics": camera_from_base.tolist(),
                      "world_frame": "robot_base"}
    rebased_calibration = {**calibration,
                           "original_metric_world_definition": calibration["metric_world_definition"],
                           "robot_pose_original_metric_world_from_base": world_from_base.tolist(),
                           "metric_world_definition": "robot base-link axes and origin (URDF meters)",
                           "robot_pose_metric_world_from_base": np.eye(4).tolist(),
                           "world_frame": "robot_base"}
    return rebased_calibration, rebased_camera


def export_metric_scene(source: Path, output: Path, result: dict) -> dict:
    camera = json.loads((source / "camera.json").read_text(encoding="utf-8"))
    calibration, metric_camera = metric_transforms(result, camera)
    depth_paths = [source / "depth" / f"depth_{frame:02d}.npy" for frame in FRAMES]
    if not all(path.is_file() for path in depth_paths):
        raise ValueError(f"missing one of 32 depth frames in {source}")
    depth_directory = output / "depth_metric"
    depth_directory.mkdir(parents=True, exist_ok=True)
    for frame, path in enumerate(depth_paths):
        depth = np.load(path, allow_pickle=False)
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError(f"invalid depth frame {frame}")
        np.save(depth_directory / f"depth_{frame:02d}.npy",
                (depth / calibration["scale_vggt_units_per_meter"]).astype(np.float32))
    (output / "camera_metric.json").write_text(json.dumps(metric_camera, indent=2) + "\n",
                                                encoding="utf-8")
    return calibration


def backfill(results: Path, processed: Path) -> list[dict]:
    from .franka_final import write_gallery

    index = json.loads((results / "index.json").read_text(encoding="utf-8"))
    if not index or any(item.get("robot") != "franka" or item.get("method") != METHOD
                        for item in index):
        raise ValueError("expected completed Franka registration results")
    updated = []
    for entry in index:
        scene = entry["scene"]
        folder = results / scene
        report = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if report != entry:
            raise ValueError(f"result.json and index.json disagree: {scene}")
        calibration = export_metric_scene(processed / "franka" / "stage1" / scene, folder, report)
        report["metric_calibration"] = calibration
        report["reason"] = "Metric scale anchored by Franka URDF; independent registration validation pending"
        (folder / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        updated.append(report)
    (results / "index.json").write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    write_gallery(updated, results)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Franka Stage 1 metric depths and camera poses")
    parser.add_argument("--results", type=Path, default=ROOT / "runs/franka_stage1_final_48")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    results = backfill(args.results, args.processed)
    print(f"Exported URDF-anchored meter scale for {len(results)} Franka scenes")


if __name__ == "__main__":
    main()
