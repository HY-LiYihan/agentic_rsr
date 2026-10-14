"""Publish compact, reproducible fixed-pose Stage 1 candidates."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import trimesh

from real2sim.blender import resolve_blender
from .geometry import MODEL, description
from .metric_calibration import metric_transforms, rebase_to_robot_base, valid_metric_method
from .pipeline import sha256
from .promote_sam3_masks import mask_digest
from .registration import REFERENCE
from .robot_surface import BASE_LINKS, FRANKA_URDF, POSES, colored_visual_meshes, link_poses
from .piper_wrist import assembly_sha256, piper_description
from .sim3_registration import ROOT, project


def registration_summary(result: dict, source: dict) -> dict:
    robot = result.get("robot")
    if not valid_metric_method(result):
        raise ValueError("expected finalized automatic registration")
    if result.get("status") != "candidate_unverified" or result.get("accepted") is not False:
        raise ValueError("registration status must remain unverified")
    if result.get("pointcloud_frames") != list(range(32)) or result.get("scoring_frames") != list(range(32)):
        raise ValueError("expected all 32 frames for fitting and scoring")
    return {"status": "candidate_unverified", "accepted": False,
            "reason": "32 frames used for fitting and scoring; no independent validation",
            "method": result["method"], "robot": robot, "scene": result["scene"],
            "source_video": source["source_video"], "video_sha256": source["video_sha256"],
            "models": source.get("models"), "initialization_sha256": result["initialization_sha256"],
            "selected_seed_rank": result["selected_seed_rank"], "seed_index": result["index"],
            "rotation_count": result["rotation_count"], "model_stages": result["model_stages"],
            "rotation_steps_degrees": result["rotation_steps_degrees"],
            "minimum_loss_improvement": result["minimum_loss_improvement"],
            "pointcloud_frames": result["pointcloud_frames"],
            "scoring_frames": result["scoring_frames"], "reference_indices": list(REFERENCE),
            "loss": result["loss"], "baseline_loss": result["baseline_loss"],
            "metrics": result["metrics"], "metric_calibration": result["metric_calibration"],
            f"{robot}_urdf_sha256": hashlib.sha256(
                (FRANKA_URDF if robot == "franka" else MODEL).read_bytes()).hexdigest(),
            "joint_positions_radians_or_meters": POSES[robot]}


def prepare_scene(source: Path, result_folder: Path, result: dict,
                  workspace: Path, model: np.ndarray) -> tuple[dict, dict]:
    robot = result["robot"]
    camera = json.loads((source / "camera.json").read_text(encoding="utf-8"))
    calibration, metric_camera = metric_transforms(result, camera)
    if result["metric_calibration"] != calibration:
        raise ValueError("saved metric calibration disagrees with registration")
    cached_camera = json.loads((result_folder / "camera_metric.json").read_text(encoding="utf-8"))
    if cached_camera != metric_camera:
        raise ValueError("cached metric camera differs from source and fitted scale")
    delivery_calibration, delivery_camera = rebase_to_robot_base(calibration, metric_camera)

    metric_depth = workspace / "depth_metric"
    metric_depth.mkdir()
    for index in range(32):
        original = np.load(source / "depth" / f"depth_{index:02d}.npy", allow_pickle=False)
        cached = np.load(result_folder / "depth_metric" / f"depth_{index:02d}.npy",
                         allow_pickle=False)
        mask = np.load(source / "sam3_masks" / f"mask_{index:02d}.npy", allow_pickle=False)
        if original.shape != (288, 512) or cached.shape != original.shape or mask.shape != original.shape:
            raise ValueError(f"observation shape mismatch at frame {index}")
        if not np.isfinite(cached).all() or not np.allclose(
                cached, original / calibration["scale_vggt_units_per_meter"], rtol=1e-5, atol=1e-6):
            raise ValueError(f"metric depth mismatch at frame {index}")
        shutil.copyfile(result_folder / "depth_metric" / f"depth_{index:02d}.npy",
                        metric_depth / f"depth_{index:02d}.npy")

    (workspace / "camera_metric.json").write_text(json.dumps(delivery_camera, indent=2) + "\n",
                                                   encoding="utf-8")
    intrinsics = np.asarray(camera["intrinsics"])
    extrinsics = np.asarray(camera["extrinsics"])
    pose = (np.asarray(result["rotation"]), np.asarray(result["translation_vggt_world"]),
            result["scale_vggt_units_per_meter"])
    tiles = []
    for index in REFERENCE:
        image = cv2.imread(str(source / "frames" / f"{index:02d}.jpg"))
        mask = np.load(source / "sam3_masks" / f"mask_{index:02d}.npy", allow_pickle=False)
        if image is None or image.shape[:2] != mask.shape:
            raise ValueError(f"missing or misaligned reference RGB {index}")
        footprint, _ = project(model, pose, {"intrinsics": intrinsics, "extrinsics": extrinsics},
                               index, mask.shape)
        overlay = image.copy()
        overlay[mask] = (0.65 * overlay[mask] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
        overlay[footprint] = (0.5 * overlay[footprint] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.putText(overlay, f"Frame {index:02d}  green=SAM3 red={robot}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        if not cv2.imwrite(str(workspace / f"overlay_{index:02d}.png"), overlay):
            raise OSError(f"cannot save overlay {index}")
        tiles.append(overlay)
    montage = np.concatenate((np.concatenate(tiles[:2], axis=1),
                              np.concatenate(tiles[2:], axis=1)), axis=0)
    if not cv2.imwrite(str(workspace / "quality_check.png"), montage):
        raise OSError("cannot save quality check montage")

    urdf = FRANKA_URDF if robot == "franka" else MODEL
    root = ET.parse(urdf).getroot() if robot == "franka" else piper_description()
    frames, _ = link_poses(root, BASE_LINKS[robot], POSES[robot])
    colored_meshes = colored_visual_meshes(root, urdf, robot, frames)
    groups = {}
    for mesh, rgba in colored_meshes:
        groups.setdefault(rgba, []).append(mesh)
    robot_meshes = []
    for index, (rgba, meshes) in enumerate(groups.items()):
        path = workspace / f"robot_visual_{index:02d}.obj"
        trimesh.util.concatenate(meshes).export(path)
        robot_meshes.append({"obj": str(path), "rgba": rgba})
    return {"width": 512, "height": 288,
            "meshes": robot_meshes,
            "robot_object": "FR3" if robot == "franka" else "Piper",
            "world_from_base": delivery_calibration["robot_pose_metric_world_from_base"],
            "views": [{"frame": index, "intrinsic": delivery_camera["intrinsics"][index],
                       "extrinsic": delivery_camera["extrinsics"][index],
                       "image": str(source / "frames" / f"{index:02d}.jpg")}
                      for index in REFERENCE]}, delivery_calibration


def publish(source: Path, result_path: Path, blender: str | Path | None, model: np.ndarray,
            *, prune: bool = False, initializations: Path | None = None) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    robot = result["robot"]
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "inference_complete" or manifest.get("robot") != robot:
        raise ValueError(f"{robot} inference must be complete: {source}")
    if result.get("scene") != source.name or result.get("source") != str(source):
        raise ValueError("registration does not belong to this inference directory")
    video = Path(manifest["source_video"])
    if not video.is_file() or sha256(video) != manifest["video_sha256"]:
        raise ValueError("original video is missing or changed since inference")
    urdf = FRANKA_URDF if robot == "franka" else MODEL
    asset = json.loads((ROOT / "asset/robot_surfaces" / robot / "metadata.json").read_text(encoding="utf-8"))
    if (asset["urdf_sha256"] != sha256(urdf) or
            any(asset["joint_positions"].get(name) != value for name, value in POSES[robot].items())):
        raise ValueError("URDF or robot joint pose changed since the registration template was sampled")
    if robot == "piper" and asset.get("assembly_sha256") != assembly_sha256():
        raise ValueError("Piper wrist D435/stand changed since the registration template was sampled")
    if initializations is None:
        initializations = ROOT / ("runs/piper_initial_128_review/piper" if robot == "piper"
                                  else "runs/initial_poses/franka")
    initialization = initializations / source.name / "initialization.json"
    if sha256(initialization) != result["initialization_sha256"]:
        raise ValueError("registration seed no longer matches the scored initialization")
    if robot == "piper" and json.loads(initialization.read_text(encoding="utf-8")).get(
            "robot_surface_sha256") != sha256(ROOT / "asset/robot_surfaces/piper/surface_40000.npz"):
        raise ValueError("Piper point cloud changed after the scored initialization")
    if robot == "piper" and (not manifest.get("sam3_masks_sha256") or
                             result.get("sam3_masks_sha256") != manifest["sam3_masks_sha256"] or
                             mask_digest(source / "sam3_masks") != manifest["sam3_masks_sha256"]):
        raise ValueError("Piper masks changed since registration")
    summary = registration_summary(result, manifest)
    with tempfile.TemporaryDirectory(prefix=".delivery-", dir=source) as temporary:
        workspace = Path(temporary)
        spec, delivery_calibration = prepare_scene(source, result_path.parent, result, workspace, model)
        summary["metric_calibration"] = delivery_calibration
        (workspace / "scene_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        executable = resolve_blender(blender)
        completed = subprocess.run(
            [str(executable), "-b", "-noaudio", "--python", str(Path(__file__).with_name("franka_blender_scene.py")),
             "--", str(workspace / "scene_spec.json"), str(workspace / "initial.blend")],
            capture_output=True, text=True, timeout=300)
        if completed.returncode or not (workspace / "initial.blend").is_file() \
                or "Traceback (most recent call last)" in completed.stderr:
            raise RuntimeError(f"Blender validation failed for {source.name}: "
                               f"{(completed.stdout + completed.stderr)[-2500:]}")
        (workspace / "registration.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                     encoding="utf-8")
        delivery = {"schema": f"{robot}-stage1-delivery-v2", "status": "candidate_unverified",
                    "accepted": False, "video_sha256": manifest["video_sha256"],
                    "reference_indices": list(REFERENCE), "depth_unit": "m",
                    "world_frame": "robot_base",
                    "artifacts": ["frames/", "sam3_masks/", "depth_metric/",
                                  "camera_metric.json", "registration.json", "initial.blend",
                                  "reference/", "quality_check.png"]}
        (workspace / "stage1_delivery.json").write_text(json.dumps(delivery, indent=2) + "\n",
                                                        encoding="utf-8")
        for name in ("camera_metric.json", "registration.json", "initial.blend",
                     "quality_check.png", "stage1_delivery.json"):
            os.replace(workspace / name, source / name)
        for index in REFERENCE:
            os.replace(workspace / f"overlay_{index:02d}.png",
                       source / "reference" / f"overlay_{index:02d}.png")
        destination = source / "depth_metric"
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(workspace / "depth_metric", destination)
    if prune:
        for name in ("raw_vggt.npz", "depth_vis", "sam3_masks_vis"):
            path = source / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
        for name in ("depth_vis", "sam3_masks_vis"):
            manifest.get("artifacts", {}).pop(name, None)
    manifest["stage1_delivery"] = "stage1_delivery.json"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=source,
                                     prefix=".manifest-", delete=False) as pending:
        json.dump(manifest, pending, indent=2)
        pending.write("\n")
    os.replace(pending.name, source / "manifest.json")
    return delivery


def write_index(directory: Path, scenes: list[str], robot: str = "franka") -> None:
    rows = []
    for scene in scenes:
        source = directory / scene
        report = json.loads((source / "registration.json").read_text(encoding="utf-8"))
        name = html.escape(scene)
        rows.append(f'<li><h2>{name}: loss {report["loss"]:.4f} '
                    f'({report["status"]})</h2><a href="{name}/quality_check.png">'
                    f'<img loading="lazy" src="{name}/quality_check.png" '
                    f'alt="{name} 四帧机械臂投影" style="max-width:100%"></a></li>')
    (directory / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        f'<title>{robot} Stage 1 交付总览</title><h1>{robot} 四帧交付预览</h1>'
        '<p>URDF 锚定米制尺度；配准仍未独立验收，不能作为 Stage 2 的放行依据。</p>'
        '<ol>' + "\n".join(rows) + '</ol></html>\n', encoding="utf-8")


def main(default_robot: str = "franka") -> None:
    parser = argparse.ArgumentParser(description="Publish and validate Stage 1 Blender deliverables")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed" / default_robot / "stage1")
    parser.add_argument("--results", type=Path, default=ROOT / "runs" /
                        ("piper_stage1_top3" if default_robot == "piper" else "franka_stage1_final_48"))
    parser.add_argument("--initializations", type=Path,
                        help="Robot initialization directory containing <task>/initialization.json")
    parser.add_argument("--only", help="Only export one named task")
    parser.add_argument("--blender", help="Blender executable path or command; defaults to BLENDER_BIN or PATH")
    parser.add_argument("--prune", action="store_true", help="Remove reproducible preview caches after successful export")
    arguments = parser.parse_args()
    scenes = [arguments.only] if arguments.only else sorted(path.name for path in arguments.processed.iterdir()
                                                   if path.is_dir() and not path.name.startswith("."))
    if not scenes:
        parser.error("no Franka scenes found")
    with np.load(ROOT / "asset/robot_surfaces" / default_robot / "surface_40000.npz") as data:
        cloud = data["points"]
    model = cloud[np.random.default_rng(2027).permutation(len(cloud))[:20000]]
    failures = []
    if default_robot == "piper":
        scenes = [scene for scene in scenes if (ROOT / "data/raw/piper" / f"{scene}.mp4").is_file()]
    for index, scene in enumerate(scenes, 1):
        print(f"[{index}/{len(scenes)}] {default_robot}/{scene}", flush=True)
        try:
            report = json.loads((arguments.results / scene / "result.json").read_text(encoding="utf-8"))
            if report["robot"] != default_robot:
                raise ValueError("result robot does not match requested robot")
            delivery = publish(arguments.processed / scene, arguments.results / scene / "result.json",
                               arguments.blender, model, prune=arguments.prune,
                               initializations=arguments.initializations)
            print(f"  {delivery['status']}; initial.blend validated", flush=True)
        except Exception as exc:
            failures.append(scene)
            print(f"  FAILED: {exc}", flush=True)
    if failures:
        raise SystemExit(f"Failed Franka deliveries: {', '.join(failures)}")
    if not arguments.only:
        write_index(arguments.processed, scenes, default_robot)


if __name__ == "__main__":
    main()
