"""Stage 1, isolated and fail-closed: candidate artifacts never imply acceptance."""

import hashlib
import json
import subprocess
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from perception.config import (backend_manifest, create_sam3, create_vggt,
                               release_backend)

from real2sim.blender import resolve_blender
from real2sim.stage1.geometry import ASSET_ROOT, visual_meshes
from real2sim.stage1.registration import REFERENCE, camera_world, register, unproject


ROOT = Path(__file__).resolve().parents[3]
THRESHOLDS = Path(__file__).with_name("thresholds.json")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(video: Path, folder: Path) -> tuple[list[Path], list[int]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"unable to read video: {video}")
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 32:
            raise ValueError(f"video needs at least 32 frames, found {count}")
        indices = np.linspace(0, count - 1, 32).astype(int).tolist()
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for order, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"video frame {index} cannot be decoded")
            height, width = frame.shape[:2]
            scale = min(512 / width, 288 / height)
            scaled = cv2.resize(frame, (round(width * scale), round(height * scale)))
            canvas = np.zeros((288, 512, 3), dtype=np.uint8)
            top = (288 - scaled.shape[0]) // 2
            left = (512 - scaled.shape[1]) // 2
            canvas[top:top + scaled.shape[0], left:left + scaled.shape[1]] = scaled
            destination = folder / f"{order:02d}.jpg"
            if not cv2.imwrite(str(destination), canvas):
                raise OSError(f"unable to save frame {index}")
            paths.append(destination)
        return paths, indices
    finally:
        capture.release()


def backends() -> tuple[object, object, dict]:
    vggt, sam3 = create_vggt(), create_sam3()
    return vggt, sam3, {"vggt": backend_manifest(vggt, "vggt"),
                        "sam3": backend_manifest(sam3, "sam3")}


def validate(arrays: dict, scale: float, *, metric_range: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = arrays["depth"].astype(np.float64) * scale
    confidence = arrays["depth_conf"]
    intrinsics, extrinsics = arrays["intrinsics"], arrays["extrinsics"].astype(np.float64).copy()
    extrinsics[:, :, 3] *= scale
    if (depth.shape != (32, 288, 512) or confidence.shape != depth.shape or
            intrinsics.shape != (32, 3, 3) or extrinsics.shape != (32, 3, 4)):
        raise ValueError("reconstruction shapes do not match the 32 aligned 512x288 RGB images")
    if (not np.isfinite(depth).all() or not np.isfinite(confidence).all() or
            not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all()):
        raise ValueError("nonfinite reconstruction")
    if metric_range:
        if np.count_nonzero((depth > 0.05) & (depth < 5)) / depth.size < 0.3:
            raise ValueError("insufficient plausible positive depth at the declared metric scale")
    elif np.count_nonzero(depth > 0) / depth.size < 0.3:
        raise ValueError("insufficient positive depth in VGGT reconstruction")
    for intrinsic, extrinsic in zip(intrinsics, extrinsics):
        rotation = extrinsic[:, :3]
        if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or
                not 0.5 < intrinsic[0, 0] / intrinsic[1, 1] < 2 or
                not 0 <= intrinsic[0, 2] < 512 or not 0 <= intrinsic[1, 2] < 288 or
                not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-3) or
                not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-2) or
                not np.isclose(np.linalg.det(rotation), 1, atol=1e-2)):
            raise ValueError("invalid camera intrinsics or world-to-camera extrinsics")
    return depth.astype(np.float32), intrinsics, extrinsics


def pointcloud(depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    points = unproject(depth, intrinsic, np.ones_like(depth, dtype=bool), stride=3)
    matrix = camera_world(extrinsic)
    return (points @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)


def run(video: Path, task: str, output: Path, *, metric_scale: float | None = None,
        vggt=None, sam3=None, blender: str | None = None) -> tuple[Path, bool]:
    if not task.strip():
        raise ValueError("task description is required")
    if metric_scale is not None and (not np.isfinite(metric_scale) or metric_scale <= 0):
        raise ValueError("metric scale must be positive and finite")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    folder = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    folder.mkdir()
    manifest = {"schema": "stage1-v1", "status": "running", "task": task,
                "video": str(video.resolve()), "metric_scale_m_per_unit": metric_scale,
                "scale_source": "user-supplied external metric calibration" if metric_scale is not None else "unknown; candidate only",
                "run_id": folder.name, "thresholds": json.loads(THRESHOLDS.read_text()),
                "artifacts": {}, "robot": "piper", "coordinates": "VGGT world, OpenCV camera (+X right, +Y down, +Z forward)"}
    save_json(folder / "manifest.json", manifest)
    try:
        manifest["video_sha256"] = sha256(video)
        manifest["robot_asset_revision"] = subprocess.run(
            ["git", "-C", str(ROOT / "external/robot_control"), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        manifest["robot_description_sha256"] = sha256(ASSET_ROOT / "piper/urdf/piper_with_gripper_description.xacro")
        frames, indices = extract(video, folder / "frames")
        manifest["source_frame_indices"] = indices
        if vggt is None or sam3 is None:
            configured_vggt, configured_sam3, selection = backends()
            vggt = vggt or configured_vggt
            sam3 = sam3 or configured_sam3
            manifest["models"] = selection
        else:
            manifest["models"] = {"vggt": {"backend": "injected"}, "sam3": {"backend": "injected"}}
        arrays = vggt.infer(frames)
        manifest["models"]["vggt"].update(backend_manifest(vggt, "vggt"))
        manifest["models"]["vggt"]["response"] = getattr(vggt, "metadata", {})
        release_backend(vggt)
        np.savez_compressed(folder / "raw_vggt.npz", **arrays)
        depth, intrinsics, extrinsics = validate(arrays, metric_scale if metric_scale is not None else 1.0,
                                                metric_range=metric_scale is not None)
        np.savez_compressed(folder / "reconstruction.npz", depth=depth, intrinsics=intrinsics,
                            extrinsics=extrinsics, depth_conf=arrays["depth_conf"])
        masks = [np.asarray(sam3.segment(frame), dtype=bool) for frame in frames]
        manifest["models"]["sam3"].update(backend_manifest(sam3, "sam3"))
        manifest["models"]["sam3"]["response"] = getattr(sam3, "metadata", {})
        release_backend(sam3)
        if any(mask.shape != (288, 512) for mask in masks):
            raise ValueError("SAM3 mask does not align with RGB/depth")
        np.savez_compressed(folder / "robot_masks.npz", masks=np.stack(masks))
        mesh_rows, model_points = visual_meshes(folder / "meshes")
        pose, report, projections = register(model_points, depth, intrinsics, extrinsics,
                                               masks, {key: value for key, value in manifest["thresholds"].items() if key != "version"})
        report["threshold_version"] = manifest["thresholds"]["version"]
        report["metric_scale_verified"] = metric_scale is not None
        report["depth_unit"] = "meters" if metric_scale is not None else "unverified VGGT units"
        report["accepted"] = report["accepted"] and metric_scale is not None
        report["status"] = "passed" if report["accepted"] else "failed"
        if metric_scale is None:
            report["rejection_reason"] = "VGGT monocular reconstruction has no independently calibrated metric scale"
        save_json(folder / "registration.json", report)
        for slot, index in enumerate(REFERENCE):
            prefix = folder / f"view_{slot:02d}"
            (prefix.with_suffix(".jpg")).write_bytes(frames[index].read_bytes())
            np.save(prefix.with_suffix(".depth.npy"), depth[index])
            np.save(prefix.with_suffix(".cloud.npy"), pointcloud(depth[index], intrinsics[index], extrinsics[index]))
            overlay = cv2.imread(str(frames[index]))
            overlay[projections[slot]] = (0.5 * overlay[projections[slot]] + np.array([0, 0, 127])).astype(np.uint8)
            cv2.imwrite(str(prefix.with_suffix(".projection.png")), overlay)
        spec = {"width": 512, "height": 288, "intrinsics": intrinsics[list(REFERENCE)].tolist(),
                "extrinsics": extrinsics[list(REFERENCE)].tolist(), "world_from_base": pose.tolist(), "meshes": mesh_rows}
        save_json(folder / "scene_spec.json", spec)
        executable = resolve_blender(blender)
        process = subprocess.run([str(executable), "-b", "-noaudio", "--python", str(Path(__file__).with_name("blender_scene.py")),
                                  "--", str(folder / "scene_spec.json"), str(folder / "initial.blend")],
                                 capture_output=True, text=True, timeout=300)
        (folder / "blender.log").write_text(process.stdout + "\n" + process.stderr)
        if (process.returncode or "Traceback (most recent call last)" in process.stderr or
                not (folder / "initial.blend").is_file() or
                any(not image.is_file() for image in [folder / "initial.smoke.png"] +
                    [folder / f"view_{index:02d}.smoke.png" for index in range(1, 4)])):
            raise RuntimeError(f"Blender scene validation failed: exit code {process.returncode}; see blender.log")
        manifest["status"] = report["status"]
        manifest["artifacts"] = {"scene": "initial.blend", "registration": "registration.json",
                                  "reconstruction": "reconstruction.npz", "masks": "robot_masks.npz",
                                  "scene_spec": "scene_spec.json", "held_out_views": list(REFERENCE),
                                  "blender_smoke": ["initial.smoke.png"] +
                                                   [f"view_{index:02d}.smoke.png" for index in range(1, 4)]}
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if not (folder / "registration.json").exists():
            save_json(folder / "registration.json", {"status": "failed", "accepted": False,
                                                     "reason": str(exc), "thresholds": manifest["thresholds"]})
        (folder / "error.log").write_text(traceback.format_exc())
    finally:
        release_backend(vggt)
        release_backend(sam3)
        save_json(folder / "manifest.json", manifest)
    return folder, manifest["status"] == "passed"
