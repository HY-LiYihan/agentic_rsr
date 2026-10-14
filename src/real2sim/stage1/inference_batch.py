"""Serial VGGT-Ω/SAM3 extraction; registration is deliberately out of scope."""

import argparse
import json
import shutil
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
from perception.sam3 import DEFAULT_PROMPT
from perception.config import backend_manifest, release_backend

from real2sim.stage1.pipeline import backends, extract, save_json, sha256, validate
from real2sim.stage1.registration import REFERENCE
from real2sim.stage1.promote_sam3_masks import mask_digest


ROOT = Path(__file__).resolve().parents[3]
GROUPS = {"frank": "franka", "piper": "piper"}


def complete(directory: Path, video_hash: str) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (FileNotFoundError, ValueError):
        return False
    if (manifest.get("schema") != "stage1-inference-v1" or
            manifest.get("status") != "inference_complete" or
            manifest.get("video_sha256") != video_hash or
            manifest.get("sam3_prompt") != DEFAULT_PROMPT):
        return False
    groups = (("frames", "jpg", ""), ("depth", "npy", "depth_"),
              ("sam3_masks", "npy", "mask_"))
    for folder, extension, prefix in groups:
        if not all((directory / folder / f"{prefix}{index:02d}.{extension}").is_file()
                   for index in range(32)):
            return False
    if not ((directory / "camera.json").is_file() and
            (directory / "depth_conf.npz").is_file() and
            all((directory / "reference" / f"frame_{slot:02d}.jpg").is_file()
                for slot in range(4))):
        return False
    if manifest.get("robot") == "piper":
        try:
            return bool(manifest.get("sam3_masks_sha256")) and mask_digest(
                directory / "sam3_masks") == manifest["sam3_masks_sha256"]
        except (OSError, ValueError):
            return False
    return True


def infer_once(service, method: str, argument):
    for attempt in range(3):
        try:
            return getattr(service, method)(argument)
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def infer_video(video: Path, directory: Path, group: str, video_hash: str,
                vggt, sam3, backends_info: dict, *, progress=print) -> dict:
    directory.mkdir(parents=True)
    manifest = {"schema": "stage1-inference-v1", "status": "running", "robot": group,
                "registration": "not_started", "source_video": str(video.resolve()),
                "video_sha256": video_hash, "task_description": video.stem.replace("_", " "),
                "reference_indices": list(REFERENCE), "depth_unit": "unverified VGGT units",
                "sam3_prompt": DEFAULT_PROMPT, "models": {name: info.copy() for name, info in backends_info.items()}}
    save_json(directory / "manifest.json", manifest)
    try:
        frames, indices = extract(video, directory / "frames")
        manifest["source_frame_indices"] = indices
        progress(f"    VGGT-Ω: 32 frames")
        arrays = infer_once(vggt, "infer", frames)
        manifest["models"]["vggt"].update(backend_manifest(vggt, "vggt"))
        release_backend(vggt)
        depth, intrinsics, extrinsics = validate(arrays, 1.0, metric_range=False)
        manifest["models"]["vggt"]["response"] = getattr(vggt, "metadata", {})
        save_json(directory / "camera.json", {"intrinsics": intrinsics.tolist(),
                                              "extrinsics": extrinsics.tolist(),
                                              "extrinsics_direction": "world_to_camera",
                                              "camera_coordinate_system": "OpenCV +X right +Y down +Z forward",
                                              "depth_unit": "unverified VGGT units",
                                              "source_frame_indices": indices})
        np.savez_compressed(directory / "depth_conf.npz", depth_conf=arrays["depth_conf"])
        for name in ("depth", "sam3_masks", "reference"):
            (directory / name).mkdir()
        for index, image in enumerate(depth):
            np.save(directory / "depth" / f"depth_{index:02d}.npy", image)
        for slot, index in enumerate(REFERENCE):
            shutil.copyfile(frames[index], directory / "reference" / f"frame_{slot:02d}.jpg")
        manifest["vggt_status"] = "complete"
        save_json(directory / "manifest.json", manifest)
        progress(f"    SAM3: 32 masks ({DEFAULT_PROMPT})")
        empty_frames = []
        for index, frame in enumerate(frames):
            mask = np.asarray(infer_once(sam3, "segment", frame), dtype=bool)
            if mask.shape != depth[index].shape:
                raise ValueError(f"SAM3 mask {index} does not align with RGB and depth")
            if not mask.any():
                empty_frames.append(index)
            np.save(directory / "sam3_masks" / f"mask_{index:02d}.npy", mask)
        manifest["models"]["sam3"]["response"] = getattr(sam3, "metadata", {})
        manifest["models"]["sam3"].update(backend_manifest(sam3, "sam3"))
        release_backend(sam3)
        if group == "piper" and not empty_frames:
            manifest["sam3_masks_sha256"] = mask_digest(directory / "sam3_masks")
        manifest["empty_mask_frames"] = empty_frames
        manifest["status"] = "inference_complete"
        manifest["artifacts"] = {"frames": "frames/", "depth": "depth/",
                                 "sam3_masks": "sam3_masks/",
                                 "reference": "reference/", "cameras": "camera.json", "depth_conf": "depth_conf.npz"}
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        (directory / "error.log").write_text(traceback.format_exc())
    finally:
        release_backend(vggt)
        release_backend(sam3)
        save_json(directory / "manifest.json", manifest)
    return manifest


def batch(raw_root: Path = ROOT / "data/raw", processed_root: Path = ROOT / "data/processed",
          *, selected: str | None = None, force: bool = False, vggt=None, sam3=None,
          progress=print) -> list[dict]:
    videos = [(robot, video) for source, robot in GROUPS.items()
              for video in sorted((raw_root / source).glob("*.mp4"))
              if selected is None or video.stem == selected or video.name == selected]
    if not videos:
        raise ValueError("no matching videos found in data/raw/frank or data/raw/piper")
    if vggt is None or sam3 is None:
        remote_vggt, remote_sam3, selection = backends()
        vggt = vggt or remote_vggt
        sam3 = sam3 or remote_sam3
    else:
        selection = {"vggt": {"backend": "injected"}, "sam3": {"backend": "injected"}}
    results = []
    for position, (robot, video) in enumerate(videos, 1):
        parent = processed_root / robot / "stage1"
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / video.stem
        video_hash = sha256(video)
        if not force and complete(target, video_hash):
            progress(f"[{position}/{len(videos)}] {robot}/{video.name}: existing inference complete")
            manifest = json.loads((target / "manifest.json").read_text())
        else:
            progress(f"[{position}/{len(videos)}] {robot}/{video.name}")
            staging = parent / f".pending-{video.stem}-{uuid.uuid4().hex[:8]}"
            manifest = infer_video(video, staging, robot, video_hash, vggt, sam3, selection, progress=progress)
            if target.exists():
                history = parent / ".history"
                history.mkdir(exist_ok=True)
                target.rename(history / f"{video.stem}-{uuid.uuid4().hex[:8]}")
            staging.rename(target)
        progress(f"    status={manifest['status']}, empty_masks={len(manifest.get('empty_mask_frames', []))}")
        results.append({"robot": robot, "video": video.name, "status": manifest["status"],
                        "output": str(target.resolve()), "error": manifest.get("error")})
        save_json(processed_root / "stage1_batch_summary.json", {"results": results,
                                                                 "registration": "not_started"})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Serial 32-frame VGGT-Ω and SAM3 inference (no registration)")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--only", help="Process a single video by filename or stem")
    parser.add_argument("--force", action="store_true", help="Re-run complete videos, archiving previous results")
    args = parser.parse_args()
    results = batch(args.raw_root, args.processed_root, selected=args.only, force=args.force)
    if any(result["status"] != "inference_complete" for result in results):
        parser.exit(1, "Some videos failed; inspect per-video manifest.json and stage1_batch_summary.json\n")


if __name__ == "__main__":
    main()
