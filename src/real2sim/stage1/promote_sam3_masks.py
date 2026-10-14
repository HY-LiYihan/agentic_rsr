"""Promote a verified Piper SAM3 prompt trial without repeating VGGT inference."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from perception.sam3 import DEFAULT_PROMPT
from .pipeline import sha256
from .sim3_registration import ROOT


def mask_digest(folder: Path) -> str:
    digest = hashlib.sha256()
    for index in range(32):
        mask_path = folder / f"mask_{index:02d}.npy"
        mask = np.load(mask_path, allow_pickle=False)
        if mask.shape != (288, 512) or mask.dtype != np.bool_ or not mask.any():
            raise ValueError(f"missing, empty or invalid SAM3 mask: {mask_path}")
        digest.update(mask_path.read_bytes())
    return digest.hexdigest()


def promote_scene(source: Path, trial: Path) -> dict:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads((trial / "report.json").read_text(encoding="utf-8"))
    video = Path(manifest["source_video"])
    if (manifest.get("status") != "inference_complete" or manifest.get("robot") != "piper"
            or not video.is_file() or sha256(video) != manifest.get("video_sha256")
            or report.get("status") != "complete" or report.get("prompt") != DEFAULT_PROMPT
            or report.get("video_sha256") != manifest["video_sha256"]
            or len(report.get("per_frame", {})) != 32):
        raise ValueError(f"incomplete or stale Piper SAM3 trial: {source.name}")
    for index in range(32):
        frame = source / "frames" / f"{index:02d}.jpg"
        depth = source / "depth" / f"depth_{index:02d}.npy"
        if not frame.is_file() or not depth.is_file():
            raise ValueError(f"missing VGGT/frame input for {source.name}/{index:02d}")
        if report["per_frame"][f"{index:02d}"].get("instance_count", 0) < 1:
            raise ValueError(f"SAM3 returned no instances for {source.name}/{index:02d}")
    digest = mask_digest(trial / "masks")
    masks = source / "sam3_masks"
    if manifest.get("sam3_prompt") == DEFAULT_PROMPT and mask_digest(masks) == digest:
        return manifest

    with tempfile.TemporaryDirectory(prefix=".sam3-promote-", dir=source) as temporary:
        pending = Path(temporary) / "masks"
        shutil.copytree(trial / "masks", pending)
        if mask_digest(pending) != digest:
            raise ValueError("mask copy changed during promotion")
        previous = Path(temporary) / "previous"
        os.replace(masks, previous)
        try:
            os.replace(pending, masks)
            manifest["sam3_prompt"] = DEFAULT_PROMPT
            manifest["sam3_masks_sha256"] = digest
            manifest["models"]["sam3"]["response"] = {
                "prompts": [DEFAULT_PROMPT], "conf_thresh": 0.25,
                "per_frame_instance_counts": [report["per_frame"][f"{index:02d}"]["instance_count"]
                                              for index in range(32)]}
            manifest["empty_mask_frames"] = []
            manifest.pop("stage1_delivery", None)
            staged_manifest = Path(temporary) / "manifest.json"
            staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            os.replace(staged_manifest, manifest_path)
        except Exception:
            if masks.exists():
                shutil.rmtree(masks)
            os.replace(previous, masks)
            raise
    for name in ("registration.json", "stage1_delivery.json", "initial.blend",
                 "camera_metric.json", "quality_check.png"):
        (source / name).unlink(missing_ok=True)
    for name in ("depth_metric",):
        shutil.rmtree(source / name, ignore_errors=True)
    for index in (0, 10, 21, 31):
        (source / "reference" / f"overlay_{index:02d}.png").unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote complete SAM3 trials, preserving VGGT outputs")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed/piper/stage1")
    parser.add_argument("--trials", type=Path, default=ROOT / "runs/sam3_prompt_trials/piper/base_camera")
    args = parser.parse_args()
    scenes = sorted(path.stem for path in (ROOT / "data/raw/piper").glob("*.mp4"))
    if not scenes:
        parser.error("no active Piper videos")
    for scene in scenes:
        manifest = promote_scene(args.processed / scene, args.trials / scene)
        print(f"{scene}: {manifest['sam3_prompt']}", flush=True)


if __name__ == "__main__":
    main()
