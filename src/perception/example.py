"""Run local Stage 1 perception on one supplied video in a new output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .config import (backend_manifest, backend_settings, create_sam3, create_vggt, dependency_versions,
                     release_backend, require_device)


def run_example(video: Path, output: Path, robot: str = "piper") -> dict:
    from real2sim.stage1.inference_batch import infer_video
    from real2sim.stage1.pipeline import sha256

    video = video.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Example output already exists; choose a new directory: {output}")
    settings = {name: backend_settings(name) for name in ("vggt", "sam3")}
    if any(value["backend"] != "local" for value in settings.values()):
        raise ValueError("The local example requires VGGT_OMEGA_BACKEND=local and SAM3_BACKEND=local")
    devices = {value["device"] for value in settings.values()}
    import torch
    for device in devices:
        require_device(torch, device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
    services = {"vggt": create_vggt(), "sam3": create_sam3()}
    started = time.monotonic()
    try:
        selection = {name: backend_manifest(service, name) for name, service in services.items()}
        manifest = infer_video(video, output, robot, sha256(video), services["vggt"],
                               services["sam3"], selection, progress=lambda message: None)
        if manifest["status"] == "inference_complete":
            previews = output / "previews"
            previews.mkdir()
            for index in manifest["reference_indices"]:
                rgb = cv2.imread(str(output / "frames" / f"{index:02d}.jpg"))
                mask = np.load(output / "sam3_masks" / f"mask_{index:02d}.npy", allow_pickle=False)
                rgb[mask] = (rgb[mask] * 0.5 + np.array([0, 127, 0])).astype(np.uint8)
                cv2.imwrite(str(previews / f"mask_{index:02d}.png"), rgb)
                depth = np.load(output / "depth" / f"depth_{index:02d}.npy", allow_pickle=False)
                finite = np.isfinite(depth) & (depth > 0)
                visual = np.zeros(depth.shape, dtype=np.uint8)
                if finite.any():
                    low, high = np.percentile(depth[finite], [2, 98])
                    visual[finite] = np.clip((depth[finite] - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
                cv2.imwrite(str(previews / f"depth_{index:02d}.png"), cv2.applyColorMap(visual, cv2.COLORMAP_TURBO))
        metrics = {device: {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                   for device in devices if device.startswith("cuda")}
        report = {"status": manifest["status"], "manifest": str(output / "manifest.json"),
                  "elapsed_seconds": round(time.monotonic() - started, 3), "cuda_memory": metrics,
                  "models": manifest["models"], "torch_version": torch.__version__,
                  "cuda_version": torch.version.cuda, "dependency_versions": dependency_versions(),
                  "metric_scale_verified": False}
        if manifest.get("error"):
            report["error"] = manifest["error"]
        (output / "example_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        for service in services.values():
            release_backend(service)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="a readable video with at least 32 frames")
    parser.add_argument("--output", type=Path, required=True, help="a new, unused output directory")
    parser.add_argument("--robot", choices=("piper", "franka"), default="piper")
    args = parser.parse_args(argv)
    report = run_example(args.video, args.output, args.robot)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "inference_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
