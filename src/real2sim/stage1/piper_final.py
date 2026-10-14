"""Run Stage 1 registration from the best of 128 scored Piper seeds."""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .franka_final import refine_scene, write_gallery
from .pipeline import sha256
from .sim3_registration import ROOT


def refine_task(scene: str, initializations: Path, processed: Path, output: Path,
                seed_rank: int = 1, target: Path | None = None) -> dict:
    model_root = ROOT / "asset/robot_surfaces/piper"
    models = {}
    for size in (5000, 10000, 40000):
        with np.load(model_root / f"surface_{size}.npz") as data:
            models[size] = data["points"].astype(float)
    full_model = models.pop(40000)
    models[20000] = full_model[np.random.default_rng(2027).permutation(len(full_model))[:20000]]
    return refine_scene(scene, processed, initializations, target or output / scene,
                        models, full_model, robot="piper", seed_rank=seed_rank)


def batch(initializations: Path, processed: Path, output: Path, *, only: str | None = None,
          workers: int = 4, progress=print) -> list[dict]:
    scenes = sorted(path.name for path in initializations.iterdir()
                    if path.is_dir() and (path / "initialization.json").is_file())
    raw = ROOT / "data/raw/piper"
    scenes = [scene for scene in scenes if (raw / f"{scene}.mp4").is_file()
              and (processed / "piper/stage1" / scene / "manifest.json").is_file()
              and sha256(raw / f"{scene}.mp4") == json.loads(
                  (processed / "piper/stage1" / scene / "manifest.json").read_text())["video_sha256"]]
    if only:
        if only not in scenes:
            raise ValueError(f"Piper scene is missing or stale: {only}")
        scenes = [only]
    if not scenes or workers < 1:
        raise ValueError("requires active Piper scenes and at least one worker")
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for scene in scenes:
        report_path = output / scene / "result.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            initialization = initializations / scene / "initialization.json"
            if (report.get("method") == "piper_128_top1_sim3_rotation_v1" and
                    report.get("initialization_sha256") == sha256(initialization) and
                    report.get("sam3_masks_sha256") and
                    report["sam3_masks_sha256"] == json.loads(
                        (processed / "piper/stage1" / scene / "manifest.json").read_text()).get("sam3_masks_sha256") and
                    report.get("status") == "candidate_unverified"):
                results[scene] = report
                progress(f"{scene}: reuse validated cached optimization")
    pending = [scene for scene in scenes if scene not in results]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(refine_task, scene, initializations, processed, output): scene
                   for scene in pending}
        for future in as_completed(futures):
            scene = futures[future]
            try:
                results[scene] = future.result()
            except Exception as exc:
                target = output / scene
                target.mkdir(parents=True, exist_ok=True)
                (target / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                results[scene] = {"scene": scene, "status": "failed", "error": str(exc)}
            progress(f"{scene}: {results[scene]['status']}")
            ordered = [results[name] for name in scenes if name in results]
            (output / "index.json").write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
            write_gallery(ordered, output, "piper")
    return [results[scene] for scene in scenes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine top-1 of 128 Piper poses with multiscale Sim(3)")
    parser.add_argument("--initializations", type=Path, default=ROOT / "runs/piper_initial_128_review/piper")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/piper_stage1_final_128")
    parser.add_argument("--only")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    results = batch(args.initializations, args.processed, args.output,
                    only=args.only, workers=args.workers)
    if any(result["status"] == "failed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
