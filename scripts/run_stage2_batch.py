#!/usr/bin/env python3
"""Run Stage 2 for all selected Franka/Piper Stage 1 deliveries."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    robot: str
    task_name: str
    task_description: str
    input_dir: Path
    output_dir: Path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _task_description(task_name: str, manifest: dict[str, Any]) -> str:
    value = manifest.get("task_description")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return task_name.replace("_", " ")


def _discover_jobs(data_root: Path, robots: list[str], output_root: Path | None = None) -> list[Job]:
    jobs: list[Job] = []
    for robot in robots:
        stage1_root = data_root / robot / "stage1"
        stage2_root = ((output_root / robot) if output_root is not None
                       else data_root / robot / "stage2")
        if not stage1_root.is_dir():
            raise FileNotFoundError(f"missing Stage 1 directory: {stage1_root}")
        for input_dir in sorted(path for path in stage1_root.iterdir() if path.is_dir()):
            manifest_path = input_dir / "stage1_delivery.json"
            scene_path = input_dir / "initial.blend"
            if not manifest_path.is_file() or not scene_path.is_file():
                raise FileNotFoundError(f"incomplete Stage 1 delivery: {input_dir}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            jobs.append(Job(
                robot=robot,
                task_name=input_dir.name,
                task_description=_task_description(input_dir.name, manifest),
                input_dir=input_dir,
                output_dir=stage2_root / input_dir.name,
            ))
    return jobs


def _run_job(job: Job, max_rounds: int, blender: str | None, model: str,
             resume: bool = False) -> dict[str, Any]:
    started = time.time()
    continue_existing = resume and (job.output_dir / "plan" / "plan.md").is_file()
    command = [
        sys.executable, "-m", "real2sim.stage2.agent_runner",
        "--phase", "geometry" if continue_existing else "all",
        "--input", str(job.input_dir),
        "--output", str(job.output_dir),
        "--task", job.task_description,
        "--max-rounds", str(max_rounds),
    ]
    if blender:
        command.extend(["--blender", blender])
    if continue_existing:
        command.extend(["--plan", str(job.output_dir / "plan" / "plan.md"), "--resume"])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["REAL2SIM_CODEX_MODEL"] = model
    process = subprocess.run(command, cwd=REPO_ROOT, env=environment,
                             capture_output=True, text=True)
    log_path = job.output_dir / ("resume_process.log" if resume else "batch_process.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND:\n" + " ".join(command) + "\n\nSTDOUT:\n" + process.stdout +
        "\nSTDERR:\n" + process.stderr,
        encoding="utf-8",
    )
    status = "failed_process" if process.returncode else "completed"
    reported_status = None
    try:
        payload = json.loads(process.stdout)
        reported_status = payload.get("status")
        if process.returncode == 0:
            status = reported_status or status
    except json.JSONDecodeError:
        payload = None
    return {
        "robot": job.robot,
        "task": job.task_name,
        "task_description": job.task_description,
        "input_dir": str(job.input_dir),
        "output_dir": str(job.output_dir),
        "status": status,
        "reported_status": reported_status,
        "model": model,
        "returncode": process.returncode,
        "elapsed_seconds": round(time.time() - started, 2),
        "log": str(log_path),
        "final_scene": (
            (payload or {}).get("final_scene") or
            ((payload or {}).get("geometry") or {}).get("final_scene")
        ) if isinstance(payload, dict) else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path,
                        help="independent root for outputs; defaults to data-root/<robot>/stage2")
    parser.add_argument("--robots", nargs="+", choices=("franka", "piper"),
                        default=["franka", "piper"])
    parser.add_argument("--workers", type=int, default=2,
                        help="maximum concurrent Stage 2 processes")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--model", default=os.getenv("REAL2SIM_CODEX_MODEL", "gpt-6-astra"),
                        help="Codex model passed to every Stage 2 agent session")
    parser.add_argument("--resume", action="store_true",
                        help="continue unfinished outputs without rerunning Plan or completed rounds")
    parser.add_argument("--blender")
    parser.add_argument("--manifest", type=Path,
                        help="batch status file; defaults to runs/stage2/metadata/batch_manifest.json")
    parser.add_argument("--metadata-root", type=Path,
                        default=REPO_ROOT / "runs/stage2/metadata",
                        help="directory for batch metadata and manifests")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_rounds < 0:
        parser.error("--max-rounds must be nonnegative")

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    jobs = _discover_jobs(data_root, args.robots, output_root)
    if args.resume:
        jobs = [job for job in jobs
                if not (job.output_dir / "geometry" / "stage2_manifest.json").exists()]
    else:
        occupied = [job.output_dir for job in jobs if job.output_dir.exists() and any(job.output_dir.iterdir())]
        if occupied:
            names = ", ".join(str(path) for path in occupied)
            raise FileExistsError(f"refusing to overwrite existing Stage 2 outputs: {names}")
    if args.dry_run:
        for job in jobs:
            print(f"{job.robot}\t{job.task_name}\t{job.output_dir}")
        return 0

    batch_root = args.metadata_root.expanduser().resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else batch_root / "batch_manifest.json"
    record: dict[str, Any] = {
        "schema": "stage2-batch-manifest-v1",
        "status": "running",
        "data_root": str(data_root),
        "robots": args.robots,
        "workers": args.workers,
        "max_rounds": args.max_rounds,
        "model": args.model,
        "output_root": str(output_root) if output_root else None,
        "jobs": [],
    }
    _write_json(manifest_path, record)
    lock = threading.Lock()
    print(f"Starting {len(jobs)} Stage 2 jobs with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_job, job, args.max_rounds, args.blender,
                                   args.model, args.resume): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                record["jobs"].append(result)
                record["jobs"].sort(key=lambda item: (item["robot"], item["task"]))
                _write_json(manifest_path, record)
            print(f"[{result['status']}] {result['robot']}/{result['task']}", flush=True)
    record["status"] = "passed" if all(item["status"] == "passed" for item in record["jobs"]) else "completed_with_failures"
    _write_json(manifest_path, record)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
