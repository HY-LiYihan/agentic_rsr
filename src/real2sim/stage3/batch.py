"""Compile and audit every accepted Stage 2 delivery, one scene at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_runner import run_agent
from .mujoco_export import EXPORTER_VERSION, ExportConfig, _stage2_delivery, export_blend, validate_xml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def accepted_tasks(processed: Path, robots: list[str]) -> list[Path]:
    tasks = []
    for robot in robots:
        for task in sorted((processed / robot / "stage2").glob("*/geometry/stage2_manifest.json")):
            manifest = json.loads(task.read_text(encoding="utf-8"))
            if manifest.get("accepted") is True and manifest.get("evaluation", {}).get("passed") is True:
                tasks.append(task.parent.parent)
    return tasks


def _existing_scene(output: Path, delivery: dict[str, Any]) -> Path | None:
    manifest_path = output / "export_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene = output / "scene.xml"
    if (manifest.get("status") != "passed" or manifest.get("exporter_version") != EXPORTER_VERSION or not scene.is_file() or
            manifest.get("stage2_scene_sha256") != hashlib.sha256(delivery["scene"].read_bytes()).hexdigest()):
        return None
    validate_xml(scene, delivery["robot"])
    return scene


def run_task(task: Path, processed: Path, run_id: str, run_physics: bool = True,
             runs_root: Path | None = None, semantics_dir: Path | None = None,
             reuse_semantics_from: Path | None = None) -> dict[str, Any]:
    delivery = _stage2_delivery(task)
    root = runs_root or REPO_ROOT / "runs/stage3/workspace"
    robot_root = root / delivery["robot"]
    if root.resolve() == processed.resolve():
        robot_root /= "stage3"
    target = robot_root / task.name
    previous_candidates = sorted(
        [candidate for candidate in target.glob("mujoco_v*")
         if candidate.is_dir() and any(candidate.iterdir())],
        key=lambda candidate: candidate.name)
    previous = previous_candidates[-1] if previous_candidates else (
        target / "mujoco" if (target / "mujoco").is_dir() and any((target / "mujoco").iterdir()) else None)
    legacy = previous is not None
    output = target / f"mujoco_v{EXPORTER_VERSION}" if legacy else target / "mujoco"
    report: dict[str, Any] = {"robot": delivery["robot"], "task": task.name,
                              "stage2": str(task), "output": str(target)}
    try:
        scene = _existing_scene(output, delivery)
        if scene is None:
            if output.exists() and any(output.iterdir()):
                session = output / "semantics_session.json"
                failure = ((session.is_file() and
                            json.loads(session.read_text(encoding="utf-8")).get("status") == "failed") or
                           (not (output / "export_manifest.json").exists() and
                            (output / "semantics_inventory.json").exists()))
                if not failure:
                    raise FileExistsError(f"incomplete or stale export; review before retrying: {output}")
                archive = output.with_name(f"{output.name}_failed_{run_id}")
                if archive.exists():
                    raise FileExistsError(f"previous failed export archive exists: {archive}")
                os.rename(output, archive)
                report["previous_failure"] = str(archive)
            semantics = (semantics_dir / delivery["robot"] / f"{task.name}.json"
                         if semantics_dir is not None else
                         previous / "semantics.json" if previous and (previous / "semantics.json").is_file() else None)
            if semantics_dir is not None and not semantics.is_file():
                raise FileNotFoundError(f"reviewed semantics missing: {semantics}")
            if semantics is None and reuse_semantics_from is not None:
                prior = reuse_semantics_from / delivery["robot"] / task.name
                source_hash = hashlib.sha256(delivery["scene"].read_bytes()).hexdigest()
                for candidate in sorted(prior.glob("mujoco*/semantics.json"), reverse=True):
                    try:
                        if json.loads(candidate.read_text(encoding="utf-8")).get("source_sha256") == source_hash:
                            semantics = candidate
                            break
                    except (OSError, ValueError):
                        continue
            manifest = export_blend(ExportConfig(None, output, stage2_delivery=task,
                                                 semantics_path=semantics))
            scene = Path(manifest["scene"])
            report["export"] = "compiled"
            if semantics:
                report["reused_semantics"] = str(semantics)
        else:
            report["export"] = "reused"
        report["scene"] = str(scene)
        semantics = json.loads((output / "semantics.json").read_text(encoding="utf-8"))
        report["roles"] = {role: sum(item["role"] == role for item in semantics["objects"])
                           for role in ("visual_only", "static", "dynamic", "operable")}
        if run_physics:
            physics = target / "physics" / run_id
            physics.mkdir(parents=True, exist_ok=False)
            reference_views = "\n".join(f"camera_{index}: {path.resolve()}"
                                         for index, path in enumerate(delivery["rgb_paths"]))
            context = (f"Task: {delivery['task']}\nRobot: {delivery['robot']}\n\n"
                       f"Stage 2 reference RGB views (original accepted pose):\n{reference_views}\n\n"
                       f"Stage 2 scene plan:\n{delivery['plan'].read_text(encoding='utf-8')}")
            context_path = target / "task_context.md"
            context_path.write_text(context, encoding="utf-8")
            objects = json.loads((output / "object_manifest.json").read_text(encoding="utf-8"))["objects"]
            movable = [item["name"] for item in objects
                       if item["semantics"]["joint_type"] in {"free", "hinge", "slide"}]
            hypotheses_path = target / "hypotheses.json"
            _write_json(hypotheses_path, {
                "source": str(delivery["manifest"]), "task_description": delivery["task"],
                "real_world_measurements": [], "hypotheses": [{
                    "objects": movable,
                "expectation": "Task objects should remain supported at rest; the Piper-aligned gripper probe has a 10 N per-jaw actuator limit.",
                    "origin": "Stage 2 visual plan and scene geometry, not a measured physical property",
                }],
            })
            result = run_agent(scene, hypotheses_path, physics, task_context=context)
            report["physics"] = str(physics)
            report["verification"] = result.get("verification")
            verification = result.get("verification", {})
            candidate = verification.get("candidate_recheck")
            report["physics_status"] = "provisional" if (
                verification.get("passed") and
                ((candidate and candidate.get("static_passed") and candidate.get("grasp_passed")) or
                 (verification.get("sim_static_passed") and
                  (verification.get("sim_grasp_passed") or verification.get("sim_articulation_passed"))))
            ) else "needs_repair"
            if report["physics_status"] == "provisional":
                report["final_scene"] = result.get("agent_result", {}).get("candidate_scene") or str(scene)
            if result.get("error"):
                report["agent_error"] = result["error"]
            report["status"] = "audited" if report["physics_status"] == "provisional" else "needs_review"
        else:
            report["status"] = "exported"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _can_resume(task: Path, processed: Path, previous: dict[str, Any]) -> bool:
    if previous.get("status") != "audited":
        return False
    verification = previous.get("verification") or {}
    candidate = verification.get("candidate_recheck") or {}
    if not (verification.get("passed") and
            ((candidate.get("static_passed") and candidate.get("grasp_passed")) or
             (verification.get("sim_static_passed") and
              (verification.get("sim_grasp_passed") or verification.get("sim_articulation_passed"))))):
        return False
    scene = Path(previous.get("scene", ""))
    manifest = scene.parent / "export_manifest.json"
    if not scene.is_file() or not manifest.is_file():
        return False
    delivery = _stage2_delivery(task)
    export = json.loads(manifest.read_text(encoding="utf-8"))
    return (export.get("exporter_version") == EXPORTER_VERSION and
            export.get("stage2_scene_sha256") == hashlib.sha256(delivery["scene"].read_bytes()).hexdigest())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=REPO_ROOT / "data/processed")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs/stage3/workspace")
    parser.add_argument("--semantics-dir", type=Path,
                        help="reuse SHA256-matched reviewed classifications from ROBOT/TASK.json")
    parser.add_argument("--reuse-semantics-from", type=Path,
                        help="find SHA256-matched reviewed semantics in prior ROBOT/TASK/mujoco* runs")
    parser.add_argument("--robot", action="append", choices=("piper", "franka"))
    parser.add_argument("--task", action="append", help="limit to task names for a trial run")
    parser.add_argument("--export-only", action="store_true", help="skip physical agent")
    parser.add_argument("--resume-from", type=Path, help="skip previously verified, physically passing tasks")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--workers", type=int, default=1, help="number of independent scene processes")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    processed = args.processed.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve()
    semantics_dir = args.semantics_dir.expanduser().resolve(strict=True) if args.semantics_dir else None
    reuse_semantics_from = (args.reuse_semantics_from.expanduser().resolve(strict=True)
                            if args.reuse_semantics_from else None)
    selected = accepted_tasks(processed, args.robot or ["piper", "franka"])
    if args.task:
        selected = [task for task in selected if task.name in args.task]
    if not selected:
        parser.error("no accepted Stage 2 deliveries match the selection")
    log_root = processed / "stage3" if runs_root == processed else REPO_ROOT / "runs/stage3"
    log = log_root / f"batch_{args.run_id}" / "batch_manifest.json"
    if log.exists():
        parser.error(f"batch run already exists: {log}")
    summary: dict[str, Any] = {"run_id": args.run_id, "total": len(selected),
                               "workers": args.workers, "tasks": []}
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume_from:
        prior = json.loads(args.resume_from.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
        completed = {(item["robot"], item["task"]): item for item in prior["tasks"]}
        summary["resumed_from"] = str(args.resume_from.resolve())
    def record(result: dict[str, Any]) -> None:
        summary["tasks"].append(result)
        _write_json(log, summary)
        print(f"[{len(summary['tasks'])}/{len(selected)}] {result['robot']}/{result['task']}: "
              f"{result['status']} {result.get('error') or result.get('agent_error') or result.get('roles')}", flush=True)

    pending = []
    for task in selected:
        previous = completed.get((task.parent.parent.name, task.name))
        if not args.export_only and previous and _can_resume(task, processed, previous):
            record({**previous, "resumed": True})
        else:
            pending.append(task)
    if args.workers == 1:
        for task in pending:
            print(f"Starting {task.parent.parent.name}/{task.name}", flush=True)
            record(run_task(task, processed, args.run_id, not args.export_only, runs_root,
                            semantics_dir, reuse_semantics_from))
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_task, task, processed, args.run_id,
                                   not args.export_only, runs_root, semantics_dir,
                                   reuse_semantics_from): task for task in pending}
            for task in pending:
                print(f"Queued {task.parent.parent.name}/{task.name}", flush=True)
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"robot": task.parent.parent.name, "task": task.name,
                              "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                record(result)
    print(f"Batch report: {log}", flush=True)
    return 1 if any(item["status"] in {"failed", "needs_review"} for item in summary["tasks"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
