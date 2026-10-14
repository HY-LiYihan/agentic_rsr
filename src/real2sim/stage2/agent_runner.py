"""Run a bounded Codex-controlled Blender Stage 2 refinement loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from real2sim.blender import resolve_blender
from .evaluator import DEFAULT_REFERENCE_INDICES, evaluate_bundle


REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_FILE = Path(__file__).with_name("agent_prompt.md")
BLENDER_WORKER = Path(__file__).with_name("blender_worker.py")
CODEX_MODEL = os.getenv("REAL2SIM_CODEX_MODEL", "gpt-6-astra")
CODEX_REASONING_EFFORT = "high"
RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "changed_objects": {"type": "array", "items": {"type": "string"}},
        "changed_categories": {"type": "array", "items": {"type": "string"}},
        "next_hypothesis": {"type": "string"},
    },
    "required": ["summary", "changed_objects", "changed_categories", "next_hypothesis"],
    "additionalProperties": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    """Keep a local, JSON-safe copy of SDK conversation objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _command_trace(items: list[Any]) -> list[dict[str, Any]]:
    trace = []
    for item in items:
        payload = getattr(item, "root", item)
        if getattr(payload, "type", None) != "commandExecution":
            continue
        trace.append({
            "command": getattr(payload, "command", None),
            "status": str(getattr(payload, "status", "")),
            "exit_code": getattr(payload, "exit_code", None),
            "aggregated_output": getattr(payload, "aggregated_output", None),
        })
    return trace


def _turn_record(turn: Any, prompt: str) -> dict[str, Any]:
    return {
        "turn_id": turn.id,
        "status": str(turn.status),
        "prompt": prompt,
        "final_response": turn.final_response,
        "commands": _command_trace(turn.items),
        "items": [_jsonable(item) for item in turn.items],
        "usage": _jsonable(getattr(turn, "usage", None)),
    }


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _blender_path(value: str | None) -> Path:
    return resolve_blender(value)


def _run_blender(blender: Path, scene: Path, command: str, output: Path,
                 args: list[str] | None = None) -> dict[str, Any]:
    command_args = [str(blender), "-b", str(scene), "--python-exit-code", "3",
                    "--python", str(BLENDER_WORKER), "--",
                    command, "--output", str(output)] + (args or [])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.run(command_args, cwd=REPO_ROOT, env=environment,
                             capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"Blender {command} failed ({process.returncode}): "
                           f"{process.stderr[-3000:]}\n{process.stdout[-1000:]}")
    return {"command": command_args, "stdout": process.stdout[-4000:],
            "stderr": process.stderr[-4000:]}


def _prepare_initial_scene(blender: Path, source: Path, target: Path) -> dict[str, Any]:
    if source.resolve() == target.resolve():
        raise ValueError("Stage 2 prepared scene must be separate from the Stage 1 source")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return {"source": str(source), "target": str(target), "reused": True}
    result = _run_blender(blender, source, "add-default-lighting", target)
    return {"source": str(source), "target": str(target), "reused": False,
            "command": result["command"]}


def _load_stage1(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "stage1_delivery.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing Stage 1 delivery manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indices = tuple(int(value) for value in manifest.get("reference_indices", DEFAULT_REFERENCE_INDICES))
    if len(indices) != 4:
        raise ValueError("Stage 1 manifest must contain four reference_indices")
    for path in (input_dir / "initial.blend", input_dir / "camera_metric.json"):
        if not path.is_file():
            raise FileNotFoundError(f"missing Stage 1 artifact: {path}")
    return {"manifest": manifest, "reference_indices": indices,
            "initial_scene": input_dir / "initial.blend"}


def _make_prompt(task_description: str, input_dir: Path, current_scene: Path,
                 run_dir: Path, frozen: Path, report: Path, edit_script: Path,
                 max_rounds: int) -> str:
    return PROMPT_FILE.read_text(encoding="utf-8").format(
        task_description=task_description, input_dir=input_dir, current_scene=current_scene,
        run_dir=run_dir, frozen_signature=frozen, evaluation_report=report,
        edit_script=edit_script, max_rounds=max_rounds)


def _save_best(source: Path, run_dir: Path, report: dict[str, Any]) -> Path:
    target = run_dir / "best" / "scene.blend"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    _write_json(run_dir / "best" / "evaluation.json", report)
    return target


def run_stage2(input_dir: Path, output: Path, task_description: str | None = None,
               max_rounds: int = 10, blender: str | None = None,
               allow_unverified_stage1: bool = False) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use an empty Stage 2 output directory: {output}")
    if input_dir == output or input_dir.is_relative_to(output):
        raise ValueError("Stage 2 output must not contain the Stage 1 input directory")
    if max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative")
    stage1 = _load_stage1(input_dir)
    manifest = stage1["manifest"]
    if not manifest.get("accepted", False) and not allow_unverified_stage1:
        raise ValueError("Stage 1 manifest is not accepted; pass --allow-unverified-stage1 "
                         "after independent human review")
    blender_path = _blender_path(blender)
    task = task_description or str(manifest.get("task_description", input_dir.name.replace("_", " ")))
    output.mkdir(parents=True, exist_ok=True)
    baseline = output / "baseline"
    baseline.mkdir()
    baseline_scene = baseline / "initial_with_default_light.blend"
    prepared_scene = _prepare_initial_scene(blender_path, stage1["initial_scene"], baseline_scene)
    inspect_path = baseline / "inspect.json"
    _run_blender(blender_path, baseline_scene, "inspect", inspect_path)
    inspect = json.loads(inspect_path.read_text(encoding="utf-8"))
    frozen_path = output / "frozen_signature.json"
    _write_json(frozen_path, inspect["frozen_signature"])
    render_dir = baseline / "render"
    _run_blender(blender_path, baseline_scene, "render", render_dir)
    baseline_report = evaluate_bundle(input_dir, render_dir, baseline / "evaluation")
    current_scene = _save_best(baseline_scene, output, baseline_report)
    record: dict[str, Any] = {
        "schema": "stage2-agent-run-v1",
        "status": "passed" if baseline_report["passed"] else "running",
        "input_dir": str(input_dir),
        "initial_scene": str(stage1["initial_scene"]),
        "initial_scene_sha256": _sha256(stage1["initial_scene"]),
        "prepared_initial_scene": str(baseline_scene),
        "prepared_initial_scene_sha256": _sha256(baseline_scene),
        "default_lighting": prepared_scene,
        "task_description": task,
        "reference_indices": list(stage1["reference_indices"]),
        "max_rounds": max_rounds,
        "blender": str(blender_path),
        "rounds": [],
    }
    _write_json(output / "session.json", record)
    if baseline_report["passed"] or max_rounds == 0:
        record["status"] = "passed" if baseline_report["passed"] else "failed_baseline"
        record["final_scene"] = str(current_scene) if baseline_report["passed"] else None
        _write_json(output / "session.json", record)
        return record

    from openai_codex import ApprovalMode, Codex, Sandbox, TextInput

    with Codex() as codex:
        thread = codex.thread_start(cwd=str(REPO_ROOT), sandbox=Sandbox.full_access,
                                    approval_mode=ApprovalMode.auto_review,
                                    model=CODEX_MODEL, ephemeral=True)
        record["thread_id"] = thread.id
        record["thread_ephemeral"] = True
        conversation_path = output / "conversation.jsonl"
        try:
            for round_id in range(1, max_rounds + 1):
                round_dir = output / f"round_{round_id:03d}"
                round_dir.mkdir()
                script_path = round_dir / "edit_scene.py"
                prompt = _make_prompt(task, input_dir, current_scene, output, frozen_path,
                                      output / "best" / "evaluation.json", script_path, max_rounds)
                (round_dir / "prompt.md").write_text(prompt, encoding="utf-8")
                try:
                    turn = thread.run([TextInput(text=prompt)], model=CODEX_MODEL,
                                      effort=CODEX_REASONING_EFFORT, output_schema=RESULT_SCHEMA)
                    turn_record = _turn_record(turn, prompt)
                    _append_jsonl(conversation_path, turn_record)
                    record.setdefault("turns", []).append({
                        "round": round_id, "turn_id": turn.id, "status": str(turn.status),
                        "final_response": turn.final_response,
                        "commands": turn_record["commands"],
                    })
                    agent_note = json.loads(turn.final_response or "{}")
                except Exception as exc:
                    round_record = {"round": round_id, "status": "agent_error", "error": repr(exc)}
                    _write_json(round_dir / "round_report.json", round_record)
                    record["rounds"].append(round_record)
                    break
                if not script_path.is_file():
                    round_record = {"round": round_id, "status": "missing_edit_script",
                                    "agent_note": agent_note}
                    _write_json(round_dir / "round_report.json", round_record)
                    record["rounds"].append(round_record)
                    break
                candidate = round_dir / "candidate.blend"
                try:
                    _run_blender(blender_path, current_scene, "apply", candidate,
                                 ["--base", str(current_scene), "--script", str(script_path),
                                  "--frozen", str(frozen_path)])
                    _run_blender(blender_path, candidate, "freeze", round_dir / "freeze.json",
                                 ["--frozen", str(frozen_path)])
                    render = round_dir / "render"
                    _run_blender(blender_path, candidate, "render", render)
                    report = evaluate_bundle(input_dir, render, round_dir / "evaluation")
                    round_record = {"round": round_id, "status": "passed" if report["passed"] else "evaluated",
                                    "candidate_scene": str(candidate), "evaluation": report,
                                    "agent_note": agent_note}
                    current_scene = candidate
                    record["rounds"].append(round_record)
                    _write_json(round_dir / "round_report.json", round_record)
                    if report["passed"]:
                        final = output / "final" / "scene.blend"
                        final.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(candidate, final)
                        _write_json(output / "final" / "stage2_manifest.json", {
                            "schema": "stage2-delivery-v1", "status": "passed", "accepted": True,
                            "scene": str(final), "scene_sha256": _sha256(final),
                            "evaluation": report, "frozen_signature": json.loads(frozen_path.read_text()),
                            "thread_id": thread.id, "round": round_id,
                        })
                        record["status"] = "passed"
                        record["final_scene"] = str(final)
                        break
                except Exception as exc:
                    round_record = {"round": round_id, "status": "candidate_rejected", "error": repr(exc),
                                    "agent_note": agent_note}
                    _write_json(round_dir / "round_report.json", round_record)
                    record["rounds"].append(round_record)
            else:
                record["status"] = "max_rounds_exceeded"
        finally:
            try:
                archive = codex.thread_archive(thread.id)
                record["thread_archived"] = True
                record["thread_archive"] = _jsonable(archive)
            except Exception as exc:
                record["thread_archived"] = False
                record["thread_archive_error"] = repr(exc)
                record["thread_cleanup"] = "ephemeral_thread_not_persisted"
            else:
                record["thread_cleanup"] = "archived"
            _write_json(output / "session.json", record)
    _write_json(output / "session.json", record)
    return record


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--phase" in argv:
        from .phase_runner import main as phase_main
        return phase_main(argv)
    if argv and argv[0].startswith("--"):
        argv = ["agent", *argv]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "render", "render-rgb"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--scene", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--blender", help="Blender executable path or command; defaults to BLENDER_BIN or PATH")
        subparser.set_defaults(command_name=command)
    depth_evaluate = subparsers.add_parser("evaluate-depth")
    depth_evaluate.add_argument("--reference", type=Path, required=True)
    depth_evaluate.add_argument("--render", type=Path, required=True)
    depth_evaluate.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--reference", type=Path, required=True)
    evaluate.add_argument("--render", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    agent = subparsers.add_parser("agent")
    agent.add_argument("--input", type=Path, required=True)
    agent.add_argument("--output", type=Path, required=True)
    agent.add_argument("--task")
    agent.add_argument("--max-rounds", type=int, default=10)
    agent.add_argument("--blender")
    agent.add_argument("--allow-unverified-stage1", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        report = evaluate_bundle(args.reference, args.render, args.output)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 2
    if args.command == "evaluate-depth":
        from .evaluator import evaluate_depth_bundle
        report = evaluate_depth_bundle(args.reference, args.render, args.output)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 2
    if args.command == "agent":
        from .phase_runner import run_phases
        result = run_phases(args.input, args.output, args.task, args.max_rounds, args.blender)
        print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                          "final_scene": result.get("geometry", {}).get("final_scene")},
                         indent=2, ensure_ascii=False))
        return 0 if result["status"] in {"passed", "failed_baseline"} else 1
    blender = _blender_path(args.blender)
    _run_blender(blender, args.scene.expanduser().resolve(strict=True), args.command,
                 args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
