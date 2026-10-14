"""Run the staged Stage 2 plan and geometry refinement workflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from . import evaluator
from .evaluator import (
    DEFAULT_REFERENCE_INDICES, DEPTH_MAE_THRESHOLD_M, LAB_DE76_THRESHOLD,
    evaluate_bundle, evaluate_depth_bundle,
)
from .agent_runner import (
    BLENDER_WORKER,
    CODEX_MODEL,
    CODEX_REASONING_EFFORT,
    REPO_ROOT,
    _append_jsonl,
    _blender_path,
    _command_trace,
    _jsonable,
    _prepare_initial_scene,
    _run_blender,
    _sha256,
    _turn_record,
    _write_json,
)


PLAN_PROMPT_FILE = Path(__file__).with_name("plan_prompt.md")
REFINEMENT_PROMPT_FILE = Path(__file__).with_name("refinement_prompt.md")
ROBOT_ROOT_NAMES = {"franka": "FR3", "piper": "Piper"}

GEOMETRY_RESULT_SCHEMA = {
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


def _load_delivery(input_dir: Path) -> tuple[dict[str, Any], tuple[int, ...]]:
    manifest_path = input_dir / "stage1_delivery.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing Stage 1 delivery manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indices = tuple(int(value) for value in manifest.get("reference_indices", DEFAULT_REFERENCE_INDICES))
    if len(indices) != 4:
        raise ValueError("Stage 1 manifest must contain exactly four reference_indices")
    return manifest, indices


def _robot_root_name(input_dir: Path) -> str:
    try:
        return ROBOT_ROOT_NAMES[input_dir.parent.parent.name]
    except KeyError as exc:
        raise ValueError(f"unsupported Stage 1 robot: {input_dir.parent.parent.name}") from exc


def _reference_rgb_paths(input_dir: Path) -> list[Path]:
    paths = [input_dir / "reference" / f"frame_{ordinal:02d}.jpg" for ordinal in range(4)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing reference RGB images: " + ", ".join(missing))
    return paths


def _format_prompt(path: Path, **values: Any) -> str:
    return path.read_text(encoding="utf-8").format(**values)


def _archive_thread(codex: Any, thread: Any, record: dict[str, Any]) -> None:
    try:
        archive = codex.thread_archive(thread.id)
        record["thread_archived"] = True
        record["thread_archive"] = _jsonable(archive)
        record["thread_cleanup"] = "archived"
    except Exception as exc:
        record["thread_archived"] = False
        record["thread_archive_error"] = repr(exc)
        record["thread_cleanup"] = "ephemeral_thread_not_persisted"


def _validate_plan(plan_path: Path, task: str) -> dict[str, Any]:
    if not plan_path.is_file():
        raise FileNotFoundError(f"Plan Agent did not create {plan_path}")
    text = plan_path.read_text(encoding="utf-8")
    required_headings = (
        "## Object hierarchy",
        "## Approximate dimensions",
        "## Required capabilities",
        "## Relative layout",
        "## Necessary assumptions",
    )
    missing = [heading for heading in required_headings if heading not in text]
    if missing:
        raise ValueError(f"plan.md is missing required headings: {missing}")
    lower = text.lower()
    forbidden_capability_terms = ("piper", "fr3", "franka", "robot arm", "robotic arm", "gripper")
    capabilities_start = lower.index("## required capabilities")
    content_start = capabilities_start + len("## required capabilities")
    next_heading = re.search(r"^##\s", lower[content_start:], re.MULTILINE)
    capabilities_end = (content_start + next_heading.start()
                        if next_heading else len(text))
    capabilities = lower[capabilities_start:capabilities_end]
    if any(term in capabilities for term in forbidden_capability_terms):
        raise ValueError("Required capabilities must describe task objects, not the robot")
    dimensions_start = lower.index("## approximate dimensions")
    dimensions_end = lower.find("## required capabilities", dimensions_start)
    dimensions = text[dimensions_start:dimensions_end if dimensions_end >= 0 else len(text)]
    dimension_units = (" m", "cm", "mm", "米", "厘米", "毫米")
    if not any(unit in dimensions.lower() for unit in dimension_units):
        raise ValueError("Approximate dimensions must include units for object size estimates")
    numeric_values = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", dimensions)
    if len(numeric_values) < 3:
        raise ValueError("Approximate dimensions must provide length, width and height values")
    if not re.search(r"(?:×|\bx\b|\bX\b)", dimensions):
        raise ValueError("Approximate dimensions must state length, width and height")
    if task.lower() not in lower and not any(token in lower for token in task.lower().split() if len(token) > 3):
        raise ValueError("plan.md does not explain the requested task")
    return {
        "path": str(plan_path),
        "sha256": _sha256(plan_path),
        "headings": list(required_headings),
        "dimensions_section_bytes": len(dimensions.encode("utf-8")),
        "bytes": len(text.encode("utf-8")),
    }


def run_plan(input_dir: Path, output: Path, task_description: str | None = None) -> dict[str, Any]:
    """Ask a fresh Codex thread for a task-only plan from four RGB references."""
    input_dir = input_dir.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    manifest, indices = _load_delivery(input_dir)
    task = task_description or str(manifest.get("task_description", input_dir.name.replace("_", " ")))
    rgb_paths = _reference_rgb_paths(input_dir)
    plan_dir = output / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / "plan.md"
    prompt = _format_prompt(
        PLAN_PROMPT_FILE,
        task_description=task,
        robot_name=_robot_root_name(input_dir),
        input_dir=input_dir,
        plan_path=plan_path,
    )
    (plan_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    record: dict[str, Any] = {
        "schema": "stage2-plan-run-v1",
        "status": "running",
        "input_dir": str(input_dir),
        "task_description": task,
        "reference_indices": list(indices),
        "reference_rgb": [str(path) for path in rgb_paths],
        "plan_path": str(plan_path),
        "thread_id": None,
        "model": CODEX_MODEL,
        "reasoning_effort": CODEX_REASONING_EFFORT,
        "sandbox": "full-access",
        "approval_mode": "auto_review",
    }
    _write_json(plan_dir / "session.json", record)
    from openai_codex import ApprovalMode, Codex, LocalImageInput, Sandbox, TextInput

    try:
        with Codex() as codex:
            thread = codex.thread_start(
                cwd=str(REPO_ROOT),
                sandbox=Sandbox.full_access,
                approval_mode=ApprovalMode.auto_review,
                model=CODEX_MODEL,
                ephemeral=True,
            )
            record["thread_id"] = thread.id
            inputs = [TextInput(text=prompt)] + [LocalImageInput(path=str(path)) for path in rgb_paths]
            turn = thread.run(inputs, model=CODEX_MODEL, effort=CODEX_REASONING_EFFORT)
            turn_record = _turn_record(turn, prompt)
            _append_jsonl(plan_dir / "conversation.jsonl", turn_record)
            record["turn_id"] = turn.id
            record["turn_status"] = str(turn.status)
            record["final_response"] = turn.final_response
            record["commands"] = _command_trace(turn.items)
            validation = _validate_plan(plan_path, task)
            record["validation"] = validation
            record["status"] = "passed"
            return record
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    finally:
        if "codex" in locals() and "thread" in locals():
            _archive_thread(codex, thread, record)
        _write_json(plan_dir / "session.json", record)


def _rgb_inputs(reference_paths: list[Path], render_dir: Path) -> list[Path]:
    paths = list(reference_paths) + [render_dir / f"camera_{ordinal}_rgb.png" for ordinal in range(4)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing RGB inputs: " + ", ".join(missing))
    return paths


def _make_geometry_prompt(task: str, current_scene: Path, round_dir: Path,
                          reference_paths: list[Path], current_rgb_dir: Path,
                          metric_path: Path, script_path: Path, max_rounds: int,
                          initial_scene: Path, plan_path: Path, plan_text: str,
                          camera_status: str, robot_name: str) -> str:
    prompt = _format_prompt(
        Path(os.environ.get("REAL2SIM_REFINEMENT_PROMPT", str(REFINEMENT_PROMPT_FILE))),
        task_description=task,
        robot_name=robot_name,
        initial_scene=initial_scene,
        current_scene=current_scene,
        round_dir=round_dir,
        plan_path=plan_path,
        plan_text=plan_text,
        camera_status=camera_status,
        reference_rgb_paths="\n".join(f"- {path}" for path in reference_paths),
        current_rgb_paths="\n".join(
            f"- {current_rgb_dir / f'camera_{ordinal}_rgb.png'}" for ordinal in range(4)
        ),
        repo_root=REPO_ROOT,
        metric_path=metric_path,
        edit_script=script_path,
        max_rounds=max_rounds,
    )
    if os.environ.get("REAL2SIM_HIDE_METRIC_FEEDBACK") == "1":
        start = prompt.index("The only numeric feedback provided")
        end = prompt.index("## Required optimization order", start)
        prompt = prompt[:start] + (
            "No numerical evaluation feedback is supplied in this experiment. "
            "Optimize using the four reference RGBs and your rendered RGBs only. "
            "The external evaluator still checks depth MAE < 0.15 m and Lab < 15 "
            "and stops the run automatically on success, with at most 10 rounds. "
            "Do not read or calculate evaluation metrics, agent_metric.json, "
            "session.json, round_report.json, evaluation directories, baseline "
            "reports, depth arrays, masks, camera JSON, or other runs. "
            "Do not invoke the evaluator yourself. Render and inspect RGB only.\n\n"
        ) + prompt[end:]
        prompt = prompt.replace("latest scalar metric", "visual comparison")
    return prompt


def _write_agent_metric(path: Path, evaluation: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("REAL2SIM_HIDE_METRIC_FEEDBACK") == "1":
        path.unlink(missing_ok=True)
        return {}
    metric = {
        "schema": "stage2-agent-final-scalar-metric-v1",
        "depth_mae_mean_m": float(evaluation["depth_mae_mean_m"]),
        "acceptable_depth_mae_mean_m": float(evaluation["thresholds"]["depth_mae_mean_m"]),
        "lab_delta_e76_mean": float(evaluation["lab_delta_e76_mean"]),
        "acceptable_lab_delta_e76_mean": float(evaluation["thresholds"]["lab_delta_e76_mean"]),
        "passed": bool(evaluation["passed"]),
        "instruction": "Scalar final metrics only. No depth arrays, depth images, depth error maps, or per-pixel metric maps are provided to the Agent.",
    }
    _write_json(path, metric)
    return metric


def _agent_rgb_render_complete(round_dir: Path) -> bool:
    render_dir = round_dir / "agent_self_render"
    return all(
        (render_dir / f"camera_{ordinal}_rgb.png").is_file()
        and (render_dir / f"camera_{ordinal}_rgb.png").stat().st_size > 0
        for ordinal in range(4)
    )


def _camera_status(inspect: dict[str, Any]) -> str:
    objects = {item["name"]: item for item in inspect.get("objects", [])}
    lines = []
    for scene in inspect.get("scenes", []):
        camera = objects.get(scene.get("camera"), {})
        lines.append(
            f"- scene={scene['name']}; camera={scene.get('camera')}; "
            f"resolution={scene.get('resolution')}; "
            f"location={camera.get('location')}; rotation={camera.get('rotation')}"
        )
    return "\n".join(lines)


def _run_geometry_legacy(input_dir: Path, output: Path, task_description: str | None = None,
                         max_rounds: int = 10, blender: str | None = None) -> dict[str, Any]:
    """Refine geometry using depth-only observations; the runner owns evaluation and saves."""
    input_dir = input_dir.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    manifest, indices = _load_delivery(input_dir)
    initial_scene = input_dir / "initial.blend"
    if not initial_scene.is_file():
        raise FileNotFoundError(f"missing Stage 1 scene: {initial_scene}")
    if max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative")
    task = task_description or str(manifest.get("task_description", input_dir.name.replace("_", " ")))
    blender_path = _blender_path(blender)
    reference_rgb_paths = _reference_rgb_paths(input_dir)
    geometry_dir = output / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    input_scene = geometry_dir / "initial_with_default_light.blend"
    lighting_record = _prepare_initial_scene(blender_path, initial_scene, input_scene)
    inspect_path = geometry_dir / "inspect.json"
    if not inspect_path.exists():
        _run_blender(blender_path, input_scene, "inspect", inspect_path)
    inspect = json.loads(inspect_path.read_text(encoding="utf-8"))
    frozen_path = geometry_dir / "frozen_signature.json"
    _write_json(frozen_path, inspect["frozen_signature"])
    baseline_render = geometry_dir / "baseline_render"
    if not (baseline_render / "render_manifest.json").is_file():
        _run_blender(blender_path, input_scene, "render", baseline_render)
    baseline_depth = geometry_dir / "baseline_depth"
    baseline_report_path = baseline_depth / "report.json"
    if baseline_report_path.is_file():
        baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    else:
        baseline_report = evaluate_depth_bundle(input_dir, baseline_render, baseline_depth, indices)
    current_scene = input_scene
    record: dict[str, Any] = {
        "schema": "stage2-geometry-run-v1",
        "status": "running",
        "input_dir": str(input_dir),
        "task_description": task,
        "reference_indices": list(indices),
        "initial_scene": str(initial_scene),
        "prepared_initial_scene": str(input_scene),
        "prepared_initial_scene_sha256": _sha256(input_scene),
        "default_lighting": lighting_record,
        "input_scene": str(input_scene),
        "blender": str(blender_path),
        "max_rounds": max_rounds,
        "baseline": baseline_report,
        "rounds": [],
        "thread_id": None,
        "model": CODEX_MODEL,
        "reasoning_effort": CODEX_REASONING_EFFORT,
        "sandbox": "full-access",
        "approval_mode": "auto_review",
    }
    _write_json(geometry_dir / "session.json", record)
    if baseline_report["passed"] or max_rounds == 0:
        record["status"] = "passed" if baseline_report["passed"] else "failed_baseline"
        record["final_scene"] = str(current_scene) if baseline_report["passed"] else None
        _write_json(geometry_dir / "session.json", record)
        return record

    from openai_codex import ApprovalMode, Codex, LocalImageInput, Sandbox, TextInput

    try:
        with Codex() as codex:
            thread = codex.thread_start(
                cwd=str(REPO_ROOT),
                sandbox=Sandbox.full_access,
                approval_mode=ApprovalMode.auto_review,
                model=CODEX_MODEL,
                ephemeral=True,
            )
            record["thread_id"] = thread.id
            conversation_path = geometry_dir / "conversation.jsonl"
            for round_id in range(1, max_rounds + 1):
                round_dir = geometry_dir / f"round_{round_id:03d}"
                round_dir.mkdir(parents=True, exist_ok=True)
                script_path = round_dir / "edit_scene.py"
                render_dir = round_dir / "render"
                eval_dir = round_dir / "evaluation"
                if round_id == 1:
                    observed_report = baseline_report
                else:
                    observed_report = json.loads(
                        (geometry_dir / f"round_{round_id - 1:03d}" / "evaluation" / "report.json").read_text(
                            encoding="utf-8"
                        )
                    )
                current_rgb_dir = round_dir / "current_rgb"
                _run_blender(blender_path, current_scene, "render-rgb", current_rgb_dir)
                metric_path = round_dir / "agent_metric.json"
                _write_agent_metric(metric_path, observed_report)
                prompt = _make_geometry_prompt(
                    task, current_scene, round_dir, reference_rgb_paths, current_rgb_dir,
                    metric_path, script_path, max_rounds,
                )
                (round_dir / "prompt.md").write_text(prompt, encoding="utf-8")
                image_inputs = _rgb_inputs(reference_rgb_paths, current_rgb_dir)
                try:
                    turn = thread.run(
                        [TextInput(text=prompt)] + [LocalImageInput(path=str(path)) for path in image_inputs],
                        model=CODEX_MODEL,
                        effort=CODEX_REASONING_EFFORT,
                        output_schema=GEOMETRY_RESULT_SCHEMA,
                    )
                    turn_record = _turn_record(turn, prompt)
                    _append_jsonl(conversation_path, turn_record)
                    agent_note = json.loads(turn.final_response or "{}")
                except Exception as exc:
                    round_record = {"round": round_id, "status": "agent_error", "error": repr(exc)}
                    _write_json(round_dir / "round_report.json", round_record)
                    record["rounds"].append(round_record)
                    break
                agent_commands = turn_record["commands"]
                self_rendered = any(
                    "render-rgb" in str(item.get("command", "")) and
                    "agent_self_render" in str(item.get("command", ""))
                    for item in agent_commands
                )
                if not self_rendered:
                    round_record = {
                        "round": round_id,
                        "status": "missing_agent_rgb_render",
                        "agent_note": agent_note,
                        "commands": agent_commands,
                        "message": "Agent must render current RGB views into agent_self_render before editing.",
                    }
                    _write_json(round_dir / "round_report.json", round_record)
                    record["rounds"].append(round_record)
                    break
                if not script_path.is_file():
                    round_record = {"round": round_id, "status": "missing_edit_script", "agent_note": agent_note}
                    _write_json(round_dir / "round_report.json", round_record)
                    record["rounds"].append(round_record)
                    break
                candidate = round_dir / "candidate.blend"
                try:
                    _run_blender(blender_path, current_scene, "apply", candidate, [
                        "--base", str(current_scene), "--script", str(script_path),
                        "--frozen", str(frozen_path),
                    ])
                    _run_blender(blender_path, candidate, "freeze", round_dir / "freeze.json",
                                 ["--frozen", str(frozen_path)])
                    _run_blender(blender_path, candidate, "render", render_dir)
                    report = evaluate_depth_bundle(input_dir, render_dir, eval_dir, indices)
                    round_record = {
                        "round": round_id,
                        "status": "passed" if report["passed"] else "evaluated",
                        "candidate_scene": str(candidate),
                        "render": str(render_dir),
                        "evaluation": report,
                        "agent_note": agent_note,
                    }
                    current_scene = candidate
                    record["rounds"].append(round_record)
                    _write_json(round_dir / "round_report.json", round_record)
                    if report["passed"]:
                        final = geometry_dir / "final.blend"
                        shutil.copy2(candidate, final)
                        record["status"] = "passed"
                        record["final_scene"] = str(final)
                        record["final_evaluation"] = report
                        break
                except Exception as exc:
                    round_record = {"round": round_id, "status": "candidate_rejected",
                                    "error": repr(exc), "agent_note": agent_note}
                    _write_json(round_dir / "round_report.json", round_record)
                    record["rounds"].append(round_record)
            else:
                record["status"] = "max_rounds_exceeded"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if "codex" in locals() and "thread" in locals():
            _archive_thread(codex, thread, record)
        _write_json(geometry_dir / "session.json", record)
    return record


def _current_threshold_report(report: dict[str, Any]) -> dict[str, Any]:
    thresholds = {**report["thresholds"],
                  "depth_mae_mean_m": DEPTH_MAE_THRESHOLD_M,
                  "lab_delta_e76_mean": LAB_DE76_THRESHOLD}
    return {
        **report,
        "thresholds": thresholds,
        "normalized_loss": 0.5 * report["depth_mae_mean_m"] / DEPTH_MAE_THRESHOLD_M +
                           0.5 * report["lab_delta_e76_mean"] / LAB_DE76_THRESHOLD,
        "passed": bool(report["depth_mae_mean_m"] < DEPTH_MAE_THRESHOLD_M and
                       report["lab_delta_e76_mean"] < LAB_DE76_THRESHOLD and
                       all(view["depth_passed"] and view["lab_passed"] for view in report["views"])),
    }


def _geometry_input_provenance(input_dir: Path, indices: tuple[int, ...],
                               plan: Path, task: str, blender: Path) -> dict[str, Any]:
    files = [input_dir / "stage1_delivery.json", input_dir / "initial.blend", plan,
             BLENDER_WORKER, Path(evaluator.__file__)]
    for ordinal, index in enumerate(indices):
        files.extend(evaluator._reference_paths(input_dir, index, ordinal))
    return {
        "schema": "stage2-refinement-inputs-v1",
        "input_dir": str(input_dir), "task_description": task,
        "blender": str(blender), "reference_indices": list(indices),
        "file_sha256": {str(path): _sha256(path) for path in files},
        "evaluation": {
            "depth_mae_mean_m": DEPTH_MAE_THRESHOLD_M,
            "lab_delta_e76_mean": LAB_DE76_THRESHOLD,
            "view_depth_limit_m": evaluator.VIEW_DEPTH_LIMIT_M,
            "view_lab_limit": evaluator.VIEW_LAB_LIMIT,
            "min_depth_coverage": evaluator.MIN_DEPTH_COVERAGE,
            "missing_depth_penalty_m": evaluator.MISSING_DEPTH_PENALTY_M,
            "max_valid_render_depth_m": evaluator.MAX_VALID_RENDER_DEPTH_M,
        },
    }


def _render_artifact_sha256(scene: Path, render: Path) -> dict[str, str]:
    files = [scene, render / "render_manifest.json"]
    files += [render / f"camera_{ordinal}_{suffix}" for ordinal in range(4)
              for suffix in ("rgb.png", "depth.npy")]
    return {str(path): _sha256(path) for path in files}


def _verify_cached_artifacts(expected: dict[str, str], scene: Path,
                             render: Path, inspect: Path | None = None) -> None:
    try:
        actual = _render_artifact_sha256(scene, render)
        if inspect is not None:
            actual[str(inspect)] = _sha256(inspect)
    except OSError as exc:
        raise ValueError("Stage 2 cache provenance is incomplete; use a new output directory") from exc
    if actual != expected:
        raise ValueError("Stage 2 cache provenance changed; use a new output directory")


def run_geometry(input_dir: Path, output: Path, task_description: str | None = None,
                 max_rounds: int = 10, blender: str | None = None,
                 plan: Path | None = None, resume: bool = False) -> dict[str, Any]:
    """Run one RGB-driven geometry/material refinement loop with scalar final metrics."""
    input_dir = input_dir.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    manifest, indices = _load_delivery(input_dir)
    initial_scene = input_dir / "initial.blend"
    if not initial_scene.is_file():
        raise FileNotFoundError(f"missing Stage 1 scene: {initial_scene}")
    if max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative")
    task = task_description or str(manifest.get("task_description", input_dir.name.replace("_", " ")))
    plan_path = (plan or (output / "plan" / "plan.md")).expanduser().resolve(strict=True)
    plan_text = plan_path.read_text(encoding="utf-8")
    blender_path = _blender_path(blender)
    reference_rgb_paths = _reference_rgb_paths(input_dir)
    geometry_dir = output / "geometry"
    provenance_path = geometry_dir / "input_provenance.json"
    provenance = _geometry_input_provenance(input_dir, indices, plan_path, task, blender_path)
    cached = geometry_dir.is_dir() and any(geometry_dir.iterdir())
    if cached:
        if not resume:
            raise FileExistsError(f"Stage 2 geometry already exists: {geometry_dir}; use --resume or a new output directory")
        if not provenance_path.is_file():
            raise ValueError("Stage 2 input provenance is missing; use a new output directory")
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if previous.get("inputs") != provenance:
            raise ValueError("Stage 2 input provenance changed; use a new output directory")
        _verify_cached_artifacts(previous.get("baseline_artifact_sha256", {}),
                                 geometry_dir / "initial_with_default_light.blend",
                                 geometry_dir / "baseline_render", geometry_dir / "inspect.json")
    geometry_dir.mkdir(parents=True, exist_ok=True)
    input_scene = geometry_dir / "initial_with_default_light.blend"
    lighting_record = _prepare_initial_scene(blender_path, initial_scene, input_scene)
    inspect_path = geometry_dir / "inspect.json"
    if not inspect_path.exists():
        _run_blender(blender_path, input_scene, "inspect", inspect_path)
    inspect = json.loads(inspect_path.read_text(encoding="utf-8"))
    frozen_path = geometry_dir / "frozen_signature.json"
    _write_json(frozen_path, inspect["frozen_signature"])
    baseline_render = geometry_dir / "baseline_render"
    if not (baseline_render / "render_manifest.json").is_file():
        _run_blender(blender_path, input_scene, "render", baseline_render)
    baseline_eval_dir = geometry_dir / "baseline_evaluation"
    # Cached metrics never authorize a delivery; evaluate the verified arrays again.
    baseline_report = evaluate_bundle(input_dir, baseline_render, baseline_eval_dir, indices)
    baseline_report = _current_threshold_report(baseline_report)
    if not cached:
        artifacts = _render_artifact_sha256(input_scene, baseline_render)
        artifacts[str(inspect_path)] = _sha256(inspect_path)
        _write_json(provenance_path, {"inputs": provenance, "baseline_artifact_sha256": artifacts})
    current_scene = input_scene
    completed_rounds: list[dict[str, Any]] = []
    start_round = 1
    accepted_round: Path | None = None
    if resume:
        for round_id in range(1, max_rounds + 1):
            round_dir = geometry_dir / f"round_{round_id:03d}"
            report_path = round_dir / "evaluation" / "report.json"
            candidate = round_dir / "candidate.blend"
            if not report_path.is_file() or not candidate.is_file():
                break
            round_record = json.loads((round_dir / "round_report.json").read_text(encoding="utf-8"))
            _verify_cached_artifacts(round_record.get("artifact_sha256", {}), candidate, round_dir / "render")
            report = evaluate_bundle(input_dir, round_dir / "render", round_dir / "evaluation", indices)
            round_record["evaluation"] = report
            round_record["status"] = "passed" if report["passed"] else "evaluated"
            completed_rounds.append(round_record)
            current_scene = candidate
            start_round = round_id + 1
            if report["passed"]:
                accepted_round = round_dir
        if (geometry_dir / "session.json").exists():
            shutil.copy2(geometry_dir / "session.json", geometry_dir / "session_before_resume.json")
    record: dict[str, Any] = {
        "schema": "stage2-refinement-run-v1",
        "status": "running",
        "input_dir": str(input_dir),
        "initial_scene": str(initial_scene),
        "prepared_initial_scene": str(input_scene),
        "prepared_initial_scene_sha256": _sha256(input_scene),
        "default_lighting": lighting_record,
        "input_scene": str(input_scene),
        "plan": str(plan_path),
        "task_description": task,
        "reference_indices": list(indices),
        "camera_status": _camera_status(inspect),
        "max_rounds": max_rounds,
        "blender": str(blender_path),
        "model": CODEX_MODEL,
        "reasoning_effort": CODEX_REASONING_EFFORT,
        "sandbox": "full-access",
        "approval_mode": "auto_review",
        "baseline": baseline_report,
        "rounds": completed_rounds,
        "thread_id": None,
    }
    _write_json(geometry_dir / "session.json", record)
    if accepted_round is not None:
        report = evaluate_bundle(input_dir, accepted_round / "render",
                                 geometry_dir / "resume_evaluation", indices)
        if not report["passed"]:
            raise ValueError(f"previous candidate failed current acceptance: {accepted_round}")
        final = geometry_dir / "final.blend"
        shutil.copy2(accepted_round / "candidate.blend", final)
        record.update(status="passed", final_scene=str(final), final_evaluation=report,
                      accepted_round=accepted_round.name)
        _write_json(geometry_dir / "stage2_manifest.json", {
            "schema": "stage2-refinement-delivery-v1", "accepted": True,
            "scene": str(final), "scene_sha256": _sha256(final),
            "evaluation": report, "plan": str(plan_path), "thread_id": None,
        })
        _write_json(geometry_dir / "session.json", record)
        return record
    if baseline_report["passed"] or max_rounds == 0:
        record["status"] = "passed" if baseline_report["passed"] else "failed_baseline"
        record["final_scene"] = None
        if baseline_report["passed"]:
            final = geometry_dir / "final.blend"
            shutil.copy2(input_scene, final)
            record.update(final_scene=str(final), final_evaluation=baseline_report)
            _write_json(geometry_dir / "stage2_manifest.json", {
                "schema": "stage2-refinement-delivery-v1", "accepted": True,
                "scene": str(final), "scene_sha256": _sha256(final),
                "evaluation": baseline_report, "plan": str(plan_path), "thread_id": None,
            })
        _write_json(geometry_dir / "session.json", record)
        return record
    if start_round > max_rounds:
        record["status"] = "max_rounds_exceeded"
        _write_json(geometry_dir / "session.json", record)
        return record

    from openai_codex import ApprovalMode, Codex, LocalImageInput, Sandbox, TextInput

    try:
        with Codex() as codex:
            thread = codex.thread_start(
                cwd=str(REPO_ROOT), sandbox=Sandbox.full_access,
                approval_mode=ApprovalMode.auto_review, model=CODEX_MODEL,
                ephemeral=True,
            )
            record["thread_id"] = thread.id
            conversation_path = geometry_dir / "conversation.jsonl"
            for round_id in range(start_round, max_rounds + 1):
                round_dir = geometry_dir / f"round_{round_id:03d}"
                round_dir.mkdir(parents=True, exist_ok=True)
                script_path = round_dir / "edit_scene.py"
                render_dir = round_dir / "render"
                eval_dir = round_dir / "evaluation"
                if round_id == 1:
                    observed_report = baseline_report
                else:
                    observed_report = _current_threshold_report(json.loads(
                        (geometry_dir / f"round_{round_id - 1:03d}" / "evaluation" / "report.json").read_text(
                            encoding="utf-8"
                        )
                    ))
                current_rgb_dir = round_dir / "current_rgb"
                _run_blender(blender_path, current_scene, "render-rgb", current_rgb_dir)
                metric_path = round_dir / "agent_metric.json"
                _write_agent_metric(metric_path, observed_report)
                prompt = _make_geometry_prompt(
                    task, current_scene, round_dir, reference_rgb_paths, current_rgb_dir,
                    metric_path, script_path, max_rounds, input_scene, plan_path,
                    plan_text, _camera_status(inspect), _robot_root_name(input_dir),
                )
                (round_dir / "prompt.md").write_text(prompt, encoding="utf-8")
                image_inputs = _rgb_inputs(reference_rgb_paths, current_rgb_dir)
                try:
                    turn = thread.run(
                        [TextInput(text=prompt)] + [LocalImageInput(path=str(path)) for path in image_inputs],
                        model=CODEX_MODEL, effort=CODEX_REASONING_EFFORT,
                        output_schema=GEOMETRY_RESULT_SCHEMA,
                    )
                    turn_record = _turn_record(turn, prompt)
                    _append_jsonl(conversation_path, turn_record)
                    agent_note = json.loads(turn.final_response or "{}")
                except Exception as exc:
                    round_record = {"round": round_id, "status": "agent_error", "error": repr(exc)}
                    _write_json(round_dir / "round_report.json", round_record)
                    record["rounds"].append(round_record)
                    break
                agent_commands = turn_record["commands"]
                if not _agent_rgb_render_complete(round_dir):
                    round_record = {
                        "round": round_id, "status": "missing_agent_rgb_render",
                        "agent_note": agent_note, "commands": agent_commands,
                        "render_dir": str(round_dir / "agent_self_render"),
                    }
                    _write_json(round_dir / "round_report.json", round_record)
                    record["rounds"].append(round_record)
                    break
                if not script_path.is_file():
                    round_record = {"round": round_id, "status": "missing_edit_script", "agent_note": agent_note}
                    _write_json(round_dir / "round_report.json", round_record)
                    record["rounds"].append(round_record)
                    break
                candidate = round_dir / "candidate.blend"
                try:
                    _run_blender(blender_path, current_scene, "apply", candidate, [
                        "--base", str(current_scene), "--script", str(script_path),
                        "--frozen", str(frozen_path),
                    ])
                    _run_blender(blender_path, candidate, "freeze", round_dir / "freeze.json",
                                 ["--frozen", str(frozen_path)])
                    _run_blender(blender_path, candidate, "render", render_dir)
                    report = evaluate_bundle(input_dir, render_dir, eval_dir, indices)
                    round_record = {
                        "round": round_id,
                        "status": "passed" if report["passed"] else "evaluated",
                        "candidate_scene": str(candidate), "render": str(render_dir),
                        "evaluation": report, "agent_note": agent_note,
                        "artifact_sha256": _render_artifact_sha256(candidate, render_dir),
                    }
                    current_scene = candidate
                    record["rounds"].append(round_record)
                    _write_json(round_dir / "round_report.json", round_record)
                    if report["passed"]:
                        final = geometry_dir / "final.blend"
                        shutil.copy2(candidate, final)
                        record["status"] = "passed"
                        record["final_scene"] = str(final)
                        record["final_evaluation"] = report
                        _write_json(geometry_dir / "stage2_manifest.json", {
                            "schema": "stage2-refinement-delivery-v1",
                            "accepted": True, "scene": str(final),
                            "scene_sha256": _sha256(final), "evaluation": report,
                            "plan": str(plan_path), "thread_id": thread.id,
                        })
                        break
                except Exception as exc:
                    round_record = {"round": round_id, "status": "candidate_rejected",
                                    "error": repr(exc), "agent_note": agent_note}
                    _write_json(round_dir / "round_report.json", round_record)
                    record["rounds"].append(round_record)
            else:
                record["status"] = "max_rounds_exceeded"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if "codex" in locals() and "thread" in locals():
            _archive_thread(codex, thread, record)
        _write_json(geometry_dir / "session.json", record)
    return record


def _write_run_manifest(output: Path, result: dict[str, Any]) -> None:
    plan = result.get("plan") or {}
    geometry = result.get("geometry") or {}
    final_scene = geometry.get("final_scene")
    _write_json(output / "run_manifest.json", {
        "schema": "stage2-run-manifest-v1",
        "status": result.get("status"),
        "input_dir": plan.get("input_dir") or geometry.get("input_dir"),
        "task_description": plan.get("task_description") or geometry.get("task_description"),
        "plan": {
            "status": plan.get("status"),
            "document": plan.get("plan_path") or geometry.get("plan") or
                        str(output / "plan" / "plan.md"),
            "session": str(output / "plan" / "session.json"),
            "conversation": str(output / "plan" / "conversation.jsonl"),
        },
        "geometry": {
            "status": geometry.get("status"),
            "session": str(output / "geometry" / "session.json"),
            "final_scene": final_scene,
            "delivery_manifest": str(output / "geometry" / "stage2_manifest.json"),
        },
        "final_scene": final_scene,
    })


def run_phases(input_dir: Path, output: Path, task_description: str | None = None,
               max_rounds: int = 10, blender: str | None = None) -> dict[str, Any]:
    output = output.expanduser().resolve()
    geometry_dir = output / "geometry"
    if geometry_dir.is_dir() and any(geometry_dir.iterdir()):
        raise FileExistsError(f"Stage 2 geometry already exists: {geometry_dir}; "
                              "use --phase geometry --resume or a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    plan = run_plan(input_dir, output, task_description)
    if plan["status"] != "passed":
        result = {"status": "plan_failed", "plan": plan}
        _write_run_manifest(output, result)
        return result
    geometry = run_geometry(input_dir, output, task_description, max_rounds, blender,
                            output / "plan" / "plan.md")
    result = {"status": geometry["status"], "plan": plan, "geometry": geometry}
    _write_run_manifest(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "geometry", "all"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--blender")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.phase == "plan":
        result = run_plan(args.input, args.output, args.task)
    elif args.phase == "geometry":
        result = run_geometry(args.input, args.output, args.task, args.max_rounds,
                              args.blender, args.plan, args.resume)
    else:
        if args.resume:
            parser.error("--resume requires --phase geometry")
        result = run_phases(args.input, args.output, args.task, args.max_rounds, args.blender)
    if args.phase != "all":
        manifest_result = {"status": result["status"], args.phase: result}
        if args.phase == "geometry":
            manifest_result["plan"] = {
                "status": "passed", "plan_path": str(args.plan or args.output / "plan" / "plan.md"),
            }
        _write_run_manifest(args.output.expanduser().resolve(), manifest_result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") in {"passed", "failed_baseline", "max_rounds_exceeded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
