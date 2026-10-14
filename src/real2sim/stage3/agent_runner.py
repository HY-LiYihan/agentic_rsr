"""Start a separate Codex SDK thread to audit a Stage 3 physics scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from real2sim.stage3.physics_probe import observe_scene


REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_FILE = Path(__file__).with_name("agent_prompt.md")
PRIOR_PROMPT_FILE = Path(__file__).with_name("force_prior_prompt.md")
PRIOR_SCHEMA = {
    "type": "object",
    "properties": {
        "body": {"type": "string"},
        "material_guess": {"type": "string"},
        "mass_range_kg": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "sliding_friction_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "grasp_force_range_per_jaw_n": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "rationale": {"type": "string"},
    },
    "required": ["body", "material_guess", "mass_range_kg", "sliding_friction_range",
                 "grasp_force_range_per_jaw_n", "confidence", "rationale"],
    "additionalProperties": False,
}
RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["provisional", "needs_repair"]},
        "real_world_verified": {"type": "boolean"},
        "objects_checked": {"type": "array", "items": {"type": "string"}},
        "static_report": {"type": "string"},
        "grasp_report": {"type": ["string", "null"]},
        "articulation_report": {"type": ["string", "null"]},
        "candidate_scene": {"type": ["string", "null"]},
        "findings": {"type": "string"},
        "limitations": {"type": "string"},
    },
    "required": ["status", "real_world_verified", "objects_checked", "static_report",
                 "grasp_report", "articulation_report", "candidate_scene", "findings", "limitations"],
    "additionalProperties": False,
}
MAX_REPAIR_TURNS = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _command_trace(items: list[Any]) -> list[dict[str, Any]]:
    trace = []
    for item in items:
        payload = getattr(item, "root", item)
        if getattr(payload, "type", None) == "commandExecution":
            output = payload.aggregated_output or ""
            marker = re.search(r"__RC__=(\d+)", output)
            trace.append({"command": payload.command, "exit_code": payload.exit_code,
                          "probe_exit_code": int(marker.group(1)) if marker else None,
                          "status": str(payload.status),
                          "output_tail": output[-1500:]})
    return trace


def _validate_visual_prior(prior: dict[str, Any], bodies: list[str]) -> None:
    if (prior.get("body") not in bodies or not isinstance(prior.get("rationale"), str) or
            not prior["rationale"].strip() or prior.get("confidence") not in ("low", "medium", "high") or
            not isinstance(prior.get("material_guess"), str) or not prior["material_guess"].strip()):
        raise ValueError("visual force prior needs a named free body and rationale")
    for field in ("mass_range_kg", "sliding_friction_range", "grasp_force_range_per_jaw_n"):
        values = prior.get(field)
        if (not isinstance(values, list) or len(values) != 2 or
                any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < (0 if field == "sliding_friction_range" else 0.000001)
                    for value in values) or values[0] > values[1]):
            raise ValueError(f"visual force prior has an invalid {field}")


def verify_run(output: Path, original_hash: str, scene: Path, hypotheses: dict[str, Any],
               trace: list[dict[str, Any]], agent_result: dict[str, Any],
               replay_physics: bool = False) -> dict[str, Any]:
    """Verify execution and generated reports independently of the agent's claims."""
    errors = []
    warnings = []
    if _sha256(scene) != original_hash:
        errors.append("original scene changed")
    reports = {}
    baseline_paths = {}
    candidate_reports = {}
    candidate = agent_result.get("candidate_scene")
    articulated = not agent_result.get("grasp_report") and bool(agent_result.get("articulation_report"))
    probe = "articulation" if articulated else "grasp"
    if not articulated and not agent_result.get("grasp_report"):
        errors.append("no grasp or articulation probe report")
    candidate_path = (Path(candidate) if Path(candidate).is_absolute() else REPO_ROOT / candidate).resolve() if candidate else None
    for name in ("static", probe):
        raw_path = agent_result.get(f"{name}_report", "")
        if ".json: " in raw_path:
            raw_path = raw_path.split(".json: ", 1)[0] + ".json"
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        try:
            if not path.resolve().is_relative_to(output.resolve()):
                raise ValueError("outside output directory")
            reported = json.loads(path.read_text(encoding="utf-8"))
            if candidate_path is not None and reported["scene"] == str(candidate_path):
                candidate_reports[name] = reported
                matches = []
                for previous in output.rglob("report.json"):
                    try:
                        earlier = json.loads(previous.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if (earlier.get("scene") == str(scene) and
                            (name == "static" and "objects" in earlier or
                             name == "grasp" and earlier.get("method") == "two_frictional_actuated_box_jaws")):
                        matches.append((previous, earlier))
                if name == "grasp":
                    candidate_forces = [trial["target_normal_force_per_jaw_n"]
                                        for trial in reported["trials"]]
                    matches = [(previous, earlier) for previous, earlier in matches
                               if earlier.get("body") == reported.get("body") and
                               all(earlier.get(field) == reported.get(field) for field in
                                   ("duration_s", "settle_s", "min_lift_m", "jaw_friction", "jaw_size_m", "grasp_axis", "jaw_damping"))]
                    source_trials = {}
                    for previous, earlier in sorted(matches, key=lambda item: item[0].stat().st_mtime_ns):
                        for trial in earlier.get("trials", []):
                            source_trials[trial["target_normal_force_per_jaw_n"]] = (previous, earlier, trial)
                    if any(force not in source_trials for force in candidate_forces):
                        raise ValueError("missing source grasp trials for candidate force grid")
                    selected = [source_trials[force] for force in candidate_forces]
                    template = selected[-1][1]
                    reports[name] = {**template, "trials": [entry[2] for entry in selected],
                                     "passed": any(entry[2]["lifted"] for entry in selected)}
                    baseline_paths[name] = sorted({str(entry[0].resolve()) for entry in selected})
                else:
                    if not matches:
                        raise ValueError(f"missing matching source {name} report for candidate")
                    previous, earlier = max(matches, key=lambda item: item[0].stat().st_mtime_ns)
                    baseline_paths[name] = str(previous.resolve())
                    reports[name] = earlier
            else:
                baseline_paths[name] = str(path.resolve())
                reports[name] = reported
            if reports[name]["scene"] != str(scene):
                raise ValueError("report references a different scene")
            if name == "static" and not reports[name].get("objects"):
                raise ValueError("static report did not test any movable objects")
            if name == "grasp" and (reports[name].get("method") !=
                                   "two_frictional_actuated_box_jaws" or
                                   not reports[name].get("trials")):
                raise ValueError("grasp report has no pinch trials")
            if name == "articulation" and (not reports[name].get("responses") or
                                           not reports[name].get("checks") or
                                           not reports[name].get("stability")):
                raise ValueError("articulation report needs responses, checks and stability")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {name} report: {exc}")
    for command in ("objects", "static", "run" if articulated else "grasp"):
        matching = [item for item in trace if re.search(
            r"(?:physics_probe|real2sim-stage3-physics)\s+" + command + r"\b", item["command"])]
        completed = any(item["exit_code"] in (0, 2) or item.get("probe_exit_code") in (0, 2)
                        for item in matching)
        if not completed and ("articulation" if command == "run" else command) in reports:
            completed = any(f'"scene": "{scene}"' in item.get("output_tail", "")
                            and '"passed":' in item.get("output_tail", "") for item in matching)
            if completed:
                warnings.append(f"{command} probe produced a valid report but its shell wrapper exited nonzero")
        if not completed:
            errors.append(f"no completed agent command: {command}")
    candidate_recheck = None
    if articulated and candidate:
        errors.append("articulated candidate requires a separate joint recheck")
    if candidate:
        path = candidate_path
        if not any(re.search(r"(?:physics_probe|real2sim-stage3-physics)\s+patch\b", item["command"])
                   and str(path) in item["command"] and item["exit_code"] == 0
                   for item in trace):
            errors.append("no completed agent patch command for candidate")
        for command_name in ("static", "grasp"):
            if not any(re.search(r"(?:physics_probe|real2sim-stage3-physics)\s+" +
                                 command_name + r"\b", item["command"])
                       and str(path) in item["command"] and item["exit_code"] in (0, 2)
                       for item in trace):
                errors.append(f"no completed agent {command_name} recheck for candidate")
        if not path.is_file() or not path.resolve().is_relative_to(output.resolve()):
            errors.append("candidate scene missing or outside output directory")
        else:
            from real2sim.stage3.physics_probe import check_grasp, check_static, load_scene

            handle = load_scene(path)
            handle.close()
            baseline_static = reports.get("static", {})
            static_candidate = check_static(
                path, duration=baseline_static.get("duration_s", 3.0),
                settle=baseline_static.get("settle_s", 0.1),
                max_drift_m=baseline_static.get("max_drift_m", 0.01),
                max_rotation_rad=baseline_static.get("max_rotation_rad", 0.1))
            candidate_recheck = {"static_passed": static_candidate["passed"],
                                 "static_objects": static_candidate["objects"]}
            if not static_candidate["passed"]:
                errors.append("candidate failed independent static recheck")
            reported_static = candidate_reports.get("static")
            if reported_static and (reported_static.get("passed") != static_candidate["passed"] or
                                    len(reported_static.get("objects", [])) != len(static_candidate["objects"]) or
                                    any(a["body"] != b["body"] or
                                        abs(a["drift_m"] - b["drift_m"]) > 1e-5
                                        for a, b in zip(reported_static["objects"], static_candidate["objects"]))):
                errors.append("candidate static report differs from independent recheck")
            grasp = candidate_reports.get("grasp") or reports.get("grasp")
            if grasp:
                forces = [trial["target_normal_force_per_jaw_n"] for trial in grasp["trials"]]
                grasp_candidate = check_grasp(
                    path, grasp["body"], forces, duration=grasp["duration_s"],
                    settle=grasp["settle_s"], min_lift_m=grasp["min_lift_m"],
                    jaw_friction=grasp["jaw_friction"], jaw_size=grasp["jaw_size_m"],
                    grasp_axis=grasp.get("grasp_axis", "x"),
                    jaw_damping=grasp.get("jaw_damping", 2.0))
                candidate_recheck["grasp_passed"] = grasp_candidate["passed"]
                candidate_recheck["grasp_trials"] = grasp_candidate["trials"]
                if not grasp_candidate["passed"]:
                    errors.append("candidate failed independent two-pad grasp recheck")
                reported_grasp = candidate_reports.get("grasp")
                if reported_grasp and (reported_grasp.get("passed") != grasp_candidate["passed"] or
                                       len(reported_grasp.get("trials", [])) != len(grasp_candidate["trials"]) or
                                       any(abs(a["rise_m"] - b["rise_m"]) > 1e-5 or
                                           a["lifted"] != b["lifted"]
                                           for a, b in zip(reported_grasp["trials"], grasp_candidate["trials"]))):
                    errors.append("candidate grasp report differs from independent recheck")
    if (reports.get("static", {}).get("passed") is False and not candidate and
            agent_result.get("status") != "needs_repair"):
        errors.append("static failure was not flagged as needing repair")
    replay = None
    if replay_physics and len(reports) == 2:
        from real2sim.stage3.physics_probe import check_grasp, check_static, run_probe

        static = reports["static"]
        static_again = check_static(scene, duration=static["duration_s"],
                                    settle=static["settle_s"],
                                    max_drift_m=static["max_drift_m"],
                                    max_rotation_rad=static["max_rotation_rad"])
        replay = {"static_reproduced": (static_again["passed"] == static["passed"] and
                                      len(static_again["objects"]) == len(static["objects"]) and
                                      all(a["body"] == b["body"] and
                                          abs(a["drift_m"] - b["drift_m"]) < 1e-5 and
                                          abs(a["rotation_rad"] - b["rotation_rad"]) < 1e-5
                                          for a, b in zip(static_again["objects"], static["objects"])))}
        if articulated:
            script_path = output / "articulation_script.json"
            if not script_path.is_file():
                errors.append("missing articulation script for independent replay")
            else:
                try:
                    script = json.loads(script_path.read_text(encoding="utf-8"))
                    if not script.get("actions") or not script.get("checks"):
                        raise ValueError("articulation script requires actions and checks")
                    rerun = run_probe(scene, script)
                    expected = reports["articulation"]
                    replay["articulation_reproduced"] = (
                        rerun["passed"] == expected["passed"] and
                        rerun["stability"]["finite_state"] == expected["stability"]["finite_state"] and
                        all(body in expected["responses"] and
                            abs(response["displacement_m"] - expected["responses"][body]["displacement_m"]) < 1e-5 and
                            abs(response["rotation_rad"] - expected["responses"][body]["rotation_rad"]) < 1e-5
                            for body, response in rerun["responses"].items()))
                except (OSError, ValueError, KeyError) as exc:
                    errors.append(f"invalid articulation script: {exc}")
        else:
            grasp = reports["grasp"]
            grasp_again = check_grasp(scene, grasp["body"],
                                      [trial["target_normal_force_per_jaw_n"] for trial in grasp["trials"]],
                                      duration=grasp["duration_s"], settle=grasp["settle_s"],
                                      min_lift_m=grasp["min_lift_m"],
                                      jaw_friction=grasp["jaw_friction"], jaw_size=grasp["jaw_size_m"],
                                      grasp_axis=grasp.get("grasp_axis", "x"),
                                      jaw_damping=grasp.get("jaw_damping", 2.0))
            replay["grasp_reproduced"] = (
                grasp_again["passed"] == grasp["passed"] and
                len(grasp_again["trials"]) == len(grasp["trials"]) and
                all(a["target_normal_force_per_jaw_n"] == b["target_normal_force_per_jaw_n"] and
                    abs(a["rise_m"] - b["rise_m"]) < 1e-5 and
                    a["held_at_end"] == b["held_at_end"] and
                    abs(a["slip_relative_to_jaws_m"] - b["slip_relative_to_jaws_m"]) < 1e-5 and
                    abs(a["two_sided_contact_fraction"] - b["two_sided_contact_fraction"]) < 1e-5 and
                    all(abs(left - right) < 1e-4 for left, right in zip(
                        a["peak_normal_force_per_jaw_n"], b["peak_normal_force_per_jaw_n"]))
                    for a, b in zip(grasp_again["trials"], grasp["trials"])))
        if not all(replay.values()):
            errors.append("independent physics replay differs from agent reports")
    if agent_result.get("real_world_verified") or agent_result.get("status") == "verified":
        errors.append("real-world verification requires an independent measurement comparator")
    return {"passed": not errors, "errors": errors, "warnings": warnings,
            "baseline_report_paths": baseline_paths,
            "sim_static_passed": reports.get("static", {}).get("passed"),
            "sim_grasp_passed": reports.get("grasp", {}).get("passed"),
            "sim_articulation_passed": reports.get("articulation", {}).get("passed"),
            "candidate_recheck": candidate_recheck,
            "independent_replay": replay,
            "real_world_verified": False}


def run_agent(scene: Path, hypotheses_path: Path, output: Path,
              task_context: str = "No task description was supplied; rely on the scene and hypotheses.") -> dict[str, Any]:
    from openai_codex import ApprovalMode, Codex, LocalImageInput, Sandbox, TextInput

    scene = scene.expanduser().resolve(strict=True)
    hypotheses_path = hypotheses_path.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if output == scene.parent or scene.is_relative_to(output) or hypotheses_path.is_relative_to(output):
        raise ValueError("output must not contain the original scene or hypotheses")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use an empty output directory: {output}")
    hypotheses = json.loads(hypotheses_path.read_text(encoding="utf-8"))
    if not isinstance(hypotheses, dict) or not isinstance(hypotheses.get("hypotheses"), list):
        raise ValueError("hypotheses must have a hypotheses list")
    original_hash = _sha256(scene)
    overview = output / "observe/overview.png"
    output.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"scene": str(scene), "hypotheses": str(hypotheses_path),
                              "scene_sha256": original_hash, "thread_id": None}
    try:
        observation = observe_scene(scene, output / "observe")
        record["observation"] = observation
        joints = {item["joint"] for item in observation["objects"]}
        probe_mode = "grasp one free-joint object" if "free" in joints else (
            "probe an existing hinge/slide joint; there is no free-joint object" if
            joints & {"hinge", "slide"} else "no movable task object; explain the limitation")
        prompt = PROMPT_FILE.read_text(encoding="utf-8").format(
            scene=scene, hypotheses=hypotheses_path, observation=output / "observe/observation.json",
            output=output, task_context=task_context, probe_mode=probe_mode,
            visual_prior=output / "visual_force_prior.json" if "free" in joints else "not applicable (joint-only scene)")
        with Codex() as codex:
            thread = codex.thread_start(cwd=str(REPO_ROOT), sandbox=Sandbox.full_access,
                                        approval_mode=ApprovalMode.deny_all, model="gpt-6-astra", ephemeral=True)
            record["thread_id"] = thread.id
            _write_json(output / "session.json", record)
            try:
                free_bodies = [item["body"] for item in observation["objects"] if item["joint"] == "free"]
                if free_bodies:
                    prior_prompt = PRIOR_PROMPT_FILE.read_text(encoding="utf-8").format(
                        body_names=json.dumps(free_bodies, ensure_ascii=False), task_context=task_context)
                    (output / "visual_force_prior_prompt.md").write_text(prior_prompt, encoding="utf-8")
                    prior_turn = thread.run([TextInput(text=prior_prompt), LocalImageInput(path=observation["image"])],
                                            model="gpt-6-astra", effort="high", output_schema=PRIOR_SCHEMA)
                    if _command_trace(prior_turn.items):
                        raise ValueError("visual prior turn must not execute commands or read simulator parameters")
                    prior = json.loads(prior_turn.final_response or "{}")
                    _validate_visual_prior(prior, free_bodies)
                    _write_json(output / "visual_force_prior.json", prior)
                    record["visual_force_prior"] = prior
                    record["visual_force_prior_turn_id"] = prior_turn.id
                    prompt += "\nImmutable visual prior (estimated before viewing simulator properties):\n" + json.dumps(prior, ensure_ascii=False) + "\n"
                (output / "prompt.md").write_text(prompt, encoding="utf-8")
                record["commands"] = []
                record["turns"] = []
                for attempt in range(MAX_REPAIR_TURNS):
                    inputs = ([TextInput(text=prompt), LocalImageInput(path=observation["image"])]
                              if attempt == 0 else [TextInput(text=feedback)])
                    turn = thread.run(inputs, model="gpt-6-astra", effort="high", output_schema=RESULT_SCHEMA)
                    record["turn_id"] = turn.id
                    record["turn_status"] = str(turn.status)
                    record["commands"].extend(_command_trace(turn.items))
                    try:
                        record["agent_result"] = json.loads(turn.final_response or "{}")
                        record["verification"] = verify_run(output, original_hash, scene, hypotheses,
                                                             record["commands"], record["agent_result"],
                                                             replay_physics=True)
                    except (json.JSONDecodeError, OSError, ValueError) as exc:
                        record["error"] = f"invalid agent result: {exc}"
                        break
                    verification = record["verification"]
                    candidate = record["agent_result"].get("candidate_scene")
                    candidate_checks = verification.get("candidate_recheck") or {}
                    static_ready = (candidate_checks.get("static_passed") if candidate else
                                    verification.get("sim_static_passed")) is True
                    probe_ready = ((candidate_checks.get("grasp_passed") if candidate else
                                    verification.get("sim_grasp_passed")) is True if free_bodies else
                                   verification.get("sim_articulation_passed") is True)
                    ready = verification.get("passed") is True and static_ready and probe_ready
                    record["turns"].append({"turn_id": turn.id, "status": str(turn.status),
                                            "agent_result": record["agent_result"],
                                            "verification": verification, "physics_ready": bool(ready)})
                    _write_json(output / "session.json", record)
                    if ready:
                        break
                    feedback = (
                        f"Independent check for turn {attempt + 1}/{MAX_REPAIR_TURNS}: "
                        + json.dumps(verification, ensure_ascii=False, default=str)
                        + "\nYour previous conclusion: " + json.dumps(record["agent_result"], ensure_ascii=False)
                        + "\nThe scene is NOT physically ready. Continue repairing in this SAME session; "
                          "do not accept a failed candidate. Preserve the original and use NEW candidate filenames "
                          "and unique probe directories. Inspect contact penetration and support geometry: "
                          "a non-supporting backdrop may need its collider removed, but removing it alone may "
                          "expose an oversized thin-sheet collision proxy. Repair the contact proxy and "
                          "body-local center of mass over actual support; assess mass independently. "
                          "Derive values from geometry and contacts, do not assume preset numbers. "
                          "Re-run matched static and two-pad grasp tests and render four reference views "
                          "for the final candidate. Return the full structured JSON again."
                    )
            finally:
                try:
                    codex.thread_archive(thread.id)
                    record["thread_cleanup"] = "archived"
                except Exception as exc:
                    record["thread_cleanup"] = "ephemeral_thread_not_persisted"
                    record["thread_archive_error"] = repr(exc)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        overview.unlink(missing_ok=True)
        if "observation" in record:
            observation.pop("image", None)
            observation["overview_consumed_and_deleted"] = True
            _write_json(output / "observe/observation.json", observation)
        _write_json(output / "session.json", record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--hypotheses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-context", type=Path, help="task description and Stage 2 plan, UTF-8")
    args = parser.parse_args(argv)
    task_context = args.task_context.read_text(encoding="utf-8") if args.task_context else None
    result = run_agent(args.scene, args.hypotheses, args.output,
                       task_context=task_context or "No task description was supplied; rely on the scene and hypotheses.")
    print(json.dumps({"thread_id": result["thread_id"], "output": str(args.output.resolve()),
                      "verification": result.get("verification"), "error": result.get("error")},
                     ensure_ascii=False, indent=2))
    return 0 if result.get("turns", [{}])[-1].get("physics_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
