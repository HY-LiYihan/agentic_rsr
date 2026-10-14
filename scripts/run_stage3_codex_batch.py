"""Run Codex-generated exact SIM policies and record Stage 3 videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim.codex_sdk import SimCodexSession
from sim.program_runner import run_program, validate_program
from sim.stage3_harness import Stage3Tools
from sim.stage3_observation import Stage3RGBDObservation
from sim.stage3_scene import Stage3Scene
from sim.stage3_verification import verify_task

STAGES = (("privileged_exact", True, None),
          ("visual_exact", False, None),
          ("visual_randomized", False, 20260926))

ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = ROOT / "runs/sim_processed_stage3_all"
DATA_ROOT = ROOT / "data/processed"
SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "files": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"path": {"type": "string"}, "source": {"type": "string"}},
        "required": ["path", "source"],
    }},
}, "required": ["files"]}
AUTHORING_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "mode": {"type": "string", "enum": ["practice", "policy", "execute"]},
    "command": {"type": "string"},
    "files": SCHEMA["properties"]["files"] | {"minItems": 0},
}, "required": ["mode", "command", "files"]}
EXPERIENCE_SCHEMA = {"type": "object", "additionalProperties": False,
                     "properties": {"experience_markdown": {"type": "string", "minLength": 1}},
                     "required": ["experience_markdown"]}


def _prompt_context(context):
    context = re.sub(
        r"Stage 2 reference RGB views \(original accepted pose\):\n(?:camera_\d+: .*\n)+\n",
        "", context,
    )
    return context.strip()


def task_prompt(robot, task, context, names, practice_socket=None, *,
                stage="visual_exact", previous_policy=None, prior_experience=None):
    privileged = stage == "privileged_exact"
    stage_note = (
        "This stage exposes api.object_poses() and CLI object_poses()."
        if privileged else
        "Object ground truth is unavailable; use api.observe() only."
    )
    prior_policy = ("\nPRIOR STAGE POLICY\n"
                    + previous_policy
                    + "\nTreat this as a draft: preserve useful structure, but re-observe and "
                      "remove hard-coded coordinates that are not justified in this stage."
                    if previous_policy else "")
    prior = ("\nPRIOR STAGE EXPERIENCE\n"
             + prior_experience + "\nUse this as development notes, not as simulator truth."
             if prior_experience else "")
    practice = f"""
Practice is available through one command per response. The harness returns JSON and both
RGB images. Use state, observe, move_pose, gripper, move_trajectory, reset, return_control,
stop{', object_poses' if privileged else ''}. Submit policy.py when ready.
""" if practice_socket is not None else ""
    return f"""You are authoring policy.py for the {robot} SIM task.

TASK
name: {task}
description:
{_prompt_context(context)}

QUERY NAMES
{json.dumps(names, ensure_ascii=False)}
These are SAM3 query identifiers, not known detections or positions.

STAGE
{stage}: {stage_note}
The order is privileged exact -> visual exact -> visual randomized. Randomization starts only
in the last stage and affects object poses, camera calibration and pose execution.
{prior_policy}{prior}

OUTPUT
Return JSON with mode, command and files. Use mode="practice", command=<one CLI command>,
files=[] while exploring. Use mode="policy", command="", files=[{{"path":"policy.py",
"source":"..."}}] when revising code. policy.py must define def run(api):.
After choosing the current policy, return mode="execute", command="run_policy", files=[];
SIM will not execute a policy until you explicitly request this.

RULES
No imports, file access, simulator internals, hidden truth, private attributes, classes,
lambda, with blocks, globals, pick/place helpers, grasp helpers, joint overrides or success
oracles. Runtime has only an overall time budget; retry and reset as needed.

API
- api.state() -> EE pose, wrist camera pose, gripper width/target width, moving. No object poses.
- api.observe(queries) -> SAM3+depth estimates, wrist RGB, and fixed third-person RGB. Query
  names with a list or name-to-English-prompt mapping; missing/occluded objects are absent.
- api.robot.move_pose(x,y,z,quaternion_wxyz): world pose, quaternion order (w,x,y,z).
- api.robot.gripper(width_m): width 0..0.08; closing does not guarantee a grasp.
- api.robot.move_trajectory(waypoints): sparse world waypoints with optional gripper widths.
- api.reset(): restore the current stage's initial scene and try again.
- api.return_control(reason): return a checkpoint without resetting.
- api.stop(): stop motion; it is not proof of success.

EXECUTION
Start with state(), then observe. Re-observe after meaningful actions. Use only fresh runtime
observations for coordinates. Return status="completed" only after post-action evidence supports
the task. You may return status="checkpoint" with a reason when the outer Agent should inspect
the result and decide whether to continue, reset, revise policy, or stop. A checkpoint is not a
task failure. Otherwise return status="failed" with a reason. Two initial images are attached:
wrist first, third-person second. Do not read image files from policy.py.
{practice}
"""


def continuation_prompt(robot, task, program, actions, reset_before_run):
    return f"""Continue the same SIM task-agent conversation for {robot}, task {task}.

The previous generated policy segment ended without a verified completion.
Runtime result:
{json.dumps(program, ensure_ascii=False, indent=2)}

Public action log (estimated observations only; no simulator truth):
{json.dumps(actions[-16:], ensure_ascii=False, indent=2)}

{"The harness reset the scene before this segment. Start a fresh attempt." if reset_before_run else "The scene was not reset. Continue from the current checkpoint and re-observe before acting."}

Return mode="execute", command="run_policy", files=[] to run the same policy again, or return
mode="policy" with a replacement policy.py. Use only the same public Stage 3 interface. Do
not assume any object pose from the previous segment; call observe() again. If the current
attempt becomes unrecoverable, call api.reset() and retry. You may return
{{"status": "checkpoint", "reason": "..."}} for the outer Agent to decide the next step
without resetting. If visual evidence is insufficient but the scene is still usable, call
api.return_control(reason) so the outer Agent can inspect the next checkpoint. Only return
{{"status": "completed"}} after fresh post-action evidence; otherwise return a concise failed
status. The fixed third-person image attached to this turn is for global verification and
the wrist image is for SAM3/depth localization.
"""


def experience_prompt(robot, task, stage, report):
    return f"""Summarize this completed {stage} SIM development stage for the next fresh Agent.

Task: {task}
Robot: {robot}
The next stage will start a new conversation and read your summary document first.

Runtime result:
{json.dumps(report.get("program"), ensure_ascii=False, indent=2)}

Independent stage result:
{report.get("physical_verdict")}

Full public action log:
{json.dumps(report.get("actions", []), ensure_ascii=False, indent=2)}

Write concise Markdown covering:
- what the task requires;
- observations and coordinates that were useful, clearly marked as estimates;
- successful motion, grasp, release and verification patterns;
- failures, IK or perception problems, and what caused them;
- what the next Agent should try or avoid.

Do not include simulator internals, hidden object truth, private identifiers, or unsupported
certainty. The attached images are the visual evidence; refer to what is visibly supported.
Return JSON with one field, experience_markdown.
"""


def experience_images(output, action_log):
    images = []
    for action in reversed(action_log):
        if action.get("tool") == "observe":
            result = action.get("result", {})
            for key in ("image_path", "third_person_image_path"):
                if result.get(key):
                    images.append(Path(result[key]))
            break
    frame_paths = sorted((output / "frames").glob("frame_*.png"))
    if frame_paths:
        selected = frame_paths if len(frame_paths) <= 8 else [
            frame_paths[index] for index in np.linspace(0, len(frame_paths) - 1, 8, dtype=int)
        ]
        tile_w, tile_h = 512, 288
        sheet = Image.new("RGB", (tile_w * 2, tile_h * 4), "black")
        for index, path in enumerate(selected):
            tile = Image.open(path).convert("RGB").resize((tile_w, tile_h))
            sheet.paste(tile, ((index % 2) * tile_w, (index // 2) * tile_h))
        sheet_path = output / "experience_visuals.png"
        sheet.save(sheet_path)
        images.insert(0, sheet_path)
    return tuple(path for path in images if path.exists())


def write_experience(session, robot, task, stage, report, output, deadline):
    path = output / "experience.md"
    try:
        if time.monotonic() < deadline:
            payload = session.run(experience_prompt(robot, task, stage, report),
                                  EXPERIENCE_SCHEMA, output / "experience_summary",
                                  images=experience_images(output, report.get("actions", [])))
            text = payload.get("experience_markdown")
            if isinstance(text, str) and text.strip():
                path.write_text(text.strip() + "\n")
                return path
    except Exception as exc:
        report["experience_error"] = f"{type(exc).__name__}: {exc}"
    path.write_text(
        f"# {stage}\n\n"
        f"- program: `{report.get('program', {}).get('status')}`\n"
        f"- physical result: `{report.get('physical_verdict')}`\n"
        f"- video: `{report.get('video')}`\n"
        f"- Full action log: `{output / 'report.json'}`\n\n"
        "The Agent summary turn was unavailable before the time budget expired. "
        "Inspect the attached video and action log before continuing.\n"
    )
    return path


def policy_source(payload):
    files = payload.get("files", []) if isinstance(payload, dict) else []
    policy = next((item for item in files if item.get("path") == "policy.py"), None)
    if policy is None or not isinstance(policy.get("source"), str):
        raise ValueError("Codex response did not contain policy.py")
    return policy["source"]


def policy_requested_checkpoint(program):
    if program.get("status") == "returned":
        return True
    result = program.get("return")
    return (program.get("status") == "finished" and isinstance(result, dict)
            and result.get("status") in {"checkpoint", "return", "returned"})


def request_command(path: Path, command: str, timeout: float | None = None):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(path))
        connection.sendall((command + "\n").encode())
        connection.shutdown(socket.SHUT_WR)
        response = connection.makefile().readline()
    if not response:
        raise RuntimeError("Practice CLI disconnected")
    return json.loads(response)


def stop_practice_cli(process, path: Path | None):
    """Release the practice model before loading the evaluation model."""
    if process.poll() is not None:
        return
    if path is not None and path.exists():
        try:
            request_command(path, "quit", timeout=5)
            process.wait(timeout=5)
            return
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            pass
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def start_practice_cli(robot, task, output, privileged=False, seed=None):
    digest = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
    socket = Path(tempfile.gettempdir()) / f"arsr_sim_{digest}.sock"
    log = (output / "practice_cli.log").open("w")
    process = subprocess.Popen([
        sys.executable, str(ROOT / "scripts/stage3_cli.py"),
        "--robot", robot, "--task", task, "--listen", str(socket),
        "--image-dir", str(output / "practice_observations"),
        *(["--privileged"] if privileged else []),
        *(["--seed", str(seed)] if seed is not None else []),
    ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(100):
        if socket.exists():
            return process, log, socket
        if process.poll() is not None:
            break
        time.sleep(0.1)
    process.terminate()
    process.wait(timeout=5)
    log.close()
    raise RuntimeError("Practice CLI did not start; see practice_cli.log")


def author_policy(session, prompt, practice_socket, output, initial_images,
                  deadline, max_practice_turns=None):
    payload = session.run(prompt, AUTHORING_SCHEMA, output / "generation",
                          images=initial_images)
    with (output / "practice_actions.jsonl").open("w") as log:
        turn = 0
        while time.monotonic() < deadline:
            if payload.get("mode") == "policy":
                try:
                    source = policy_source(payload)
                    validate_program(source)
                    return source
                except (ValueError, SyntaxError) as exc:
                    response = f"Your proposed policy.py is invalid: {type(exc).__name__}: {exc}. Repair it."
                    images = ()
            elif payload.get("mode") == "practice":
                command = payload.get("command", "")
                if practice_socket is None:
                    response = "Practice CLI is unavailable in this visual stage. Submit policy.py."
                    images = ()
                elif not isinstance(command, str) or len(command) > 4096:
                    response = "Practice command must be a string of at most 4096 characters."
                    images = ()
                else:
                    try:
                        result = request_command(practice_socket, command)
                    except Exception as exc:
                        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                    log.write(json.dumps({"command": command, "result": result},
                                         ensure_ascii=False) + "\n")
                    log.flush()
                    response = f"Practice CLI result for {command!r}:\n{json.dumps(result, ensure_ascii=False)}"
                    images = tuple(Path(result["images"][view]["path"])
                                   for view in ("wrist", "third_person")
                                   if result.get("images", {}).get(view, {}).get("path"))
            else:
                response = "Return mode=practice with one CLI command, or mode=policy with a valid policy.py."
                images = ()
            if time.monotonic() >= deadline:
                raise TimeoutError("SIM authoring time budget exceeded")
            payload = session.run(response + "\nUse the same JSON format with mode, command, files.",
                                  AUTHORING_SCHEMA, output / f"practice_turn_{turn:03d}",
                                  images=images)
            turn += 1
    raise ValueError("Codex did not produce a valid policy.py within the authoring budget")


def request_policy_execution(session, source, output, deadline, turn=0):
    prompt = """policy.py is ready. Decide whether to run it now.
Return mode="execute", command="run_policy", files=[] to execute this exact policy.
If it needs changes, return mode="policy" with a replacement policy.py instead.
Do not use hidden state or assume execution succeeded before receiving its result."""
    while time.monotonic() < deadline:
        payload = session.run(prompt, AUTHORING_SCHEMA,
                              output / f"execution_request_{turn:03d}")
        if payload.get("mode") == "execute" and payload.get("command") == "run_policy":
            return source
        if payload.get("mode") == "policy":
            source = policy_source(payload)
            validate_program(source)
            prompt = "Policy revised. Return mode=execute with command=run_policy to run it, or revise it again."
            turn += 1
            continue
        prompt = "Return mode=execute with command=run_policy, or mode=policy with a valid policy.py."
        turn += 1
    raise TimeoutError("SIM authoring time budget exceeded before policy execution")


def load_task(robot, task):
    directory = DATA_ROOT / robot / "stage3" / task
    context = (directory / "task_context.md").read_text()
    manifest = json.loads((directory / "mujoco_v8/object_manifest.json").read_text())
    objects = {item["name"]: item for item in manifest["objects"]}
    instruction = json.loads((directory / "final_manifest.json").read_text()).get("task", task)
    return directory, context, objects, instruction


def sam3_endpoint():
    from perception.config import sam3_endpoint as configured_endpoint
    return configured_endpoint()


def assess_report(report, output):
    outcome = report.get("program", {}).get("return")
    claimed = (report.get("program", {}).get("status") == "finished"
               and isinstance(outcome, dict) and outcome.get("status") == "completed")
    nonphysical = any(action["tool"] == "object.set_joint"
                      for action in report.get("actions", []))
    actions = report.get("actions", [])
    last_reset = max((index for index, action in enumerate(actions)
                      if action["tool"] == "reset"), default=-1)
    stopped = any(action["tool"] in ("stop", "task.stop")
                  for action in actions[last_reset + 1:])
    review_path = output / "video_review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else None
    physical = report.get("physical_verdict")
    verified = (claimed and not nonphysical and
                (physical == "passed" or (physical is None and review is not None
                                              and review.get("verdict") == "passed")))
    report["policy_claimed_success"] = claimed
    report["success"] = verified
    report["verification"] = ("nonphysical_joint_override" if nonphysical else
                              "episode_stopped" if stopped and not claimed else
                              "physical_check_passed" if verified and physical == "passed" else
                              "video_review_passed" if verified else
                              "physical_check_failed" if physical == "failed" else
                              "pending_video_review" if claimed else
                              "checkpoint_pending" if report.get("program", {}).get("status") == "returned"
                              else "policy_failed")
    return report


def run_one(robot, task, model, timeout, effort, run_name="codex_rgbd",
            max_iterations=2, total_timeout=1800, stage=None,
            previous_policy=None, prior_experience=None):
    deadline = time.monotonic() + total_timeout
    source_scene = SCENE_ROOT / robot / task / "scene_for_sim.xml"
    output = SCENE_ROOT / robot / task / run_name
    if stage is not None:
        output = output / stage[0]
    stage_name, privileged, seed = stage or ("visual_exact", False, None)
    if output.exists():
        report_path = output / "report.json"
        if report_path.exists():
            report = assess_report(json.loads(report_path.read_text()), output)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            return report
        return {"robot": robot, "task": task, "status": "skipped_existing",
                "success": False}
    import mujoco

    host, port = sam3_endpoint()
    output.mkdir(parents=True)
    directory, context, objects, instruction = load_task(robot, task)
    probe = Stage3Scene(source_scene, robot)
    try:
        probe.set_randomization(seed)
        names = [name for name in probe.object_bodies if name != "world"
                 and objects.get(name, {}).get("semantics", {}).get("role") != "visual_only"]
        initial_image = output / "initial_wrist_rgb.jpg"
        initial_third_person_image = output / "initial_third_person_rgb.jpg"
        Image.fromarray(probe.read_rgbd().color).save(initial_image, format="JPEG", quality=95)
        Image.fromarray(probe.render_camera("stage3_camera_0", width=512, height=288)).save(
            initial_third_person_image, format="JPEG", quality=95)
    finally:
        probe.close()
    bundle = output / "bundle"
    scene = renderer = perception = None
    practice = practice_log = practice_socket = None
    try:
        practice, practice_log, practice_socket = start_practice_cli(
            robot, task, output, privileged=privileged, seed=seed)
        prompt = task_prompt(robot, task, context, names, practice_socket,
                             stage=stage_name, previous_policy=previous_policy,
                             prior_experience=prior_experience)
        (output / "prompt.txt").write_text(prompt)
        transcript = output / "authoring_conversation.jsonl"
        transcript.write_text(json.dumps({"type": "task", "prompt": prompt}, ensure_ascii=False) + "\n")
        with SimCodexSession(model, ROOT, transcript=transcript, effort=effort,
                             ephemeral=True) as session:
            source = author_policy(session, prompt, practice_socket, output,
                                   (initial_image, initial_third_person_image), deadline)
            source = request_policy_execution(session, source, output, deadline)
            if practice is not None:
                stop_practice_cli(practice, practice_socket)
                practice_socket.unlink(missing_ok=True)
                practice_log.close()
            bundle.mkdir()
            (bundle / "policy.py").write_text(source)
            (output / "authoring_thread.json").write_text(json.dumps({
                "thread_id": session.id, "model": model,
                "ephemeral": True, "resumable": False,
            }, indent=2))
            scene = Stage3Scene(source_scene, robot)
            scene.set_randomization(seed)
            renderer = mujoco.Renderer(scene.model, height=288, width=512)
            frames = output / "frames"
            frames.mkdir()
            frame_id = 0
            current_action = "episode start"

            def capture(label):
                nonlocal frame_id
                mujoco.mj_forward(scene.model, scene.data)
                renderer.update_scene(scene.data, camera="stage3_camera_0")
                image = Image.fromarray(renderer.render()).convert("RGB")
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 512, 30), fill=(0, 0, 0))
                draw.text((8, 8), f"Codex policy | {robot} | {label}", fill=(255, 255, 255))
                image.save(frames / f"frame_{frame_id:05d}.png")
                frame_id += 1

            scene.set_step_callback(lambda index, total: capture(current_action)
                                    if index == 0 or index == total - 1
                                    or index % max(1, total // (6 if total < 800 else 45)) == 0
                                    else None)
            perception = Stage3RGBDObservation(scene, tuple(names), output / "observations",
                                               host, port)
            tools = Stage3Tools(scene, task, instruction, perception=perception,
                                privileged=privileged)
            original_dispatch = tools.dispatch
            action_log = []

            def dispatch(name, *args):
                nonlocal current_action
                current_action = name
                record = {"tool": name, "args": list(args)}
                action_log.append(record)
                capture(name)
                try:
                    result = original_dispatch(name, *args)
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    capture(name + " (failed)")
                    raise
                record["result"] = ({
                    "objects": result["objects"],
                    "image_path": result["image"]["path"],
                    "third_person_image_path": result["third_person_image"]["path"],
                } if name == "observe" else result)
                capture(name + " (done)")
                return result

            capture("episode start")
            program = None
            iteration = 0
            while time.monotonic() < deadline:
                iteration_dir = output / "iterations" / f"{iteration:03d}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                (iteration_dir / "policy.py").write_text(source)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    program = {"status": "timeout", "error": "Total run time limit exceeded"}
                    break
                program = run_program(source, dispatch, timeout=remaining,
                                      interface="stage3")
                if (program.get("status") == "finished"
                        and isinstance(program.get("return"), dict)
                        and program["return"].get("status") == "completed"):
                    verdict = verify_task(scene, task)
                    if verdict != "failed":
                        break
                    program = {"status": "verification_failed", "error":
                               "Independent task check failed; reset and retry."}
                iteration += 1
                if time.monotonic() >= deadline:
                    break
                reset_before_run = not policy_requested_checkpoint(program)
                if reset_before_run:
                    try:
                        dispatch("reset")
                    except Exception:
                        break
                latest_images = []
                for action in reversed(action_log):
                    if action.get("tool") == "observe":
                        result = action.get("result", {})
                        for key in ("image_path", "third_person_image_path"):
                            path = result.get(key)
                            if path:
                                latest_images.append(Path(path))
                        break
                refinement = continuation_prompt(robot, task, program, action_log,
                                                 reset_before_run)
                visuals = tuple(dict.fromkeys(latest_images + list(
                    experience_images(output, action_log))))
                payload = session.run(refinement, AUTHORING_SCHEMA,
                                      output / f"generation_{iteration + 1:03d}",
                                      images=visuals)
                if payload.get("mode") == "execute" and payload.get("command") == "run_policy":
                    pass
                else:
                    source = policy_source(payload)
                    validate_program(source)
                    source = request_policy_execution(
                        session, source, output, deadline, iteration + 1)
                (bundle / "policy.py").write_text(source)
            physical_verdict = (verify_task(scene, task) if program and
                                program.get("status") == "finished" else "failed")
            randomization_manifest = (scene.randomizer.to_manifest()
                                      if scene.randomizer is not None else None)
            capture("episode end")
            perception_metadata = perception.metadata
            perception.close()
            perception = None
            renderer.close()
            renderer = None
            scene.close()
            scene = None
            video = output / "simulation.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
                            "-i", str(frames / "frame_%05d.png"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], check=True)
            report = {"robot": robot, "task": task, "status": "evaluated",
                      "program": program, "actions": action_log, "frames": frame_id,
                      "video": str(video), "policy": str(bundle / "policy.py"),
                      "iterations": iteration + 1,
                      "authoring_thread_id": session.id,
                      "authoring_thread_ephemeral": True, "stage": stage_name,
                      "randomization_seed": seed,
                      "perception": perception_metadata,
                      "physical_verdict": physical_verdict}
            if randomization_manifest is not None:
                (output / "randomization.json").write_text(json.dumps(
                    randomization_manifest, ensure_ascii=False, indent=2))
            write_experience(session, robot, task, stage_name, report, output, deadline)
    except Exception as exc:
        report = {"robot": robot, "task": task, "status": "failed",
                  "error": f"{type(exc).__name__}: {exc}"}
        if perception is not None:
            report["perception"] = perception.metadata
    finally:
        try:
            if perception is not None:
                perception.close()
        finally:
            try:
                if renderer is not None:
                    renderer.close()
            finally:
                try:
                    if scene is not None:
                        scene.close()
                finally:
                    if practice is not None:
                        stop_practice_cli(practice, practice_socket)
                    if practice_socket is not None:
                        practice_socket.unlink(missing_ok=True)
                    if practice_log is not None and not practice_log.closed:
                        practice_log.close()
    report = assess_report(report, output)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def run_three_stages(robot, task, model, timeout, effort, run_name="codex_rgbd",
                     max_iterations=2, total_timeout=1800, random_seed=20260926):
    output = SCENE_ROOT / robot / task / run_name
    summary_path = output / "report.json"
    if summary_path.exists() and "stages" not in json.loads(summary_path.read_text()):
        return {"robot": robot, "task": task, "status": "skipped_existing_legacy_run",
                "success": False, "verification": "use_a_new_run_name"}
    if output.exists() and not summary_path.exists() and any(output.iterdir()):
        return {"robot": robot, "task": task, "status": "skipped_existing_incomplete_run",
                "success": False, "verification": "use_a_new_run_name"}
    completed = []
    experience_notes = []
    previous_policy = None
    for name, privileged, seed in STAGES:
        stage = (name, privileged, random_seed if seed is not None else None)
        report = run_one(robot, task, model, timeout, effort, run_name,
                         max_iterations, total_timeout, stage=stage,
                         previous_policy=previous_policy,
                         prior_experience="\n\n".join(experience_notes) or None)
        completed.append({"stage": stage[0], "success": report.get("success", False),
                          "verification": report.get("verification"),
                          "report": str(output / stage[0] / "report.json")})
        if not report.get("success"):
            break
        policy_path = Path(report.get("policy", ""))
        if policy_path.exists():
            previous_policy = policy_path.read_text()
        experience_path = output / stage[0] / "experience.md"
        if experience_path.exists():
            experience_notes.append(f"## {stage[0]}\n{experience_path.read_text()}")
            (output / "experience_memory.md").write_text(
                "\n\n".join(experience_notes))
    result = {"robot": robot, "task": task, "stages": completed,
              "success": len(completed) == len(STAGES) and completed[-1]["success"],
              "verification": completed[-1]["verification"]}
    output.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("piper", "franka"))
    parser.add_argument("--task")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--run-name", default="codex_three_stage_v1",
                        help="Output directory name per scene; existing runs are not overwritten")
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--max-iterations", type=int, default=2,
                        help="Maximum outer Codex continuation/refinement iterations")
    parser.add_argument("--total-timeout", type=int, default=1800,
                        help="Total run time budget in seconds, including Codex turns")
    parser.add_argument("--random-seed", type=int, default=20260926,
                        help="Base seed for the randomized stage; resets use consecutive seeds")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        parser.error("--run-name must be a single directory name")
    jobs = []
    for robot in (args.robot,) if args.robot else ("piper", "franka"):
        for directory in sorted((DATA_ROOT / robot / "stage3").iterdir()):
            if directory.is_dir() and (args.task is None or directory.name == args.task):
                jobs.append((robot, directory.name))
    results = [run_three_stages(robot, task, args.model, args.timeout, args.effort,
                                args.run_name, args.max_iterations, args.total_timeout,
                                args.random_seed)
               for robot, task in jobs]
    (SCENE_ROOT / f"{args.run_name}_batch_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({"count": len(results),
                      "verified": sum(r.get("success", False) for r in results),
                      "policy_claimed": sum(r.get("policy_claimed_success", False) for r in results),
                      "failed_or_invalid": sum(not r.get("policy_claimed_success", False)
                                               or r.get("verification") == "nonphysical_joint_override"
                                               for r in results),
                      "pending_review": sum(r.get("verification") == "pending_video_review"
                                            for r in results)}, indent=2))


if __name__ == "__main__":
    main()
