"""A truth-separated, dual-profile Codex tabletop test harness."""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .episode import SimEpisode
from .observation import ExactObservation, SAM3RGBDObservation
from .program_runner import run_program
from .policy_loop import run_policy_loop
from .policy_terms import LOOP_NAME, POLICY_TYPE
from .randomization import Pose3D, RandomizationConfig
from .scene_adapter import DOWN_QUATERNION, PiperTabletopScene


EXACT_CONFIG = RandomizationConfig(
    object_position_jitter_m=(0, 0, 0), object_rotation_jitter_rad=(0, 0, 0),
    camera_mount_position_jitter_m=(0, 0, 0), camera_mount_rotation_jitter_rad=(0, 0, 0),
    camera_calibration_position_error_m=(0, 0, 0),
    camera_calibration_rotation_error_rad=(0, 0, 0),
    control_position_jitter_m=(0, 0, 0), control_rotation_jitter_rad=(0, 0, 0),
)
RANDOMIZED_CONFIG = replace(
    EXACT_CONFIG,
    object_position_jitter_m=(0.005, 0.005, 0),
    object_rotation_jitter_rad=(0, 0, math.radians(3)),
    camera_mount_position_jitter_m=(0.002, 0.002, 0.002),
    camera_mount_rotation_jitter_rad=(math.radians(1),) * 3,
    camera_calibration_position_error_m=(0.003, 0.003, 0.003),
    camera_calibration_rotation_error_rad=(math.radians(0.5),) * 3,
    control_position_jitter_m=(0.003, 0.003, 0.003),
    control_rotation_jitter_rad=(math.radians(1),) * 3,
    workspace_min_xy_m=(-0.3, -0.3), workspace_max_xy_m=(0.1, 0.3),
    min_object_separation_m=0.08,
)


@dataclass(frozen=True)
class TabletopTask:
    source: str
    target: str
    instruction: str
    scene_kind: str = "tabletop"


EMERGENCY_OBJECTS = {"emergency stop button": "Stage2_EStopHousing",
                     "green storage box": "Stage2_TrayFloor"}


def emergency_inside_box(source_xyz, box_xyz) -> bool:
    """Conservative interior volume of this Stage 3 tray, excluding its walls."""
    delta = np.asarray(source_xyz) - np.asarray(box_xyz)
    return bool(abs(delta[0]) < 0.075 and abs(delta[1]) < 0.032
                and 0.003 < delta[2] < 0.057)


def prompt_for(task: TabletopTask) -> str:
    return f"""You are writing one COMPLETE Python task program for Piper/MuJoCo.
Task: {task.instruction}
Return ONLY one complete Python file defining `def run(api):` (no Markdown, imports, or external files).

Available tools (these are the ONLY ways to interact with the simulation):
- api.observe() -> {{"objects": {{"blue block": {{"xy_m": [x,y], "xyz_m": [x,y,z]}}, ...}}, "camera": "wrist", "image": "path or null"}}. EXACT gives accurate task-object positions through this public schema; RANDOMIZED returns only objects detected by wrist RGB-D + SAM3, with estimated positions. A randomized detection can also include bbox_pixels.
- api.robot.pick_place(source_name: str, source_xy: list[float], target_xy: list[float]) -> {{"status": "completed"}}. This workspace-bounded primitive performs contact-verified grasp, transport and release, injecting control error at each motion. Use observed XY; do NOT invent coordinates.
- api.robot.survey() -> observation dict. Move through preflighted wrist viewpoints, observing with the active profile's backend at each; merge detections from the SAME episode. Use it when either required object is missing from api.observe().
- api.robot.move_pose(x, y, z, quaternion_wxyz) -> {{"status": "completed"}}; all coordinates are WORLD metres, tabletop z=0.75, approach z=0.97, grasp z=0.92, release in bowl z=0.93. Tool-down quaternion = (0,0,1,0). Motion requests are bounded.
- api.robot.gripper(width_m) -> {{"status": "completed"}}, width 0..0.08 m.
- api.agent.ask(label: str, question: str, options: list[str]) -> selected option string. ONLY when an ambiguous situation requires Codex judgement: suspend the running program, take a fresh PUBLIC observation, send it and recent public actions to Codex in the SAME conversation, and resume at this call with one of your named options. Include an "abort" option. Max 4 asks. Use if/elif or a bounded state-machine loop to select the next stage or retry; arbitrary Python line-number jumps are not supported. Do not ask for fresh code mid-episode.
- api.task.check_success() -> bool, a public observation-based check; final evaluation uses hidden simulator truth separately.
- api.task.stop() -> None.

First call api.observe(). You may observe repeatedly after ANY movement. If EITHER {task.source!r} or {task.target!r} is missing, call api.robot.survey() and use its combined observation; if still missing, return a failure dict without grasping. Read both xy_m values and use api.robot.pick_place(source_name, source_xy, target_xy). After the action, use api.task.check_success() to decide whether to return {{"status": "completed"}}. If an observation does not resolve whether grasping succeeded, use api.agent.ask("grasp_review", "Is the object grasped?", ["retry_grasp", "continue", "abort"]) and branch on the returned choice. You may catch an expected, recoverable grasp failure, then ask with its public error and a fresh observation; never silently hide errors. Prefer uninterrupted execution when the program can judge itself; do not claim completion solely because an action returned completed. Never inspect simulator internals, backend state, ground truth, scene XML, or files. EXACT is for first bring-up; the IDENTICAL file must also pass RANDOMIZED episodes to be accepted. No imports are allowed.
"""


def near_release_pose(measured, target_xy, height: float) -> bool:
    position = np.asarray(measured, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    return bool(position.shape == (3,) and target.shape == (2,)
                and np.isfinite(position).all() and np.isfinite(target).all()
                and np.linalg.norm(position[:2] - target) <= 0.012
                and height <= position[2] <= height + 0.015)


class EpisodeTools:
    def __init__(self, scene, episode, perception, task):
        self.scene, self.episode, self.perception, self.task = scene, episode, perception, task
        self.calls = []
        self.last_observation = None
        self.last_target_xy = None
        self.stopped = False
        self.agent_asks = 0
        self.max_calls = 32

    def dispatch(self, name, *args):
        if self.stopped and name != "task.stop":
            raise RuntimeError("Episode stopped")
        if len(self.calls) >= self.max_calls and name != "task.stop":
            raise RuntimeError("Tool-call limit exceeded")
        record = {"tool": name, "args": list(args), "timestamp_s": time.time()}
        self.calls.append(record)
        try:
            result = getattr(self, name.replace(".", "_"))(*args)
            record["result"] = result
            return result
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def observe(self):
        self.last_observation = self.perception.observe()
        return self.last_observation

    def agent_ask(self, label, question, options):
        if not isinstance(label, str) or not 1 <= len(label) <= 64:
            raise ValueError("Ask label must be a 1..64 character string")
        if not isinstance(question, str) or not 1 <= len(question) <= 500:
            raise ValueError("Ask question must be a 1..500 character string")
        if (not isinstance(options, (list, tuple)) or not 2 <= len(options) <= 5
                or any(not isinstance(option, str) or not 1 <= len(option) <= 64
                       for option in options) or len(set(options)) != len(options)
                or "abort" not in options):
            raise ValueError("Ask options must be 2..5 distinct named branches including abort")
        if self.agent_asks >= 4:
            raise RuntimeError("Codex ask limit exceeded")
        self.agent_asks += 1
        recent = [dict(tool=call["tool"], args=call["args"],
                       **({"error": call["error"]} if "error" in call else {}))
                  for call in self.calls[:-1] if call["tool"] not in ("observe", "agent.ask")][-8:]
        return {"label": label, "question": question, "options": list(options),
                "observation": self.dispatch("observe"), "recent_actions": recent}

    def record_agent_choice(self, label, choice):
        for call in reversed(self.calls):
            if call["tool"] == "agent.ask" and call["args"][0] == label:
                call["decision"] = choice
                return

    def robot_survey(self):
        known = dict(self.last_observation["objects"]) if self.last_observation else {}
        for position in ((-0.18, 0.17, 0.97), (-0.18, -0.17, 0.97),
                         (0.0, 0.17, 0.97), (0.0, -0.17, 0.97)):
            if self.task.source in known and self.task.target in known:
                break
            self.dispatch("robot.move_pose", *position, DOWN_QUATERNION)
            current = self.dispatch("observe")
            known.update(current["objects"])
        self.last_observation = {"objects": known, "camera": "wrist",
                                 "survey": True}
        return self.last_observation

    def robot_move_pose(self, x, y, z, quaternion_wxyz):
        coordinates = np.asarray((x, y, z), dtype=float)
        if (coordinates.shape != (3,) or not np.isfinite(coordinates).all()
                or not (-0.3 <= x <= 0.1 and -0.3 <= y <= 0.3 and 0.89 <= z <= 1.08)):
            raise ValueError("Target pose is outside the safe tabletop workspace")
        pose = Pose3D(tuple(coordinates), tuple(quaternion_wxyz))
        if abs(np.dot(pose.quaternion_wxyz, DOWN_QUATERNION)) < 0.99:
            raise ValueError("Only downward-facing tool poses are supported")
        self.episode.move_pose(pose, label="move_pose")
        return {"status": "completed", "requested": list(pose.position)}

    def robot_gripper(self, width_m):
        self.scene.gripper(float(width_m))
        return {"status": "completed"}

    def robot_pick_place(self, source_name, source_xy, target_xy):
        if self.last_observation is None or source_name not in self.last_observation["objects"]:
            raise ValueError("Source must have been observed")
        source = np.asarray(source_xy, dtype=float)
        target = np.asarray(target_xy, dtype=float)
        if (source.shape != (2,) or target.shape != (2,)
                or not np.isfinite(source).all() or not np.isfinite(target).all()
                or np.linalg.norm(source - self.last_observation["objects"][source_name]["xy_m"]) > 0.015
                or not (-0.3 <= target[0] <= 0.1 and -0.3 <= target[1] <= 0.3)):
            raise ValueError("Pick/place coordinates must derive from the public observation")
        if source_name != self.task.source:
            raise ValueError("Source is not the task block")
        if self.task.target not in self.last_observation["objects"] or np.linalg.norm(
                target - self.last_observation["objects"][self.task.target]["xy_m"]) > 0.015:
            raise ValueError("Target must derive from the public observation")
        self.last_target_xy = target.tolist()
        height = 0.965 if self.task.target.endswith(" block") else 0.93
        self.dispatch("robot.gripper", 0.08)
        grasped = False
        try:
            for position in ((*source, 0.97), (*source, 0.92)):
                self.dispatch("robot.move_pose", *position, DOWN_QUATERNION)
            self.dispatch("robot.gripper", 0.0)
            self.scene.grasp(source_name)
            grasped = True
            for position in ((*source, 0.97), (*target, 0.97)):
                self.dispatch("robot.move_pose", *position, DOWN_QUATERNION)
            try:
                self.dispatch("robot.move_pose", *target, height, DOWN_QUATERNION)
            except RuntimeError:
                measured = self.scene.robot.state().pose.position
                if not near_release_pose(measured, target, height):
                    raise
                self.calls.append({"tool": "robot.release_contact",
                                   "timestamp_s": time.time(),
                                   "measured_xyz_m": [float(value) for value in measured]})
        finally:
            if grasped:
                self.scene.release(source_name)
        self.scene.robot.step(300)
        return {"status": "completed"}

    def task_check_success(self):
        observation = self.observe()["objects"]
        if self.task.source not in observation or self.task.target not in observation:
            return False
        return bool(np.linalg.norm(np.asarray(observation[self.task.source]["xy_m"])
                                   - np.asarray(observation[self.task.target]["xy_m"])) < 0.04)

    def task_stop(self):
        self.stopped = True


class EmergencyStopTools(EpisodeTools):
    def robot_survey(self):
        known = dict(self.last_observation["objects"]) if self.last_observation else {}
        for position in ((0.35, -0.16, 0.2), (0.36, 0.15, 0.2)):
            if self.task.source in known and self.task.target in known:
                break
            self.dispatch("robot.move_pose", *position, DOWN_QUATERNION)
            known.update(self.dispatch("observe")["objects"])
        self.last_observation = {"objects": known, "camera": "wrist", "survey": True}
        return self.last_observation

    def robot_move_pose(self, x, y, z, quaternion_wxyz):
        coordinates = np.asarray((x, y, z), dtype=float)
        if (coordinates.shape != (3,) or not np.isfinite(coordinates).all()
                or not (0.25 <= x <= 0.46 and -0.25 <= y <= 0.25 and 0.14 <= z <= 0.22)):
            raise ValueError("Target pose is outside the emergency-stop workspace")
        pose = Pose3D(tuple(coordinates), tuple(quaternion_wxyz))
        if abs(np.dot(pose.quaternion_wxyz, DOWN_QUATERNION)) < 0.99:
            raise ValueError("Only downward-facing tool poses are supported")
        self.episode.move_pose(pose, label="move_pose")
        return {"status": "completed", "requested": list(pose.position)}

    def robot_pick_place(self, source_name, source_xy, target_xy):
        if self.last_observation is None or source_name not in self.last_observation["objects"]:
            raise ValueError("Source must have been observed")
        source, target = np.asarray(source_xy, dtype=float), np.asarray(target_xy, dtype=float)
        if (source.shape != (2,) or target.shape != (2,)
                or not np.isfinite(source).all() or not np.isfinite(target).all()
                or np.linalg.norm(source - self.last_observation["objects"][source_name]["xy_m"]) > 0.015
                or not (0.25 <= source[0] <= 0.46 and -0.25 <= source[1] <= 0.25)
                or not (0.25 <= target[0] <= 0.46 and -0.25 <= target[1] <= 0.25)):
            raise ValueError("Pick/place coordinates must derive from public observation")
        if source_name != self.task.source or self.task.target not in self.last_observation["objects"]:
            raise ValueError("Task objects must have been observed")
        if np.linalg.norm(target - self.last_observation["objects"][self.task.target]["xy_m"]) > 0.015:
            raise ValueError("Target must derive from public observation")
        self.last_target_xy = target.tolist()
        self.dispatch("robot.gripper", 0.08)
        grasped = False
        try:
            for height in (0.20, 0.16, 0.15):
                self.dispatch("robot.move_pose", *source, height, DOWN_QUATERNION)
            self.dispatch("robot.gripper", 0.0)
            self.scene.grasp(source_name)
            grasped = True
            for xy, height in ((source, 0.20), (target, 0.20), (target, 0.18)):
                self.dispatch("robot.move_pose", *xy, height, DOWN_QUATERNION)
        finally:
            if grasped:
                self.scene.release(source_name)
        self.scene.robot.step(1000)
        return {"status": "completed"}

    def task_check_success(self):
        observation = self.observe()["objects"]
        if self.task.source not in observation:
            return False
        if self.task.target in observation:
            target_xy = observation[self.task.target]["xy_m"]
        elif self.last_target_xy is not None:
            target_xy = self.last_target_xy
        else:
            return False
        source = np.asarray(observation[self.task.source]["xyz_m"])
        return bool(np.linalg.norm(source[:2] - target_xy) < 0.035
                    and 0.005 < source[2] < 0.08)


def run_episode(scene_path: Path, task: TabletopTask, source: str, profile: str,
                seed: int, output: Path, host: str | None = None,
                port: int | None = None, on_pause=None,
                bundle=None, segmenter=None) -> dict:
    if profile not in ("exact", "randomized"):
        raise ValueError(f"Unknown profile: {profile}")
    output.mkdir(parents=True, exist_ok=True)
    scene_hash = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    if task.scene_kind not in ("tabletop", "stage3_emergency"):
        raise ValueError(f"Unknown scene kind: {task.scene_kind}")
    emergency = task.scene_kind == "stage3_emergency"
    scene = PiperTabletopScene(scene_path, (task.source, task.target),
                               object_aliases=EMERGENCY_OBJECTS if emergency else None,
                               randomizable_objects=(task.source,) if emergency else None)
    policy_hash = bundle.hash if bundle is not None else hashlib.sha256(source.encode()).hexdigest()
    perception = None
    try:
        config = EXACT_CONFIG if profile == "exact" else (
            replace(RANDOMIZED_CONFIG, workspace_min_xy_m=(0.25, -0.25),
                    workspace_max_xy_m=(0.46, 0.25), min_object_separation_m=None)
            if emergency else RANDOMIZED_CONFIG)
        episode = SimEpisode(scene, scene_path, seed, config)
        episode.reset(scene.nominal_objects(), scene.nominal_cameras())
        perception = (ExactObservation(scene, (task.source, task.target)) if profile == "exact"
                      else SAM3RGBDObservation(
                          scene, episode, (task.source, task.target), output / "frames",
                          host, port, prompts={task.source: "red and white emergency stop switch"}
                          if emergency else None,
                          fallback_prompts={task.source: "red button"} if emergency else None,
                          segmenter=segmenter))
        tools = (EmergencyStopTools if emergency else EpisodeTools)(scene, episode, perception, task)
        before = {name: scene.object_xyz(name) for name in (task.source, task.target)}
        def decide(request):
            choice = on_pause({"profile": profile, **request})
            if choice in request["options"]:
                tools.record_agent_choice(request["label"], choice)
            return choice

        if bundle is not None:
            if on_pause is None:
                raise ValueError("Policy execution loop requires a Codex turn controller")
            episode_controller = (on_pause.episode(output) if hasattr(on_pause, "episode")
                                  else nullcontext(on_pause))
            with episode_controller as decide_episode:
                program = run_policy_loop(bundle, tools, decide_episode, profile)
        else:
            program = run_program(source, tools.dispatch, timeout=240,
                                  on_pause=decide if on_pause is not None else None)
        after = {name: scene.object_xyz(name) for name in (task.source, task.target)}
        distance = math.dist(after[task.source][:2], after[task.target][:2])
        outcome = program.get("return")
        program_completed = (isinstance(outcome, dict)
                             and (outcome.get("status") == "completed"
                                  or outcome.get("success") is True))
        success = (program["status"] == "finished" and program_completed
                   and (emergency_inside_box(after[task.source], after[task.target])
                        if emergency else distance < 0.035))
        if not emergency and task.target.endswith(" block"):
            success = success and after[task.source][2] > after[task.target][2] + 0.035
        if hashlib.sha256(scene_path.read_bytes()).hexdigest() != scene_hash:
            raise RuntimeError("SIM episode modified its source scene")
        result = {"profile": profile, "seed": seed, "policy_hash": policy_hash,
                  "policy_type": (POLICY_TYPE if bundle is not None
                                   else "legacy_programmatic"),
                  "loop": LOOP_NAME if bundle is not None else "single_program",
                  "policy_bundle": bundle.manifest() if bundle is not None else None,
                  "scene_sha256": scene_hash,
                  "perception": getattr(perception, "metadata", {"backend": "exact"}),
                  "success": success, "distance_xy_m": distance,
                  "before_xyz": before, "after_xyz": after, "program": program,
                  "calls": tools.calls, "episode": episode.to_manifest()}
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    finally:
        try:
            if perception is not None and hasattr(perception, "close"):
                perception.close()
        finally:
            scene.close()


def run_both(scene_path: Path, task: TabletopTask, source: str, output: Path,
             host: str | None = None, port: int | None = None,
             seeds: tuple[int, ...] = (101, 202),
             on_pause=None, bundle=None, segmenter=None) -> dict:
    results = [run_episode(scene_path, task, source, "exact", 0, output / "exact", host, port,
                           on_pause=on_pause, bundle=bundle, segmenter=segmenter)]
    results.extend(run_episode(scene_path, task, source, "randomized", seed,
                               output / "randomized" / str(seed), host, port,
                               on_pause=on_pause, bundle=bundle,
                               segmenter=segmenter) for seed in seeds)
    report = {"accepted": all(result["success"] for result in results),
              "loop": LOOP_NAME if bundle is not None else "single_program",
              "policy_type": (POLICY_TYPE if bundle is not None
                              else "legacy_programmatic"),
              "policy_hash": bundle.hash if bundle is not None else hashlib.sha256(source.encode()).hexdigest(),
              "results": [{key: result[key] for key in ("profile", "seed", "success", "distance_xy_m")}
                          for result in results]}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report
