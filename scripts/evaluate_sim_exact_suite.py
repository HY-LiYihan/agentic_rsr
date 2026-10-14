"""Run one locked SIM policy on the 22 archived Code as Policies exact scenes."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from sim.episode import SimEpisode
from sim.harness import EXACT_CONFIG, EpisodeTools, near_release_pose
from sim.observation import ExactObservation
from sim.policy_bundle import PolicyBundle
from sim.program_runner import run_program
from sim.scene_adapter import DOWN_QUATERNION, PiperTabletopScene


class SuiteTools(EpisodeTools):
    def __init__(self, scene, episode, case, initial, score_function):
        super().__init__(scene, episode, ExactObservation(scene, tuple(case.objects)), case)
        self.initial = initial
        self.score_function = score_function
        self.max_calls = 128

    def observe(self):
        result = super().observe()
        result["instruction"] = self.task.command
        result["workspace_xy_m"] = {"min": [-0.3, -0.3], "max": [0.1, 0.3]}
        return result

    def robot_survey(self):
        return self.dispatch("observe")

    def robot_pick_place(self, source_name, source_xy, target_xy):
        observation = self.last_observation
        if observation is None or source_name not in observation["objects"]:
            raise ValueError("Source must be present in a recent public observation")
        if not source_name.endswith(" block"):
            raise ValueError("The Piper primitive only grasps blocks")
        source = np.asarray(source_xy, dtype=float)
        target = np.asarray(target_xy, dtype=float)
        measured_source = np.asarray(observation["objects"][source_name]["xy_m"])
        if (source.shape != (2,) or target.shape != (2,)
                or not np.isfinite(source).all() or not np.isfinite(target).all()
                or np.linalg.norm(source - measured_source) > 0.015):
            raise ValueError("Source must match a recent public observation and coordinates must be finite")
        height = 0.93
        for name, item in observation["objects"].items():
            if name != source_name and name.endswith(" block") and np.linalg.norm(
                    target - np.asarray(item["xy_m"])) <= 0.015:
                height = min(1.06, 0.965 + max(0.0, item["xyz_m"][2] - 0.772))
                break
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
                self.calls.append({"tool": "robot.release_contact", "timestamp_s": time.time(),
                                   "measured_xyz_m": [float(value) for value in measured]})
        finally:
            if grasped:
                self.scene.release(source_name)
        self.scene.robot.step(300)
        return {"status": "completed"}

    def task_check_success(self):
        observed = self.observe()["objects"]
        current = {name: item["xyz_m"] for name, item in observed.items()}
        return bool(self.score_function(self.task, self.initial, current))


def save_final_image(scene, path: Path):
    import mujoco
    from PIL import Image

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-0.10, 0.0, 0.76]
    camera.distance = 1.1
    camera.azimuth = 90
    camera.elevation = -65
    renderer = mujoco.Renderer(scene.backend.model, height=288, width=512)
    try:
        renderer.update_scene(scene.backend.data, camera=camera)
        Image.fromarray(renderer.render().copy()).save(path)
    finally:
        renderer.close()


def run_trial(case, scene_path: Path, bundle: PolicyBundle, output: Path, score_function):
    output.mkdir(parents=True, exist_ok=True)
    scene = PiperTabletopScene(scene_path, tuple(case.objects))
    try:
        episode = SimEpisode(scene, scene_path, 0, EXACT_CONFIG)
        episode.reset(scene.nominal_objects(), scene.nominal_cameras())
        names = tuple(case.objects)
        initial = {name: scene.object_xyz(name) for name in names}
        tools = SuiteTools(scene, episode, case, initial, score_function)
        error = None
        try:
            program = run_program(bundle.sources["run"][0], tools.dispatch, timeout=240)
        except Exception as exc:
            program = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        final = {name: scene.object_xyz(name) for name in names}
        actions = [call for call in tools.calls if call["tool"] == "robot.pick_place"]
        action_success = bool(actions) and all(call.get("result", {}).get("status") == "completed"
                                               for call in actions)
        outcome = program.get("return")
        completed = (program["status"] == "finished" and isinstance(outcome, dict)
                     and outcome.get("status") == "completed" and not tools.stopped)
        try:
            image = output / "final.png"
            save_final_image(scene, image)
        except Exception as exc:
            image = None
            error = f"Final image: {type(exc).__name__}: {exc}"
        result = {
            "case_id": case.id, "instruction": case.command, "category": case.category,
            "scene": str(scene_path), "policy_hash": bundle.hash,
            "profile": "exact", "success": bool(completed and action_success and
                                               score_function(case, initial, final)),
            "physical_goal": bool(score_function(case, initial, final)),
            "action_success": action_success, "program": program,
            "initial_xyz": initial, "final_xyz": final,
            "calls": tools.calls, "final_image": str(image) if image else None,
            "render_error": error,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    finally:
        scene.close()


def save_contact_sheet(results: list[dict], output: Path):
    from PIL import Image, ImageDraw

    tile_width, tile_height, columns = 320, 210, 4
    rows = (len(results) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, result in enumerate(results):
        offset_x = index % columns * tile_width
        offset_y = index // columns * tile_height
        image = result.get("final_image")
        if image and Path(image).is_file():
            with Image.open(image) as frame:
                sheet.paste(frame.resize((tile_width, 180)), (offset_x, offset_y))
        label = f"{index:02d} {result['category']} {'PASS' if result['success'] else 'FAIL'}"
        draw.text((offset_x + 4, offset_y + 185), label, fill="green" if result["success"] else "red")
    path = output / "final_contact_sheet.png"
    sheet.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path.home() / "code_as_policy")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--indices", type=int, nargs="+", help="Zero-based archive trial indices")
    args = parser.parse_args()
    archive = args.archive or args.reference / "results/archive/legacy-runs-20260923/20260923-paper-suite-reachable"
    if (args.output / "summary.json").exists():
        parser.error("Refusing to overwrite an existing evaluation")
    sys.path.insert(0, str(args.reference.resolve()))
    Case = importlib.import_module("paper_suite").Case
    score_case = importlib.import_module("task_scoring").score_case
    bundle = PolicyBundle(args.bundle)
    trials = [json.loads(line) for line in (archive / "trials.jsonl").read_text().splitlines()]
    if args.indices is not None:
        if (len(set(args.indices)) != len(args.indices)
                or any(index < 0 or index >= len(trials) for index in args.indices)):
            parser.error("Trial indices must be distinct and within the archive")
        selected = [(index, trials[index]) for index in args.indices]
    else:
        selected = list(enumerate(trials[:args.limit]))
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, trial in selected:
        case = Case(**trial["case"])
        scene_path = archive / "scenes" / Path(trial["scene"]).name
        try:
            result = run_trial(case, scene_path, bundle,
                               args.output / f"trial_{index:04d}", score_case)
        except Exception as exc:
            result = {"case_id": case.id, "instruction": case.command,
                      "category": case.category, "profile": "exact", "success": False,
                      "error": f"{type(exc).__name__}: {exc}", "final_image": None}
        results.append(result)
        print(f"{len(results)}/{len(selected)} {case.id}: {result['success']}"
              + (f" ({result['error']})" if "error" in result else ""), flush=True)
        (args.output / "progress.json").write_text(json.dumps(
            [{"id": item["case_id"], "success": item["success"]} for item in results], indent=2))
    image = save_contact_sheet(results, args.output)
    summary = {"profile": "exact", "completed": len(results),
               "successes": sum(result["success"] for result in results),
               "rate": sum(result["success"] for result in results) / len(results),
               "policy_hash": bundle.hash, "contact_sheet": str(image),
               "results": [{"id": item["case_id"], "success": item["success"],
                            "error": item.get("error")} for item in results]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
