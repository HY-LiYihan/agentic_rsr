"""Minimal robot and perception tools for Stage 3 policies."""

from __future__ import annotations

import time

import numpy as np

from .randomization import Pose3D
from .stage3_scene import Stage3Scene


def interpolate_quaternion(start, end, fraction):
    first = np.asarray(start, dtype=float)
    second = np.asarray(end, dtype=float)
    dot = float(np.dot(first, second))
    if dot < 0:
        second = -second
        dot = -dot
    if dot > 0.9995:
        result = first + fraction * (second - first)
    else:
        angle = np.arccos(np.clip(dot, -1, 1))
        result = (np.sin((1 - fraction) * angle) * first
                  + np.sin(fraction * angle) * second) / np.sin(angle)
    return tuple(result / np.linalg.norm(result))


class Stage3Tools:
    ALLOWED_TOOLS = frozenset({"state", "observe", "robot.move_pose", "robot.gripper",
                               "robot.move_trajectory", "reset", "return_control", "stop"})

    def __init__(self, scene: Stage3Scene, task: str, instruction: str,
                 initial_observation: dict | None = None, perception=None,
                 privileged: bool = False):
        self.scene = scene
        self.task = task
        self.instruction = instruction
        self.initial = initial_observation
        self.perception = perception
        self.privileged = privileged
        self.calls = []
        self.stopped = False
        self.reset_count = 0
        self.attempt = 0

    def dispatch(self, name, *args):
        if name == "object_poses" and not self.privileged:
            raise ValueError("Object poses are unavailable in this stage")
        if name not in self.ALLOWED_TOOLS and name != "object_poses":
            raise ValueError(f"Stage 3 tool is not available: {name}")
        if self.stopped and name not in ("stop", "reset"):
            raise RuntimeError("Episode stopped")
        record = {"tool": name, "args": list(args), "timestamp_s": time.time()}
        self.calls.append(record)
        try:
            result = getattr(self, name.replace(".", "_"))(*args)
            if name == "observe":
                record["result"] = {
                    "objects": list(result["objects"]),
                    "image_path": result["image"]["path"],
                    "third_person_image_path": result["third_person_image"]["path"],
                }
            else:
                record["result"] = result
            return result
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def state(self):
        return self.scene.state()

    def object_poses(self):
        observation = self.scene.observe()["objects"]
        return {name: observation[name] for name in self.perception.names
                if name in observation}

    def observe(self, queries=None):
        if self.perception is None:
            raise RuntimeError("Stage 3 observation requires a configured SAM3 service")
        return self.perception.observe(queries)

    def robot_move_pose(self, x, y, z, quaternion_wxyz):
        coordinates = np.asarray((x, y, z), dtype=float)
        if coordinates.shape != (3,) or not np.isfinite(coordinates).all():
            raise ValueError("Pose must contain three finite coordinates")
        pose = Pose3D(tuple(coordinates), tuple(quaternion_wxyz))
        self.scene.move_pose(pose)
        actual = self.scene.state()["end_effector_pose"]
        error = float(np.linalg.norm(np.asarray(actual["xyz_m"]) - coordinates))
        return {"status": "completed", "requested_xyz_m": list(pose.position),
                "actual_xyz_m": actual["xyz_m"], "position_error_m": error}

    def robot_gripper(self, width_m):
        self.scene.gripper(float(width_m))
        return {"status": "completed", "gripper": self.scene.state()["gripper"]}

    def robot_move_trajectory(self, waypoints):
        if not isinstance(waypoints, (list, tuple)) or not 1 <= len(waypoints) <= 16:
            raise ValueError("Trajectory requires 1..16 sparse waypoints")
        validated = []
        for waypoint in waypoints:
            if (not isinstance(waypoint, dict)
                    or not {"xyz_m", "quaternion_wxyz"} <= waypoint.keys()
                    or set(waypoint) - {"xyz_m", "quaternion_wxyz", "gripper_width_m"}):
                raise ValueError("Waypoint must contain xyz_m, quaternion_wxyz, and optional gripper_width_m")
            pose = Pose3D(tuple(waypoint["xyz_m"]), tuple(waypoint["quaternion_wxyz"]))
            width = waypoint.get("gripper_width_m")
            if width is not None:
                try:
                    width = float(width)
                except (TypeError, ValueError) as exc:
                    raise ValueError("gripper_width_m must be within 0..0.08") from exc
                if not np.isfinite(width) or not 0 <= width <= 0.08:
                    raise ValueError("gripper_width_m must be within 0..0.08")
            validated.append((pose, width))

        current = self.scene.robot.state().pose
        start = Pose3D(tuple(current.position), tuple(current.quaternion))
        segments = []
        for target, width in validated:
            distance = np.linalg.norm(np.asarray(target.position) - start.position)
            dot = abs(float(np.dot(start.quaternion_wxyz, target.quaternion_wxyz)))
            rotation = 2 * np.arccos(np.clip(dot, -1, 1))
            count = max(1, int(np.ceil(distance / 0.04)),
                        int(np.ceil(rotation / np.deg2rad(12))))
            segments.append((start, target, width, count))
            start = target
        if sum(count for _, _, _, count in segments) > 64:
            raise ValueError("Trajectory exceeds 64 interpolated motion segments")

        for start, target, width, count in segments:
            for index in range(1, count + 1):
                fraction = index / count
                position = ((1 - fraction) * np.asarray(start.position)
                            + fraction * np.asarray(target.position))
                quaternion = interpolate_quaternion(start.quaternion_wxyz,
                                                     target.quaternion_wxyz, fraction)
                duration_s = max(0.9, min(3.0, 0.9 + 4.0 * float(np.linalg.norm(
                    np.asarray(target.position) - np.asarray(start.position))) + rotation))
                self.scene.move_pose(Pose3D(tuple(position), quaternion), duration_s=duration_s)
            if width is not None:
                self.scene.gripper(float(width))
        actual = self.scene.state()
        return {"status": "completed", "waypoints": len(validated),
                "motion_segments": sum(count for _, _, _, count in segments),
                "last_requested_xyz_m": list(validated[-1][0].position),
                "last_position_error_m": float(np.linalg.norm(
                    np.asarray(actual["end_effector_pose"]["xyz_m"])
                    - np.asarray(validated[-1][0].position))),
                "state": actual}

    def reset(self):
        self.scene.reset()
        self.reset_count += 1
        self.attempt += 1
        self.stopped = False
        return {"status": "reset", "attempt": self.attempt, "state": self.scene.state()}

    def return_control(self, reason=""):
        return {"status": "return_control", "reason": str(reason),
                "attempt": self.attempt, "state": self.scene.state()}

    def stop(self):
        self.stopped = True
        self.scene.robot.stop()
        return {"status": "stopped"}
