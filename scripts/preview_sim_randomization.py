"""Render and measure nominal versus randomized Piper tabletop episodes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

from robot_control import PiperRobot
from robot_control.api.types import Pose
from robot_control.backends.mujoco import MujocoBackend
from sim.episode import SimEpisode
from sim.randomization import Pose3D, RandomizationConfig
from sim.scene_adapter import START_JOINTS_RAD


OBJECT_BODIES = {
    "blue block": "blue_block",
    "green block": "green_block",
    "red block": "red_block",
    "yellow bowl": "yellow_bowl",
    "blue bowl": "blue_bowl",
    "green bowl": "green_bowl",
}
CAMERA_NAMES = ("d435i_color_optical_camera", "d435i_depth_optical_camera")
DOWN_QUATERNION = (0.0, 0.0, 1.0, 0.0)


def pose_difference(first: Pose3D, second: Pose3D) -> tuple[np.ndarray, float]:
    position = np.asarray(first.position) - np.asarray(second.position)
    dot = abs(float(np.dot(first.quaternion_wxyz, second.quaternion_wxyz)))
    angle = 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))
    return position, angle


def pose_from_body(model, data, body_name: str) -> Pose3D:
    body_id = model.body(body_name).id
    joint_id = int(model.body_jntadr[body_id])
    if joint_id >= 0 and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
        qpos_address = int(model.jnt_qposadr[joint_id])
        return Pose3D(
            tuple(float(value) for value in data.qpos[qpos_address:qpos_address + 3]),
            tuple(float(value) for value in data.qpos[qpos_address + 3:qpos_address + 7]),
        )
    return Pose3D(
        tuple(float(value) for value in data.xpos[body_id]),
        tuple(float(value) for value in data.xquat[body_id]),
    )


class PiperSceneAdapter:
    def __init__(self, scene_path: Path):
        self.backend = MujocoBackend(scene=scene_path, wrist_camera=True, realtime=False)
        self.backend.connect()
        self.robot = PiperRobot(self.backend)
        self.nominal_qpos = self.backend.data.qpos.copy()
        self.nominal_qvel = self.backend.data.qvel.copy()
        self.nominal_ctrl = self.backend.data.ctrl.copy()
        self.nominal_body_pos = self.backend.model.body_pos.copy()
        self.nominal_body_quat = self.backend.model.body_quat.copy()
        self.nominal_cam_pos = self.backend.model.cam_pos.copy()
        self.nominal_cam_quat = self.backend.model.cam_quat.copy()
        self.measured_pose: Pose3D | None = None

        self.backend.data.qpos[self.backend._arm_qpos] = START_JOINTS_RAD
        self.backend.data.ctrl[self.backend._arm_actuators] = START_JOINTS_RAD
        mujoco.mj_forward(self.backend.model, self.backend.data)
        self.nominal_qpos = self.backend.data.qpos.copy()
        self.nominal_qvel = self.backend.data.qvel.copy()
        self.nominal_ctrl = self.backend.data.ctrl.copy()

    def close(self) -> None:
        self.robot.disconnect()

    def reset_nominal(self) -> None:
        self.backend.data.qpos[:] = self.nominal_qpos
        self.backend.data.qvel[:] = self.nominal_qvel
        self.backend.data.ctrl[:] = self.nominal_ctrl
        self.backend.model.body_pos[:] = self.nominal_body_pos
        self.backend.model.body_quat[:] = self.nominal_body_quat
        self.backend.model.cam_pos[:] = self.nominal_cam_pos
        self.backend.model.cam_quat[:] = self.nominal_cam_quat
        self.measured_pose = None
        mujoco.mj_forward(self.backend.model, self.backend.data)

    def set_object_pose(self, name: str, pose: Pose3D) -> None:
        body_id = self.backend.model.body(OBJECT_BODIES[name]).id
        joint_id = int(self.backend.model.body_jntadr[body_id])
        if joint_id >= 0 and self.backend.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            qpos_address = int(self.backend.model.jnt_qposadr[joint_id])
            self.backend.data.qpos[qpos_address:qpos_address + 3] = pose.position
            self.backend.data.qpos[qpos_address + 3:qpos_address + 7] = pose.quaternion_wxyz
        else:
            self.backend.model.body_pos[body_id] = pose.position
            self.backend.model.body_quat[body_id] = pose.quaternion_wxyz
        mujoco.mj_forward(self.backend.model, self.backend.data)

    def set_camera_pose(self, name: str, pose: Pose3D) -> None:
        del name
        for camera_name in CAMERA_NAMES:
            camera_id = self.backend.model.cam(camera_name).id
            self.backend.model.cam_pos[camera_id] = pose.position
            self.backend.model.cam_quat[camera_id] = pose.quaternion_wxyz
        mujoco.mj_forward(self.backend.model, self.backend.data)

    def move_pose(self, pose: Pose3D) -> None:
        self.robot.move_p(Pose(pose.position, pose.quaternion_wxyz))
        self.robot.step(1200)
        measured = self.robot.state().pose
        self.measured_pose = Pose3D(tuple(measured.position), tuple(measured.quaternion))

    def nominal_objects(self) -> dict[str, Pose3D]:
        return {
            name: pose_from_body(self.backend.model, self.backend.data, body_name)
            for name, body_name in OBJECT_BODIES.items()
        }

    def nominal_camera(self) -> Pose3D:
        camera_id = self.backend.model.cam(CAMERA_NAMES[0]).id
        return Pose3D(
            tuple(float(value) for value in self.backend.model.cam_pos[camera_id]),
            tuple(float(value) for value in self.backend.model.cam_quat[camera_id]),
        )

    def render(self, output: Path, camera: str | None = None) -> np.ndarray:
        renderer = mujoco.Renderer(self.backend.model, height=480, width=640)
        if camera is None:
            free_camera = mujoco.MjvCamera()
            free_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            free_camera.lookat[:] = [0.0, 0.0, 0.78]
            free_camera.distance = 1.55
            free_camera.azimuth = 135
            free_camera.elevation = -28
            renderer.update_scene(self.backend.data, camera=free_camera)
        else:
            renderer.update_scene(self.backend.data, camera=camera)
        image = renderer.render().copy()
        renderer.close()
        Image.fromarray(image).save(output)
        return image


def config_for(kind: str) -> RandomizationConfig:
    object_position = (0.01, 0.01, 0.0) if kind in {"objects", "combined"} else (0.0, 0.0, 0.0)
    object_rotation = (0.0, 0.0, math.radians(5.0)) if kind in {"objects", "combined"} else (0.0, 0.0, 0.0)
    camera_position = (0.004, 0.004, 0.004) if kind in {"camera", "combined"} else (0.0, 0.0, 0.0)
    camera_rotation = tuple(math.radians(2.0) for _ in range(3)) if kind in {"camera", "combined"} else (0.0, 0.0, 0.0)
    control_position = (0.005, 0.005, 0.005) if kind == "control" else (0.0, 0.0, 0.0)
    control_rotation = tuple(math.radians(2.0) for _ in range(3)) if kind == "control" else (0.0, 0.0, 0.0)
    return RandomizationConfig(
        object_position_jitter_m=object_position,
        object_rotation_jitter_rad=object_rotation,
        camera_mount_position_jitter_m=camera_position,
        camera_mount_rotation_jitter_rad=camera_rotation,
        camera_calibration_position_error_m=(0.005, 0.005, 0.005),
        camera_calibration_rotation_error_rad=tuple(math.radians(1.0) for _ in range(3)),
        control_position_jitter_m=control_position,
        control_rotation_jitter_rad=control_rotation,
        workspace_min_xy_m=(-0.30, -0.30),
        workspace_max_xy_m=(0.10, 0.30),
        min_object_separation_m=0.04,
    )


def render_variants(scene_path: Path, output: Path, seed: int) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    output.mkdir(parents=True, exist_ok=True)
    nominal_objects = None
    nominal_camera = None
    for kind in ("nominal", "objects", "camera", "combined"):
        adapter = PiperSceneAdapter(scene_path)
        try:
            if nominal_objects is None:
                nominal_objects = adapter.nominal_objects()
                nominal_camera = adapter.nominal_camera()
            episode = SimEpisode(adapter, scene_path, seed, config_for("nominal" if kind == "nominal" else kind))
            context = episode.reset(nominal_objects, {"wrist": nominal_camera})
            image = adapter.render(output / f"{kind}_free.png")
            adapter.render(output / f"{kind}_wrist.png", camera=CAMERA_NAMES[0])
            if kind == "nominal":
                nominal_manifest = episode.to_manifest()
            else:
                nominal_manifest = None
            yield_data = {
                "kind": kind,
                "manifest": episode.to_manifest(),
                "objects": {name: pose.to_manifest() for name, pose in context.objects.items()},
                "camera": context.cameras["wrist"].to_manifest(),
            }
            (output / f"{kind}.json").write_text(json.dumps(yield_data, indent=2) + "\n")
            if "records" not in locals():
                records = {}
                images = {}
                wrist_images = {}
            records[kind] = yield_data
            images[kind] = image
            wrist_images[kind] = np.asarray(Image.open(output / f"{kind}_wrist.png"))
        finally:
            adapter.close()
    return records, images, wrist_images


def pixel_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    channel_mean = difference.mean(axis=2)
    nonzero = difference[difference > 0]
    return {
        "mean_abs_rgb": float(difference.mean()),
        "p95_abs_rgb": float(np.percentile(difference, 95)),
        "p95_nonzero_abs_rgb": float(np.percentile(nonzero, 95)) if len(nonzero) else 0.0,
        "max_abs_rgb": float(difference.max()),
        "changed_pixel_fraction_gt_10": float(np.mean(channel_mean > 10.0)),
        "mse": float(np.mean(difference ** 2)),
    }


def save_comparison(images: dict[str, np.ndarray], output: Path) -> dict[str, dict[str, float]]:
    reference = images["nominal"]
    metrics = {}
    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    for column, name in enumerate(("nominal", "objects", "camera", "combined")):
        axes[0, column].imshow(images[name])
        axes[0, column].set_title(name)
        axes[0, column].axis("off")
        metrics[name] = pixel_metrics(reference, images[name])
        difference = np.abs(reference.astype(np.float32) - images[name].astype(np.float32)).mean(axis=2)
        axes[1, column].imshow(difference, cmap="magma", vmin=0, vmax=80)
        axes[1, column].set_title(f"difference: mean={metrics[name]['mean_abs_rgb']:.2f}")
        axes[1, column].axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return metrics


def run_control_probe(scene_path: Path, output: Path, seed: int) -> dict[str, object]:
    nominal_objects = None
    nominal_camera = None
    target = Pose3D((-0.08, -0.08, 0.97), DOWN_QUATERNION)
    results = {}
    for kind in ("nominal", "control"):
        adapter = PiperSceneAdapter(scene_path)
        try:
            if nominal_objects is None:
                nominal_objects = adapter.nominal_objects()
                nominal_camera = adapter.nominal_camera()
            episode = SimEpisode(adapter, scene_path, seed, config_for(kind))
            episode.reset(nominal_objects, {"wrist": nominal_camera})
            execution = episode.move_pose(target, "control_probe")
            assert adapter.measured_pose is not None
            measured = adapter.measured_pose
            commanded_error, commanded_angle = pose_difference(execution.commanded_pose, measured)
            applied_error, applied_angle = pose_difference(execution.arrived_pose, measured)
            results[kind] = {
                "commanded_pose": execution.commanded_pose.to_manifest(),
                "arrived_command_pose": execution.arrived_pose.to_manifest(),
                "measured_pose": measured.to_manifest(),
                "commanded_to_measured_mm": (commanded_error * 1000).tolist(),
                "commanded_to_measured_norm_mm": float(np.linalg.norm(commanded_error) * 1000),
                "arrived_command_to_measured_mm": (applied_error * 1000).tolist(),
                "arrived_command_to_measured_norm_mm": float(np.linalg.norm(applied_error) * 1000),
                "commanded_to_measured_rotation_deg": math.degrees(commanded_angle),
                "arrived_command_to_measured_rotation_deg": math.degrees(applied_angle),
                "control_error": execution.error.to_manifest(),
            }
        finally:
            adapter.close()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = ("x", "y", "z")
    x = np.arange(3)
    width = 0.35
    for index, kind in enumerate(("nominal", "control")):
        values = results[kind]["commanded_to_measured_mm"]
        axes[0].bar(x + (index - 0.5) * width, values, width, label=kind)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("measured - requested (mm)")
    axes[0].set_title("end-effector position error")
    axes[0].legend()
    for kind, color in (("nominal", "tab:blue"), ("control", "tab:orange")):
        values = results[kind]["measured_pose"]["position_m"]
        axes[1].scatter(values[0], values[1], s=90, label=kind, color=color)
    axes[1].scatter(target.position[0], target.position[1], marker="x", s=120, color="black", label="requested")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_title("requested vs measured XY")
    axes[1].axis("equal")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path,
                        default=ROOT / "tests/fixtures/stage3/code_as_policy_tabletop.xml")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/sim_randomization_preview")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records, images, wrist_images = render_variants(args.scene, args.output, args.seed)
    image_metrics = save_comparison(images, args.output / "free_camera_comparison.png")
    wrist_metrics = save_comparison(wrist_images, args.output / "wrist_camera_comparison.png")
    control_metrics = run_control_probe(args.scene, args.output / "control_error_comparison.png", args.seed)
    report = {
        "scene": str(args.scene.resolve()),
        "seed": args.seed,
        "configs": {kind: config_for("nominal" if kind == "nominal" else kind).to_manifest()
                    for kind in ("nominal", "objects", "camera", "combined", "control")},
        "image_metrics_free_camera": image_metrics,
        "image_metrics_wrist_camera": wrist_metrics,
        "control_metrics": control_metrics,
        "variants": records,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "image_metrics_free_camera": image_metrics,
                      "image_metrics_wrist_camera": wrist_metrics,
                      "control_metrics": control_metrics}, indent=2))


if __name__ == "__main__":
    main()
