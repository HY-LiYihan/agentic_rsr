"""Generic exact Stage 3 scene adapter for Piper and FR3."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .randomization import EpisodeRandomizer, Pose3D, RandomizationConfig, _apply_delta

DOWN_QUATERNION = (0.0, 0.0, 1.0, 0.0)


class Stage3Scene:
    def __init__(self, scene: Path, robot_name: str):
        import mujoco
        from robot_control.api.robot import Robot

        self.mujoco = mujoco
        self.robot_name = "franka_fr3" if robot_name == "franka" else robot_name
        config = {"scene": Path(scene).resolve()}
        self.robot = Robot.connect(backend="mujoco", robot=self.robot_name, config=config)
        self.backend = self.robot._backend
        self.model = self.backend.model
        self.data = self.backend.data
        mount_name = "piper_mount" if robot_name == "piper" else "fr3_mount"
        mount_id = self.model.body(mount_name).id
        descendants = set()
        pending = [mount_id]
        while pending:
            body_id = pending.pop()
            descendants.add(body_id)
            pending.extend(np.flatnonzero(self.model.body_parentid == body_id).tolist())
        self.robot_body_ids = descendants
        self.object_bodies = {}
        self.free_joints = {}
        self.articulated_joints = {}
        for body_id in range(self.model.nbody):
            name = self.model.body(body_id).name
            if body_id in descendants or name.startswith("stage3_"):
                continue
            joint_id = int(self.model.body_jntadr[body_id])
            if joint_id >= 0:
                joint_name = self.model.joint(joint_id).name
                joint_type = int(self.model.jnt_type[joint_id])
                if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                    self.free_joints[name] = joint_name
                elif joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                    int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    self.articulated_joints[name] = joint_name
            self.object_bodies[name] = body_id
        self.nominal_qpos = self.data.qpos.copy()
        self.nominal_ctrl = self.data.ctrl.copy()
        self.nominal_eq_data = self.model.eq_data.copy()
        self.nominal_body_pos = self.model.body_pos.copy()
        self.nominal_body_quat = self.model.body_quat.copy()
        self.nominal_site_pos = self.model.site_pos.copy()
        self.nominal_site_quat = self.model.site_quat.copy()
        self.nominal_cam_pos = self.model.cam_pos.copy()
        self.nominal_cam_quat = self.model.cam_quat.copy()
        self.random_seed = None
        self.randomizer = None
        self.randomization_resets = 0
        self.camera_calibration = None
        self._on_step = None
        self._camera_renderers = {}

    def close(self):
        for renderer in self._camera_renderers.values():
            renderer.close()
        self._camera_renderers.clear()
        self.robot.disconnect()

    def set_step_callback(self, callback):
        self._on_step = callback

    def step(self, steps: int):
        for index in range(int(steps)):
            self.robot.step(1)
            if self._on_step is not None:
                self._on_step(index, int(steps))

    def reset(self):
        self.backend._motion = None
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.nominal_qpos
        self.data.ctrl[:] = self.nominal_ctrl
        self.model.eq_data[:] = self.nominal_eq_data
        self.model.body_pos[:] = self.nominal_body_pos
        self.model.body_quat[:] = self.nominal_body_quat
        self.model.site_pos[:] = self.nominal_site_pos
        self.model.site_quat[:] = self.nominal_site_quat
        self.model.cam_pos[:] = self.nominal_cam_pos
        self.model.cam_quat[:] = self.nominal_cam_quat
        self.mujoco.mj_forward(self.model, self.data)
        if self.random_seed is not None:
            self.randomizer = EpisodeRandomizer(
                self.random_seed + self.randomization_resets,
                RandomizationConfig(
                    camera_mount_position_jitter_m=(0.003, 0.003, 0.003),
                    camera_mount_rotation_jitter_rad=(math.radians(0.5),) * 3,
                ),
            )
            self.randomization_resets += 1
            poses = {name: Pose3D(self.body_xyz(name), self.body_quat(name))
                     for name in self.free_joints}
            for name, pose in self.randomizer.sample_object_poses(poses).items():
                joint_id = self.model.joint(self.free_joints[name]).id
                address = self.model.jnt_qposadr[joint_id]
                self.data.qpos[address:address + 7] = (*pose.position, *pose.quaternion_wxyz)
            if self.robot_name == "piper":
                mount_id = self.model.body("d435i_link").id
                nominal = Pose3D(tuple(self.model.body_pos[mount_id]),
                                 tuple(self.model.body_quat[mount_id]))
                camera = self.randomizer.sample_camera("wrist_mount", nominal)
                self.model.body_pos[mount_id] = camera.physical_pose.position
                self.model.body_quat[mount_id] = camera.physical_pose.quaternion_wxyz
            else:
                site_id = self.model.site("d435i_color_optical_frame").id
                nominal = Pose3D(tuple(self.model.site_pos[site_id]),
                                 tuple(self.model.site_quat[site_id]))
                camera = self.randomizer.sample_camera("wrist_mount", nominal)
                self.model.site_pos[site_id] = camera.physical_pose.position
                self.model.site_quat[site_id] = camera.physical_pose.quaternion_wxyz
            self.camera_calibration = camera
            camera_id = self.model.camera("stage3_camera_0").id
            nominal = Pose3D(tuple(self.model.cam_pos[camera_id]),
                             tuple(self.model.cam_quat[camera_id]))
            third_person = self.randomizer.sample_camera("third_person", nominal)
            self.model.cam_pos[camera_id] = third_person.physical_pose.position
            self.model.cam_quat[camera_id] = third_person.physical_pose.quaternion_wxyz
            self.mujoco.mj_forward(self.model, self.data)
        else:
            self.camera_calibration = None
            self.randomizer = None

    def set_randomization(self, seed: int | None):
        self.random_seed = seed
        self.randomization_resets = 0
        self.reset()

    def render_camera(self, camera_name: str = "stage3_camera_0",
                      width: int = 512, height: int = 288) -> np.ndarray:
        """Render a fixed MuJoCo camera, including the saved real-to-SIM view."""
        if not camera_name or self.model.camera(camera_name).id < 0:
            raise ValueError(f"Unknown camera: {camera_name}")
        key = (camera_name, int(width), int(height))
        renderer = self._camera_renderers.get(key)
        if renderer is None:
            renderer = self.mujoco.Renderer(self.model, height=int(height), width=int(width))
            self._camera_renderers[key] = renderer
        self.mujoco.mj_forward(self.model, self.data)
        renderer.update_scene(self.data, camera=camera_name)
        return np.asarray(renderer.render()).copy()

    def body_xyz(self, name: str):
        self.mujoco.mj_forward(self.model, self.data)
        return tuple(float(value) for value in self.data.xpos[self.object_bodies[name]])

    def body_quat(self, name: str):
        self.mujoco.mj_forward(self.model, self.data)
        return tuple(float(value) for value in self.data.xquat[self.object_bodies[name]])

    def observe(self):
        result = {}
        for name, body_id in self.object_bodies.items():
            xyz = self.body_xyz(name)
            result[name] = {
                "xyz_m": list(xyz), "xy_m": list(xyz[:2]),
                "quat_wxyz": list(self.body_quat(name)),
                "movable": name in self.free_joints,
                "joint_type": "free" if name in self.free_joints else (
                    "articulated" if name in self.articulated_joints else "fixed"),
            }
        return {"objects": result, "bodies": result, "camera": "stage3_camera_0", "image": None}

    def camera_pose(self):
        self.mujoco.mj_forward(self.model, self.data)
        if self.robot_name == "piper":
            optical_id = self.model.body("d435i_color_optical_frame").id
            position = self.data.xpos[optical_id]
            quaternion = self.data.xquat[optical_id]
        else:
            optical_id = self.model.site("d435i_color_optical_frame").id
            position = self.data.site_xpos[optical_id]
            quaternion = np.empty(4)
            self.mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[optical_id])
        if self.camera_calibration is not None:
            camera = self.camera_calibration
            estimate = _apply_delta(Pose3D(tuple(position), tuple(quaternion)),
                                    camera.calibration_position_error_m,
                                    camera.calibration_rotation_error_rad)
            position, quaternion = np.asarray(estimate.position), np.asarray(estimate.quaternion_wxyz)
        return {"xyz_m": position.tolist(), "quaternion_wxyz": quaternion.tolist(),
                "frame": "d435i_color_optical_frame"}

    def state(self):
        robot_state = self.robot.state()
        finger_controls = self.data.ctrl[self.backend._finger_actuators]
        target_width = (finger_controls[0] - finger_controls[1] if self.robot_name == "piper"
                        else sum(finger_controls))
        return {
            "end_effector_pose": {"xyz_m": [float(value) for value in robot_state.pose.position],
                                  "quaternion_wxyz": [float(value) for value in robot_state.pose.quaternion]},
            "camera_pose": self.camera_pose(),
            "gripper": {"width_m": float(robot_state.joints.gripper),
                        "target_width_m": float(target_width)},
            "moving": bool(robot_state.moving),
        }

    def read_rgbd(self):
        return self.robot.camera(width=512, height=288)

    def move_pose(self, pose: Pose3D, duration_s: float | None = None):
        from robot_control.api.types import Pose
        if self.randomizer is not None:
            error = self.randomizer.sample_control_error("move_pose")
            pose = self.randomizer.apply_control_error(pose, error)
        self.robot.move_p(Pose(pose.position, pose.quaternion_wxyz), duration_s=duration_s)
        if duration_s is None:
            self.step(1800 if self.robot_name == "piper" else 3000)
        else:
            self.step(math.ceil((duration_s + 0.45) / self.model.opt.timestep))

    def gripper(self, width: float):
        if not np.isfinite(width) or width < 0 or width > 0.09:
            raise ValueError("Gripper width must be between 0 and 0.09 m")
        if width >= 0.04:
            for name in self.free_joints:
                self.data.eq_active[self.model.eq(f"sim_grasp_{name}").id] = False
            self.mujoco.mj_forward(self.model, self.data)
        self.robot.gripper(float(width))
        self.step(600)
        if width < 0.04:
            fingers = ({"gripper_link1", "gripper_link2"} if self.robot_name == "piper"
                       else {"fr3_leftfinger", "fr3_rightfinger"})
            contacts = {name: set() for name in self.free_joints}
            for contact in self.data.contact[:self.data.ncon]:
                bodies = {self.model.body(int(self.model.geom_bodyid[geom_id])).name
                          for geom_id in (contact.geom1, contact.geom2)}
                for name in contacts.keys() & bodies:
                    contacts[name].update(bodies & fingers)
            if not any(self.data.eq_active[self.model.eq(f"sim_grasp_{name}").id]
                       for name in self.free_joints):
                for name, touching in contacts.items():
                    if touching == fingers:
                        self.grasp(name)
                        break

    def grasp(self, name: str):
        if name not in self.free_joints:
            raise ValueError(f"Object is not a free movable body: {name}")
        finger_names = (("gripper_link1", "gripper_link2") if self.robot_name == "piper"
                        else ("fr3_leftfinger", "fr3_rightfinger"))
        touching = set()
        body_name = name
        for contact in self.data.contact[:self.data.ncon]:
            bodies = {self.model.body(int(self.model.geom_bodyid[geom_id])).name
                      for geom_id in (contact.geom1, contact.geom2)}
            if body_name in bodies:
                touching.update(bodies.intersection(finger_names))
        if touching != set(finger_names):
            raise RuntimeError(f"Both fingers must contact {name} before grasping")
        equality_name = f"sim_grasp_{name}"
        equality_id = self.model.eq(equality_name).id
        parent_name = "gripper_base" if self.robot_name == "piper" else "fr3_hand"
        parent_id = self.model.body(parent_name).id
        object_id = self.object_bodies[name]
        rotation = self.data.xmat[parent_id].reshape(3, 3)
        self.model.eq_data[equality_id, 3:6] = rotation.T @ (self.data.xpos[object_id] - self.data.xpos[parent_id])
        conjugate = np.empty(4)
        relative = np.empty(4)
        self.mujoco.mju_negQuat(conjugate, self.data.xquat[parent_id])
        self.mujoco.mju_mulQuat(relative, conjugate, self.data.xquat[object_id])
        self.model.eq_data[equality_id, 6:10] = relative
        self.data.eq_active[equality_id] = True
        self.mujoco.mj_forward(self.model, self.data)

    def release(self, name: str):
        equality_id = self.model.eq(f"sim_grasp_{name}").id
        self.data.eq_active[equality_id] = False
        self.mujoco.mj_forward(self.model, self.data)
        self.gripper(0.08)

    def set_joint(self, body_name: str, value: float):
        if body_name not in self.articulated_joints:
            raise ValueError(f"Object is not articulated: {body_name}")
        joint_id = self.model.joint(self.articulated_joints[body_name]).id
        value = float(value)
        lower, upper = self.model.jnt_range[joint_id]
        if self.model.jnt_limited[joint_id] and not lower <= value <= upper:
            raise ValueError(f"Joint value {value} is outside [{lower}, {upper}]")
        self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
        self.mujoco.mj_forward(self.model, self.data)
