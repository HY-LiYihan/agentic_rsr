"""MuJoCo runtime adapter for the Piper tabletop vertical slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .randomization import Pose3D


START_JOINTS_RAD = np.deg2rad((0.0, 30.0, -45.0, 0.0, 60.0, 0.0))
DOWN_QUATERNION = (0.0, 0.0, 1.0, 0.0)


class PiperTabletopScene:
    """Own a private backend instance; no backend handles cross the policy API."""

    def __init__(self, scene: Path, object_names: tuple[str, ...],
                 object_aliases: dict[str, str] | None = None,
                 randomizable_objects: tuple[str, ...] | None = None):
        import mujoco
        from robot_control import PiperRobot
        from robot_control.backends.mujoco import MujocoBackend

        self.mujoco = mujoco
        self.backend = MujocoBackend(scene=scene, wrist_camera=True)
        self.backend.connect()
        self.robot = PiperRobot(self.backend)
        model = self.backend.model
        self.object_bodies = {
            model.body(body_id).name.replace("_", " "): model.body(body_id).name
            for body_id in range(model.nbody)
            if model.body(body_id).name.endswith(("_block", "_bowl"))
        }
        if object_aliases:
            for alias, body_name in object_aliases.items():
                model.body(body_name)
                self.object_bodies[alias] = body_name
        if not set(object_names) <= self.object_bodies.keys():
            raise ValueError("Task objects must exist in the Piper tabletop scene")
        self.randomizable_objects = (randomizable_objects if randomizable_objects is not None
                                     else tuple(self.object_bodies))
        if not set(self.randomizable_objects) <= self.object_bodies.keys():
            raise ValueError("Randomizable objects must exist in the scene")
        self.mount_id = self.backend.model.body("d435i_link").id
        self.mount_parent_id = int(self.backend.model.body_parentid[self.mount_id])
        self.nominal_body_pos = self.backend.model.body_pos.copy()
        self.nominal_body_quat = self.backend.model.body_quat.copy()
        self.backend.data.qpos[self.backend._arm_qpos] = START_JOINTS_RAD
        self.backend.data.ctrl[self.backend._arm_actuators] = START_JOINTS_RAD
        mujoco.mj_forward(self.backend.model, self.backend.data)
        self.nominal_qpos = self.backend.data.qpos.copy()
        self.nominal_ctrl = self.backend.data.ctrl.copy()
        self.camera = None

    def close(self) -> None:
        if self.camera is not None:
            self.camera.disconnect()
        self.robot.disconnect()

    def reset_nominal(self) -> None:
        model, data = self.backend.model, self.backend.data
        self.backend._motion = None
        model.body_pos[:] = self.nominal_body_pos
        model.body_quat[:] = self.nominal_body_quat
        self.mujoco.mj_resetData(model, data)
        data.qpos[:] = self.nominal_qpos
        data.ctrl[:] = self.nominal_ctrl
        self.mujoco.mj_forward(model, data)

    def nominal_objects(self) -> dict[str, Pose3D]:
        model, data = self.backend.model, self.backend.data
        result = {}
        for name in self.randomizable_objects:
            body_name = self.object_bodies[name]
            body_id = model.body(body_name).id
            result[name] = Pose3D(tuple(data.xpos[body_id]), tuple(data.xquat[body_id]))
        return result

    def nominal_cameras(self) -> dict[str, Pose3D]:
        model = self.backend.model
        return {"wrist": Pose3D(tuple(model.body_pos[self.mount_id]),
                                tuple(model.body_quat[self.mount_id]))}

    def set_object_pose(self, name: str, pose: Pose3D) -> None:
        model, data = self.backend.model, self.backend.data
        body_id = model.body(self.object_bodies[name]).id
        joint_id = int(model.body_jntadr[body_id])
        if joint_id >= 0 and model.jnt_type[joint_id] == self.mujoco.mjtJoint.mjJNT_FREE:
            address = int(model.jnt_qposadr[joint_id])
            data.qpos[address:address + 7] = (*pose.position, *pose.quaternion_wxyz)
        else:
            model.body_pos[body_id] = pose.position
            model.body_quat[body_id] = pose.quaternion_wxyz
        self.mujoco.mj_forward(model, data)

    def set_camera_pose(self, name: str, pose: Pose3D) -> None:
        if name != "wrist":
            raise KeyError(name)
        self.backend.model.body_pos[self.mount_id] = pose.position
        self.backend.model.body_quat[self.mount_id] = pose.quaternion_wxyz
        self.mujoco.mj_forward(self.backend.model, self.backend.data)

    def move_pose(self, pose: Pose3D) -> None:
        from robot_control.api.types import Pose

        self.robot.move_p(Pose(pose.position, pose.quaternion_wxyz))
        self.robot.step(1200)
        measured = np.asarray(self.robot.state().pose.position)
        if np.linalg.norm(measured - pose.position) > 0.01:
            raise RuntimeError(f"Robot did not reach requested pose: {measured.tolist()}")

    def gripper(self, width: float) -> None:
        if not 0 <= width <= 0.08:
            raise ValueError("Gripper width must be between 0 and 0.08 m")
        self.robot.gripper(width)
        self.robot.step(500)

    def grasp(self, name: str) -> None:
        model, data = self.backend.model, self.backend.data
        body_name = self.object_bodies[name]
        touching = set()
        for contact in data.contact[:data.ncon]:
            bodies = {model.body(int(model.geom_bodyid[geom_id])).name
                      for geom_id in (contact.geom1, contact.geom2)}
            if body_name in bodies:
                touching.update(bodies & {"gripper_link1", "gripper_link2"})
        if touching != {"gripper_link1", "gripper_link2"}:
            raise RuntimeError(f"Both fingers must contact {name} before grasping")
        weld_id = model.eq(f"{body_name}_grasp").id
        parent_id = model.body("gripper_base").id
        object_id = model.body(body_name).id
        rotation = data.xmat[parent_id].reshape(3, 3)
        model.eq_data[weld_id, 3:6] = rotation.T @ (data.xpos[object_id] - data.xpos[parent_id])
        conjugate = np.empty(4)
        relative = np.empty(4)
        self.mujoco.mju_negQuat(conjugate, data.xquat[parent_id])
        self.mujoco.mju_mulQuat(relative, conjugate, data.xquat[object_id])
        model.eq_data[weld_id, 6:10] = relative
        data.eq_active[weld_id] = True
        self.mujoco.mj_forward(model, data)

    def release(self, name: str) -> None:
        self.backend.data.eq_active[self.backend.model.eq(f"{self.object_bodies[name]}_grasp").id] = False
        self.mujoco.mj_forward(self.backend.model, self.backend.data)
        self.gripper(0.08)

    def object_xyz(self, name: str) -> tuple[float, float, float]:
        self.mujoco.mj_forward(self.backend.model, self.backend.data)
        return tuple(float(value) for value in self.backend.data.xpos[
            self.backend.model.body(self.object_bodies[name]).id])

    def read_rgbd(self):
        from robot_control.sensors.mujoco_rgbd import MujocoRGBDCamera

        if self.camera is None:
            self.camera = MujocoRGBDCamera(self.backend.model, self.backend.data,
                                          width=512, height=288)
            self.camera.connect()
        return self.camera.read()

    def estimated_world_from_camera(self, estimated_mount: Pose3D) -> np.ndarray:
        model, data = self.backend.model, self.backend.data
        self.mujoco.mj_forward(model, data)
        parent_rotation = data.xmat[self.mount_parent_id].reshape(3, 3)
        mount_rotation = data.xmat[self.mount_id].reshape(3, 3)
        optical_id = model.body("d435i_color_optical_frame").id
        optical_rotation = data.xmat[optical_id].reshape(3, 3)
        local_rotation = mount_rotation.T @ optical_rotation
        local_translation = mount_rotation.T @ (data.xpos[optical_id] - data.xpos[self.mount_id])
        estimated_rotation = np.empty(9)
        self.mujoco.mju_quat2Mat(estimated_rotation, np.asarray(estimated_mount.quaternion_wxyz))
        world_rotation = parent_rotation @ estimated_rotation.reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :3] = world_rotation @ local_rotation
        transform[:3, 3] = (data.xpos[self.mount_parent_id]
                            + parent_rotation @ np.asarray(estimated_mount.position)
                            + world_rotation @ local_translation)
        return transform
