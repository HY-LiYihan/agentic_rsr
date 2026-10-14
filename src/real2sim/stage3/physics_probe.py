"""Probe and patch MuJoCo scenes produced by Stage 3.

The probe intentionally operates on a complete MJCF scene rather than on the
Blender file. This makes it useful both for Stage 3 exports and for external
MJCF scenes such as the Code as Policies Piper tabletop scene.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_CONTROL_SRC = REPO_ROOT / "external" / "robot_control" / "src"
FREE_JOINT = 0
SLIDE_JOINT = 2
HINGE_JOINT = 3


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("Install the simulator extra with: python -m pip install -e '.[sim]'") from exc
    return mujoco


def _numbers(values: Any) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return array


def _normalize_quaternion(value: Any) -> np.ndarray:
    quaternion = _finite_vector(value, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ValueError("quaternion must be nonzero")
    return quaternion / norm


def _axis_angle_quaternion(value: Any) -> np.ndarray:
    axis_angle = _finite_vector(value, 4, "axis_angle")
    axis = axis_angle[:3]
    angle = float(axis_angle[3])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        if abs(angle) > 1e-9:
            raise ValueError("axis_angle axis must be nonzero for a nonzero rotation")
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / norm
    half = angle / 2.0
    return np.concatenate(([math.cos(half)], axis * math.sin(half)))


def _scene_has_piper_mount(scene: Path) -> bool:
    root = ET.parse(scene).getroot()
    return len(root.findall("./worldbody/body[@name='piper_mount']")) == 1


@dataclass
class SceneHandle:
    model: Any
    data: Any

    def close(self) -> None:
        self.data = None
        self.model = None


def load_scene(scene: Path, attach_piper: str = "auto") -> SceneHandle:
    """Load a complete MJCF or attach Piper to a validated environment scene."""
    mujoco = _mujoco()
    scene = scene.expanduser().resolve()
    if not scene.is_file():
        raise FileNotFoundError(f"MuJoCo scene does not exist: {scene}")
    if attach_piper not in {"auto", "never", "always"}:
        raise ValueError("attach_piper must be auto, never, or always")

    direct_error = None
    if attach_piper != "always":
        try:
            model = mujoco.MjModel.from_xml_path(str(scene))
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stage2_initial")
            if key_id >= 0:
                data = mujoco.MjData(model)
                mujoco.mj_resetDataKeyframe(model, data, key_id)
                mujoco.mj_forward(model, data)
                return SceneHandle(model, data)
            return SceneHandle(model, None)
        except Exception as exc:
            direct_error = exc
            if attach_piper == "never" or not _scene_has_piper_mount(scene):
                raise RuntimeError(f"Could not load scene {scene}: {exc}") from exc

    if attach_piper == "auto" and direct_error is None:
        return SceneHandle(mujoco.MjModel.from_xml_path(str(scene)), None)
    if not ROBOT_CONTROL_SRC.is_dir():
        raise RuntimeError(f"Piper attachment requires robot_control source: {ROBOT_CONTROL_SRC}") from direct_error
    if str(ROBOT_CONTROL_SRC) not in sys.path:
        sys.path.insert(0, str(ROBOT_CONTROL_SRC))
    try:
        from robot_control.backends.piper_model import DEFAULT_MODEL, build_piper_scene
        from robot_control.backends.scene_builder import compile_scene, validate_scene

        model = compile_scene(build_piper_scene(DEFAULT_MODEL), validate_scene(scene))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return SceneHandle(model, data)
    except Exception as exc:
        raise RuntimeError(f"Could not attach Piper to scene {scene}: {exc}") from exc


def _body_id(model: Any, body_name: str) -> int:
    try:
        return int(model.body(body_name).id)
    except Exception as exc:
        raise ValueError(f"Unknown body: {body_name}") from exc


def _body_joint(model: Any, body_id: int) -> tuple[int, int, int] | None:
    joint_id = int(model.body_jntadr[body_id])
    joint_count = int(model.body_jntnum[body_id])
    if joint_id < 0 or joint_count < 1:
        return None
    return joint_id, int(model.jnt_type[joint_id]), int(model.jnt_qposadr[joint_id])


def _robot_body(model: Any, body_id: int) -> bool:
    while body_id > 0:
        if model.body(body_id).name in {"piper_mount", "fr3_mount"}:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _body_state(model: Any, data: Any, body_id: int) -> dict[str, Any]:
    mujoco = _mujoco()
    velocity = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
    return {
        "position": data.xpos[body_id].copy(),
        "quaternion": data.xquat[body_id].copy(),
        "linear_velocity": velocity[3:].copy(),
        "angular_velocity": velocity[:3].copy(),
    }


def _state_json(state: dict[str, Any]) -> dict[str, Any]:
    return {key: np.asarray(value).astype(float).tolist() for key, value in state.items()}


def _rotation_angle(first: Any, second: Any) -> float:
    mujoco = _mujoco()
    relative = np.empty(4, dtype=float)
    conjugate = np.empty(4, dtype=float)
    mujoco.mju_negQuat(conjugate, _normalize_quaternion(first))
    mujoco.mju_mulQuat(relative, _normalize_quaternion(second), conjugate)
    return float(2.0 * math.acos(min(1.0, abs(float(relative[0])))))


def _body_info(model: Any, body_id: int) -> dict[str, Any]:
    joint = _body_joint(model, body_id)
    geom_ids = [index for index in range(model.ngeom) if int(model.geom_bodyid[index]) == body_id]
    return {
        "name": model.body(body_id).name,
        "id": body_id,
        "mass_kg": float(model.body_mass[body_id]),
        "joint": None if joint is None else {
            "id": joint[0], "type": int(joint[1]), "qpos_address": joint[2],
        },
        "free_joint": bool(joint is not None and joint[1] == FREE_JOINT),
        "movable": bool(joint is not None and joint[1] in {FREE_JOINT, SLIDE_JOINT, HINGE_JOINT}),
        "robot_body": _robot_body(model, body_id),
        "geoms": [
            {
                "id": geom_id,
                "name": model.geom(geom_id).name,
                "type": int(model.geom_type[geom_id]),
                "friction": model.geom_friction[geom_id].astype(float).tolist(),
            }
            for geom_id in geom_ids
        ],
    }


def inspect_scene(handle: SceneHandle) -> dict[str, Any]:
    model = handle.model
    bodies = [_body_info(model, body_id) for body_id in range(1, model.nbody)]
    return {
        "bodies": bodies,
        "body_count": len(bodies),
        "geom_count": int(model.ngeom),
        "joint_count": int(model.njnt),
        "actuator_count": int(model.nu),
        "timestep": float(model.opt.timestep),
        "gravity": model.opt.gravity.astype(float).tolist(),
    }


def task_objects(scene: Path, attach_piper: str = "auto", settle: float = 0.1) -> dict[str, Any]:
    """Return a compact list of task bodies rather than the entire robot tree."""
    handle = load_scene(scene, attach_piper=attach_piper)
    try:
        mujoco = _mujoco()
        model = handle.model
        data = handle.data if handle.data is not None else mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        if not np.isfinite(settle) or settle < 0:
            raise ValueError("settle must be finite and nonnegative")
        for _ in range(int(round(settle / model.opt.timestep))):
            mujoco.mj_step(model, data)
        objects = []
        for body_id in range(1, handle.model.nbody):
            item = _body_info(handle.model, body_id)
            if item["robot_body"] or not item["geoms"]:
                continue
            joint_type = None if item["joint"] is None else item["joint"]["type"]
            objects.append({
                "body": item["name"],
                "joint": {FREE_JOINT: "free", SLIDE_JOINT: "slide",
                          HINGE_JOINT: "hinge"}.get(joint_type, "fixed"),
                "position_m": data.xpos[body_id].astype(float).tolist(),
                "quaternion_wxyz": data.xquat[body_id].astype(float).tolist(),
                "mass_kg": item["mass_kg"],
                "friction": [geom["friction"] for geom in item["geoms"]],
            })
        return {"scene": str(scene.expanduser().resolve()), "objects": objects}
    finally:
        handle.close()


def _overview_image(scene: Path, output: Path, settle: float, attach_piper: str) -> Path:
    executable = Path(sys.executable).with_name("mjpython") if sys.platform == "darwin" else Path(sys.executable)
    if not executable.is_file():
        raise FileNotFoundError(f"MuJoCo rendering requires the matching mjpython: {executable}")
    env = os.environ.copy()
    source = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([str(executable), "-m", "real2sim.stage3.render_overview",
                             "--scene", str(scene), "--output", str(output),
                             "--settle", str(settle), "--attach-piper", attach_piper],
                            capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(f"third-person rendering failed: {result.stderr.strip()}")
    return output.expanduser().resolve()


def observe_scene(scene: Path, output: Path, settle: float = 0.1,
                  attach_piper: str = "auto") -> dict[str, Any]:
    """Record object poses, parameters and an independent overview camera."""
    output = output.expanduser().resolve()
    observation = task_objects(scene, attach_piper=attach_piper, settle=settle)
    observation["camera"] = "third_person_free_camera_not_stage3_reference"
    observation["image"] = str(_overview_image(scene, output / "overview.png", settle, attach_piper))
    output.mkdir(parents=True, exist_ok=True)
    (output / "observation.json").write_text(json.dumps(observation, indent=2) + "\n", encoding="utf-8")
    return observation


def check_static(scene: Path, output: Path | None = None, duration: float = 3.0,
                 settle: float = 0.1, max_drift_m: float = 0.01,
                 max_rotation_rad: float = 0.1) -> dict[str, Any]:
    """Detect spontaneous movement after settling, without external force."""
    if not np.isfinite([duration, settle]).all() or duration <= 0 or settle < 0:
        raise ValueError("duration must be finite and positive; settle must be finite and nonnegative")
    if not np.isfinite([max_drift_m, max_rotation_rad]).all() or min(max_drift_m, max_rotation_rad) < 0:
        raise ValueError("static tolerances must be finite and nonnegative")
    initial = {item["body"]: item for item in task_objects(scene, settle=settle)["objects"]}
    final = {item["body"]: item for item in task_objects(scene, settle=settle + duration)["objects"]}
    rows = []
    for name, item in initial.items():
        if item["joint"] == "fixed":
            continue
        other = final[name]
        drift = float(np.linalg.norm(np.asarray(item["position_m"]) - other["position_m"]))
        rotation = _rotation_angle(item["quaternion_wxyz"], other["quaternion_wxyz"])
        rows.append({"body": name, "drift_m": drift, "rotation_rad": rotation,
                     "passed": bool(drift <= max_drift_m and rotation <= max_rotation_rad)})
    report = {"scene": str(scene.expanduser().resolve()), "duration_s": duration,
              "settle_s": settle, "max_drift_m": max_drift_m,
              "max_rotation_rad": max_rotation_rad, "objects": rows,
              "passed": bool(rows and all(item["passed"] for item in rows))}
    if output is not None:
        _write_report(report, output, print_stdout=False)
    return report


def _rigid_target_bodies(model: Any, target: int) -> set[int]:
    bodies = {target}
    for body_id in range(target + 1, model.nbody):
        if int(model.body_parentid[body_id]) in bodies and not model.body_jntnum[body_id]:
            bodies.add(body_id)
    return bodies


def _grasp_geometry(scene: Path, body: str, settle: float, jaw_friction: float,
                    jaw_size: float, attach_piper: str, grasp_axis: str = "x",
                    jaw_damping: float = 2.0) -> ET.Element:
    if grasp_axis not in {"x", "y"}:
        raise ValueError("grasp axis must be x or y")
    if not np.isfinite(jaw_damping) or jaw_damping < 0:
        raise ValueError("jaw damping must be finite and nonnegative")
    mujoco = _mujoco()
    handle = load_scene(scene, attach_piper)
    try:
        model = handle.model
        target = _body_id(model, body)
        joint = _body_joint(model, target)
        if _robot_body(model, target) or joint is None or joint[1] != FREE_JOINT:
            raise ValueError(f"{body} needs a non-robot free joint for grasp testing")
        data = handle.data if handle.data is not None else mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for _ in range(int(round(settle / model.opt.timestep))):
            mujoco.mj_step(model, data)
        rigid_bodies = _rigid_target_bodies(model, target)
        geom_ids = [index for index in range(model.ngeom) if int(model.geom_bodyid[index]) in rigid_bodies
                    and model.geom_contype[index] and model.geom_conaffinity[index]]
        if not geom_ids:
            raise ValueError(f"{body} has no collidable rigid geometry")
        lower_bounds = []
        upper_bounds = []
        for geom_id in geom_ids:
            geom_type = int(model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                extent = abs(data.geom_xmat[geom_id].reshape(3, 3)) @ model.geom_size[geom_id]
            else:
                extent = model.geom_rbound[geom_id]
            lower_bounds.append(data.geom_xpos[geom_id] - extent)
            upper_bounds.append(data.geom_xpos[geom_id] + extent)
        lower = np.min(lower_bounds, axis=0)
        upper = np.max(upper_bounds, axis=0)
        center = (lower + upper) / 2
        half_extent = (upper - lower) / 2
    finally:
        handle.close()

    root = ET.parse(scene).getroot()
    target_body = _find_body_element(root, body)
    target_names = []
    def collect_colliders(node: ET.Element) -> None:
        for geom in node.findall("geom"):
            if int(geom.get("contype", "1")) and int(geom.get("conaffinity", "1")):
                geom_name = geom.get("name") or f"stage3_probe_target_geom_{len(target_names)}"
                geom.set("name", geom_name)
                target_names.append(geom_name)
        for child in node.findall("body"):
            if child.find("joint") is None and child.find("freejoint") is None:
                collect_colliders(child)
    collect_colliders(target_body)
    if not target_names:
        raise ValueError(f"{body} has no collidable rigid geometry")
    world = root.find("worldbody")
    if world is None:
        raise ValueError("scene has no worldbody")
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    axis_index = 0 if grasp_axis == "x" else 1
    axis_vector = np.eye(3)[axis_index]
    pad_size = np.maximum(half_extent * 0.9, jaw_size)
    pad_size[axis_index] = jaw_size
    for side, sign in (("left", -1), ("right", 1)):
        jaw = ET.SubElement(world, "body", {"name": f"stage3_probe_jaw_{side}",
                                             "pos": _numbers(center + sign * (half_extent[axis_index] + jaw_size + 0.004) * axis_vector),
                                             "gravcomp": "1"})
        close_joint = {"name": f"stage3_probe_close_{side}", "type": "slide",
                       "axis": _numbers(-sign * axis_vector), "damping": f"{jaw_damping:.9g}",
                       "limited": "true", "range": "0 0.05",
                       "actuatorfrclimited": "true", "actuatorfrcrange": "-10 10"}
        ET.SubElement(jaw, "joint", close_joint)
        ET.SubElement(jaw, "joint", {"name": f"stage3_probe_raise_{side}",
                                      "type": "slide", "axis": "0 0 1", "damping": "2"})
        ET.SubElement(jaw, "geom", {"name": f"stage3_probe_pad_{side}", "type": "box",
                                     "size": _numbers(pad_size),
                                     "friction": _numbers([jaw_friction, 0.005, 0.0001]),
                                     "rgba": "0.3 0.3 0.3 1", "mass": "0.025"})
        ET.SubElement(actuator, "position", {"name": f"stage3_probe_close_motor_{side}",
                                             "joint": f"stage3_probe_close_{side}",
                                             "kp": "200", "ctrlrange": "0 0.05",
                                             "forcerange": "-10 10"})
        ET.SubElement(actuator, "position", {"name": f"stage3_probe_raise_motor_{side}",
                                             "joint": f"stage3_probe_raise_{side}",
                                             "kp": "100", "kv": "2"})
        for target_name in target_names:
            ET.SubElement(contact, "pair", {"geom1": f"stage3_probe_pad_{side}",
                                            "geom2": target_name,
                                            "condim": "3",
                                            "friction": _numbers([jaw_friction, jaw_friction,
                                                                  0.005, 0.0001, 0.0001])})
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "joint", {"name": "stage3_probe_mimic",
                                       "joint1": "stage3_probe_close_right",
                                       "joint2": "stage3_probe_close_left",
                                       "polycoef": "0 1 0 0 0"})
    initial = root.find("./keyframe/key[@name='stage2_initial']")
    if initial is not None:
        for field in ("qpos", "ctrl"):
            initial.set(field, initial.get(field, "") + " 0 0 0 0")
    return root


def check_grasp(scene: Path, body: str, normal_forces: list[float], output: Path | None = None,
                duration: float = 0.4, settle: float = 0.1, min_lift_m: float = 0.01,
                jaw_friction: float = 1.0, jaw_size: float = 0.006,
                attach_piper: str = "auto", grasp_axis: str = "x",
                jaw_damping: float = 2.0) -> dict[str, Any]:
    """Close symmetric position-controlled pads with at most 10 N per finger."""
    values = [*normal_forces, duration, settle, min_lift_m, jaw_friction, jaw_size]
    if not normal_forces or not np.isfinite(values).all() or any(force <= 0 or force > 10 for force in normal_forces):
        raise ValueError("per-finger actuator force limits must be finite and in (0, 10] N")
    if duration <= 0 or settle < 0 or min_lift_m < 0 or jaw_friction < 0 or jaw_size <= 0:
        raise ValueError("duration and jaw size must be positive; tolerances nonnegative")
    mujoco = _mujoco()
    root = _grasp_geometry(scene, body, settle, jaw_friction, jaw_size,
                           attach_piper, grasp_axis, jaw_damping)
    with tempfile.TemporaryDirectory(prefix="stage3_grasp_") as directory:
        augmented = Path(directory) / "grasp.xml"
        _rebase_scene_assets(root, scene, augmented)
        ET.ElementTree(root).write(augmented, encoding="utf-8")
        handle = load_scene(augmented, attach_piper)
        try:
            model = handle.model
            target = _body_id(model, body)
            pads = tuple(int(model.geom(f"stage3_probe_pad_{side}").id)
                         for side in ("left", "right"))
            jaw_bodies = tuple(int(model.body(f"stage3_probe_jaw_{side}").id)
                               for side in ("left", "right"))
            close_motors = tuple(int(model.actuator(f"stage3_probe_close_motor_{side}").id)
                                 for side in ("left", "right"))
            raise_motors = tuple(int(model.actuator(f"stage3_probe_raise_motor_{side}").id)
                                 for side in ("left", "right"))
            rigid_bodies = _rigid_target_bodies(model, target)
            target_geoms = {index for index in range(model.ngeom)
                            if int(model.geom_bodyid[index]) in rigid_bodies
                            and model.geom_contype[index] and model.geom_conaffinity[index]}
            results = []
            for normal_force in normal_forces:
                for close_motor in close_motors:
                    model.actuator_forcerange[close_motor] = [-normal_force, normal_force]
                data = mujoco.MjData(model)
                key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stage2_initial")
                if key_id >= 0:
                    mujoco.mj_resetDataKeyframe(model, data, key_id)
                mujoco.mj_forward(model, data)
                for _ in range(int(round(settle / model.opt.timestep))):
                    mujoco.mj_step(model, data)
                initial_z = float(data.xpos[target, 2])
                initial_jaw_z = float(np.mean(data.xpos[list(jaw_bodies), 2]))
                close_steps = max(1, int(round(0.25 / model.opt.timestep)))
                lift_steps = max(1, int(round(duration / model.opt.timestep)))
                pad_normal = [0.0, 0.0]
                contact_steps = 0
                peak_normal = [0.0, 0.0]
                for step in range(close_steps + lift_steps):
                    for close_motor in close_motors:
                        data.ctrl[close_motor] = 0.05
                    for raise_motor in raise_motors:
                        data.ctrl[raise_motor] = (0.08 * (step - close_steps + 1) / lift_steps
                                                  if step >= close_steps else 0)
                    mujoco.mj_step(model, data)
                    pad_normal = [0.0, 0.0]
                    for contact_id in range(data.ncon):
                        contact = data.contact[contact_id]
                        for index, pad in enumerate(pads):
                            if (contact.geom1 == pad and contact.geom2 in target_geoms or
                                    contact.geom2 == pad and contact.geom1 in target_geoms):
                                force = np.zeros(6)
                                mujoco.mj_contactForce(model, data, contact_id, force)
                                pad_normal[index] += abs(float(force[0]))
                    peak_normal = [max(peak_normal[index], value)
                                   for index, value in enumerate(pad_normal)]
                    if step >= close_steps and all(value > 1e-6 for value in pad_normal):
                        contact_steps += 1
                rise = float(data.xpos[target, 2] - initial_z)
                jaw_rise = float(np.mean(data.xpos[list(jaw_bodies), 2]) - initial_jaw_z)
                held_at_end = all(value > 1e-6 for value in pad_normal)
                success = bool(np.isfinite(data.qpos).all() and rise >= min_lift_m and
                               contact_steps >= lift_steps * 0.5 and held_at_end)
                results.append({"target_normal_force_per_jaw_n": normal_force,
                                "peak_normal_force_per_jaw_n": peak_normal,
                                "rise_m": rise, "jaw_rise_m": jaw_rise,
                                "slip_relative_to_jaws_m": jaw_rise - rise,
                                "two_sided_contact_fraction": contact_steps / lift_steps,
                                "held_at_end": held_at_end,
                                "lifted": success})
        finally:
            handle.close()
    successes = [row["target_normal_force_per_jaw_n"] for row in results if row["lifted"]]
    report = {"scene": str(scene.expanduser().resolve()), "body": body,
              "method": "two_frictional_actuated_box_jaws",
              "control_mode": "symmetric_position_servo",
              "max_actuator_force_per_jaw_n": 10.0,
              "jaw_mass_kg": 0.025,
              "duration_s": duration, "settle_s": settle, "min_lift_m": min_lift_m,
              "close_duration_s": 0.25, "requested_jaw_lift_m": 0.08,
              "jaw_friction": jaw_friction, "jaw_size_m": jaw_size,
              "grasp_axis": grasp_axis,
              "jaw_damping": jaw_damping,
              "trials": results,
              "lowest_tested_grasp_normal_force_per_jaw_n": min(successes) if successes else None,
              "passed": bool(successes)}
    if output is not None:
        _write_report(report, output, print_stdout=False)
    return report


def _action_vector(action: dict[str, Any]) -> np.ndarray:
    value = action.get("vector", action.get("value"))
    return _finite_vector(value, 3, f"{action.get('type', 'action')} vector")


def _action_window(action: dict[str, Any], duration: float) -> tuple[float, float]:
    start = float(action.get("start", action.get("at", 0.0)))
    if not np.isfinite(start) or start < 0 or start >= duration:
        raise ValueError(f"action start/at must be within [0, {duration})")
    if "end" in action:
        end = float(action["end"])
    else:
        end = start + float(action.get("duration", duration - start))
    if not np.isfinite(end) or end <= start:
        raise ValueError("action end must be finite and later than start")
    return start, min(end, duration)


def _free_qpos_qvel(model: Any, body_id: int) -> tuple[int, int]:
    joint = _body_joint(model, body_id)
    if joint is None or joint[1] != FREE_JOINT:
        raise ValueError(f"Body {model.body(body_id).name} does not have a free joint")
    return joint[2], int(model.jnt_dofadr[joint[0]])


def _world_vector(model: Any, data: Any, body_id: int, vector: np.ndarray, frame: str) -> np.ndarray:
    if frame == "world":
        return vector
    if frame != "body":
        raise ValueError("action frame must be world or body")
    return data.xmat[body_id].reshape(3, 3) @ vector


def _apply_one_shot(model: Any, data: Any, action: dict[str, Any]) -> None:
    action_type = str(action.get("type", "")).lower()
    body_id = _body_id(model, str(action["body"]))
    qpos_address, qvel_address = _free_qpos_qvel(model, body_id)
    if action_type in {"set_position", "position"}:
        data.qpos[qpos_address:qpos_address + 3] = _action_vector(action)
    elif action_type in {"set_linear_velocity", "linear_velocity"}:
        data.qvel[qvel_address:qvel_address + 3] = _action_vector(action)
    elif action_type in {"set_angular_velocity", "angular_velocity"}:
        data.qvel[qvel_address + 3:qvel_address + 6] = _action_vector(action)
    elif action_type in {"set_rotation", "rotation"}:
        value = action.get("quaternion")
        if value is None:
            value = _axis_angle_quaternion(action["axis_angle"])
        data.qpos[qpos_address + 3:qpos_address + 7] = _normalize_quaternion(value)
    elif action_type in {"rotate", "relative_rotation"}:
        value = action.get("quaternion")
        if value is None:
            value = _axis_angle_quaternion(action["axis_angle"])
        target = np.empty(4, dtype=float)
        _mujoco().mju_mulQuat(target, _normalize_quaternion(value), data.qpos[qpos_address + 3:qpos_address + 7])
        data.qpos[qpos_address + 3:qpos_address + 7] = _normalize_quaternion(target)
    else:
        raise ValueError(f"Unsupported one-shot action type: {action_type}")
    _mujoco().mj_forward(model, data)


def _apply_wrench(model: Any, data: Any, action: dict[str, Any]) -> None:
    body_id = _body_id(model, str(action["body"]))
    frame = str(action.get("frame", "world"))
    action_type = str(action.get("type", "")).lower()
    if action_type == "force":
        vector = _world_vector(model, data, body_id, _action_vector(action), frame)
        data.xfrc_applied[body_id, :3] += vector
    elif action_type == "torque":
        vector = _world_vector(model, data, body_id, _action_vector(action), frame)
        data.xfrc_applied[body_id, 3:] += vector
    elif action_type in {"wrench", "force_torque"}:
        force = _world_vector(model, data, body_id, _finite_vector(action["force"], 3, "force"), frame)
        torque = _world_vector(model, data, body_id, _finite_vector(action["torque"], 3, "torque"), frame)
        data.xfrc_applied[body_id, :3] += force
        data.xfrc_applied[body_id, 3:] += torque
    else:
        raise ValueError(f"Unsupported continuous action type: {action_type}")


def _validate_actions(model: Any, actions: list[dict[str, Any]], duration: float) -> None:
    continuous = {"force", "torque", "wrench", "force_torque"}
    one_shot = {
        "set_position", "position", "set_linear_velocity", "linear_velocity",
        "set_angular_velocity", "angular_velocity", "set_rotation", "rotation",
        "rotate", "relative_rotation",
    }
    for action in actions:
        if not isinstance(action, dict) or "body" not in action or "type" not in action:
            raise ValueError("each action needs body and type")
        action_type = str(action["type"]).lower()
        body_id = _body_id(model, str(action["body"]))
        if action_type in continuous:
            _action_window(action, duration)
            if action.get("frame", "world") not in {"world", "body"}:
                raise ValueError("action frame must be world or body")
            if action_type == "wrench" or action_type == "force_torque":
                _finite_vector(action.get("force"), 3, "force")
                _finite_vector(action.get("torque"), 3, "torque")
            else:
                _action_vector(action)
        elif action_type in one_shot:
            _free_qpos_qvel(model, body_id)
            at = float(action.get("at", action.get("start", 0.0)))
            if not np.isfinite(at) or at < 0 or at >= duration:
                raise ValueError(f"one-shot action time must be within [0, {duration})")
            if action_type in {"set_rotation", "rotation", "rotate", "relative_rotation"}:
                value = action.get("quaternion")
                if value is None:
                    _axis_angle_quaternion(action["axis_angle"])
                else:
                    _normalize_quaternion(value)
            else:
                _action_vector(action)
        else:
            raise ValueError(f"Unsupported action type: {action_type}")


def _action_active(action: dict[str, Any], time: float, duration: float) -> bool:
    start, end = _action_window(action, duration)
    return start <= time < end or (start == end and abs(time - start) < 1e-12)


def _check_results(report: dict[str, Any], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    responses = report["responses"]
    for check in checks:
        body = str(check["body"])
        response = responses.get(body)
        if response is None:
            raise ValueError(f"check references unknown body: {body}")
        check_type = str(check.get("type", "move")).lower()
        if check_type in {"move", "displacement"}:
            value = float(response["displacement_m"])
            threshold = float(check.get("min_m", check.get("min_displacement_m", 0.001)))
            maximum = float(check["max_m"]) if "max_m" in check else None
            passed = value >= threshold and (maximum is None or value <= maximum)
            measured = value
        elif check_type in {"rotate", "rotation"}:
            value = float(response["rotation_rad"])
            threshold = float(check.get("min_rad", check.get("min_rotation_rad", 0.05)))
            maximum = float(check["max_rad"]) if "max_rad" in check else None
            passed = value >= threshold and (maximum is None or value <= maximum)
            measured = value
        elif check_type in {"stationary", "stay"}:
            value = float(response["displacement_m"])
            threshold = float(check.get("max_m", check.get("max_displacement_m", 0.001)))
            maximum = threshold
            passed = value <= threshold
            measured = value
        else:
            raise ValueError(f"Unsupported check type: {check_type}")
        if not np.isfinite(threshold) or (maximum is not None and not np.isfinite(maximum)):
            raise ValueError(f"check thresholds for {body} must be finite")
        results.append({"body": body, "type": check_type, "measured": measured,
                        "threshold": threshold, "max_threshold": maximum,
                        "passed": bool(passed)})
    return results


def run_probe(scene: Path, script: dict[str, Any], attach_piper: str = "auto") -> dict[str, Any]:
    mujoco = _mujoco()
    handle = load_scene(scene, attach_piper=attach_piper)
    try:
        model = handle.model
        data = handle.data if handle.data is not None else mujoco.MjData(model)
        duration = float(script.get("duration", 1.0))
        settle = float(script.get("settle", 0.2))
        if not np.isfinite(duration) or not np.isfinite(settle) or duration <= 0 or settle < 0:
            raise ValueError("duration must be finite and positive; settle must be finite and nonnegative")
        if "timestep" in script:
            timestep = float(script["timestep"])
            if timestep <= 0 or not np.isfinite(timestep):
                raise ValueError("timestep must be positive and finite")
            model.opt.timestep = timestep
        actions = list(script.get("actions", []))
        checks = list(script.get("checks", []))
        _validate_actions(model, actions, duration)
        mujoco.mj_forward(model, data)
        for _ in range(int(math.ceil(settle / model.opt.timestep))):
            mujoco.mj_step(model, data)
        initial = {model.body(body_id).name: _body_state(model, data, body_id)
                   for body_id in range(1, model.nbody)}
        sample_interval = float(script.get("sample_interval", model.opt.timestep * 10))
        if not np.isfinite(sample_interval) or sample_interval <= 0:
            raise ValueError("sample_interval must be positive and finite")
        sample_interval = max(sample_interval, model.opt.timestep)
        sample_next = 0.0
        samples = []
        max_linear_speed = 0.0
        max_angular_speed = 0.0
        max_force = 0.0
        max_torque = 0.0
        finite = True
        fired: set[int] = set()
        steps = int(math.ceil(duration / model.opt.timestep))
        for step in range(steps):
            time = step * float(model.opt.timestep)
            data.xfrc_applied[:] = 0
            for index, action in enumerate(actions):
                action_type = str(action["type"]).lower()
                if action_type in {"force", "torque", "wrench", "force_torque"}:
                    if _action_active(action, time, duration):
                        _apply_wrench(model, data, action)
                elif index not in fired and time >= float(action.get("at", action.get("start", 0.0))):
                    _apply_one_shot(model, data, action)
                    fired.add(index)
            max_force = max(max_force, float(np.max(np.linalg.norm(data.xfrc_applied[:, :3], axis=1))))
            max_torque = max(max_torque, float(np.max(np.linalg.norm(data.xfrc_applied[:, 3:], axis=1))))
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                finite = False
                break
            if time + float(model.opt.timestep) >= sample_next:
                snapshot = {
                    model.body(body_id).name: _state_json(_body_state(model, data, body_id))
                    for body_id in range(1, model.nbody)
                }
                samples.append({"time": min(duration, time + float(model.opt.timestep)), "bodies": snapshot})
                sample_next += sample_interval
            for body_id in range(1, model.nbody):
                state = _body_state(model, data, body_id)
                max_linear_speed = max(max_linear_speed, float(np.linalg.norm(state["linear_velocity"])))
                max_angular_speed = max(max_angular_speed, float(np.linalg.norm(state["angular_velocity"])))
        final = {model.body(body_id).name: _body_state(model, data, body_id)
                 for body_id in range(1, model.nbody)}
        responses = {}
        for name, start in initial.items():
            end = final[name]
            responses[name] = {
                "displacement_m": float(np.linalg.norm(end["position"] - start["position"])),
                "rotation_rad": _rotation_angle(start["quaternion"], end["quaternion"]),
                "final_position": end["position"].astype(float).tolist(),
                "final_quaternion": end["quaternion"].astype(float).tolist(),
                "final_linear_velocity": end["linear_velocity"].astype(float).tolist(),
                "final_angular_velocity": end["angular_velocity"].astype(float).tolist(),
            }
        report = {
            "scene": str(scene.expanduser().resolve()),
            "duration_s": duration,
            "timestep_s": float(model.opt.timestep),
            "steps": steps,
            "settle_s": settle,
            "stability": {
                "finite_state": bool(finite),
                "max_linear_speed_m_s": max_linear_speed,
                "max_angular_speed_rad_s": max_angular_speed,
                "max_applied_force_n": max_force,
                "max_applied_torque_nm": max_torque,
            },
            "responses": responses,
            "checks": _check_results({"responses": responses}, checks),
            "samples": samples,
        }
        report["passed"] = bool(
            report["stability"]["finite_state"]
            and all(item["passed"] for item in report["checks"])
        )
        return report
    finally:
        handle.close()


def auto_probe(scene: Path, duration: float = 0.2, settle: float = 0.1,
               force_n: float = 0.02, torque_nm: float = 0.001,
               min_displacement_m: float = 1e-6, min_rotation_rad: float = 1e-4,
               attach_piper: str = "auto", body: str | None = None) -> dict[str, Any]:
    """Run the same small perturbation independently on each task joint."""
    handle = load_scene(scene, attach_piper=attach_piper)
    try:
        bodies = inspect_scene(handle)["bodies"]
        axes = {item["name"]: handle.model.jnt_axis[item["joint"]["id"]].astype(float).tolist()
                for item in bodies if item["joint"] is not None
                and item["joint"]["type"] in {SLIDE_JOINT, HINGE_JOINT}}
    finally:
        handle.close()
    if duration <= 0.02 or not np.isfinite(duration):
        raise ValueError("auto probe duration must be finite and greater than 0.02 s")
    movable = [item for item in bodies if item["movable"] and not item["robot_body"]]
    if body is not None:
        movable = [item for item in movable if item["name"] == body]
        if not movable:
            raise ValueError(f"Body {body} is not a movable task object in {scene}")
    static = [item["name"] for item in bodies if not item["movable"] and not item["robot_body"]]
    robot = [item["name"] for item in bodies if item["robot_body"]]
    if not np.isfinite([force_n, torque_nm, min_displacement_m, min_rotation_rad]).all():
        raise ValueError("probe magnitudes and response thresholds must be finite")
    if force_n < 0 or torque_nm < 0 or min_displacement_m < 0 or min_rotation_rad < 0:
        raise ValueError("probe magnitudes and response thresholds must be nonnegative")
    baseline = run_probe(scene, {"duration": duration, "settle": settle, "sample_interval": duration},
                         attach_piper=attach_piper)
    results = []
    for item in movable:
        body_name = item["name"]
        joint_type = item["joint"]["type"]
        if joint_type == FREE_JOINT:
            actions = [
                {"body": body_name, "type": "force", "start": 0.02,
                 "duration": duration * 0.4, "vector": [force_n, 0, 0]},
                {"body": body_name, "type": "torque", "start": 0.02,
                 "duration": duration * 0.4, "vector": [0, 0, torque_nm]},
            ]
        else:
            axis = axes[body_name]
            actions = [{"body": body_name,
                        "type": "force" if joint_type == SLIDE_JOINT else "torque",
                        "start": 0.02, "duration": duration * 0.4,
                        "frame": "body",
                        "vector": (np.asarray(axis) * (force_n if joint_type == SLIDE_JOINT else torque_nm)).tolist()}]
        report = run_probe(scene, {
            "duration": duration,
            "settle": settle, "sample_interval": duration,
            "actions": actions,
        }, attach_piper=attach_piper)
        response = report["responses"][body_name]
        baseline_response = baseline["responses"][body_name]
        displacement = float(np.linalg.norm(np.asarray(response["final_position"])
                                            - np.asarray(baseline_response["final_position"])))
        rotation = _rotation_angle(baseline_response["final_quaternion"], response["final_quaternion"])
        responded = (displacement >= min_displacement_m or rotation >= min_rotation_rad)
        results.append({
            "body": body_name,
            "joint_type": joint_type,
            "passed": bool(baseline["stability"]["finite_state"]
                           and report["stability"]["finite_state"] and responded),
            "responded": bool(responded),
            "response_vs_baseline": {"displacement_m": displacement, "rotation_rad": rotation},
            "stability": report["stability"],
            "response": response,
        })
    return {
        "scene": str(scene.expanduser().resolve()),
        "mode": "auto_response_sweep",
        "tested_bodies": results,
        "skipped_fixed_bodies": static,
        "skipped_robot_bodies": robot,
        "passed": bool(movable and all(item["passed"] for item in results)),
        "probe": {
            "duration_s": duration,
            "settle_s": settle,
            "force_n": force_n,
            "torque_nm": torque_nm,
            "min_displacement_m": min_displacement_m,
            "min_rotation_rad": min_rotation_rad,
        },
    }


def try_parameters(scene: Path, body: str, output: Path,
                   mass_kg: float | None = None, friction: float | None = None,
                   damping: float | None = None, duration: float = 0.2,
                   settle: float = 0.1, force_n: float = 0.02,
                   torque_nm: float = 0.001, attach_piper: str = "auto") -> dict[str, Any]:
    """Compare one candidate against the original using an identical perturbation."""
    changes = {key: value for key, value in (("mass_kg", mass_kg), ("friction", friction),
                                            ("damping", damping)) if value is not None}
    if not changes:
        raise ValueError("try requires --mass-kg, --friction, or --damping")
    output = output.expanduser().resolve()
    candidate_scene = output / "candidate.xml"
    if scene.expanduser().resolve() == candidate_scene:
        raise ValueError("candidate scene must differ from the original scene")
    original_objects = {item["body"]: item for item in task_objects(scene, attach_piper)["objects"]}
    if body not in original_objects or original_objects[body]["joint"] == "fixed":
        raise ValueError(f"Body {body} is not a movable task object in {scene}")
    original = auto_probe(scene, duration, settle, force_n, torque_nm,
                          attach_piper=attach_piper, body=body)
    patch = patch_scene(scene, candidate_scene, {body: changes}, attach_piper)
    candidate = auto_probe(candidate_scene, duration, settle, force_n, torque_nm,
                           attach_piper=attach_piper, body=body)
    candidate_objects = {item["body"]: item for item in task_objects(candidate_scene, attach_piper)["objects"]}
    before = original["tested_bodies"][0]
    after = candidate["tested_bodies"][0]
    report = {
        "scene": str(scene.expanduser().resolve()),
        "candidate_scene": str(candidate_scene),
        "body": body,
        "probe": original["probe"],
        "patch": patch["changed"],
        "before": {"parameters": original_objects[body],
                   "response_vs_baseline": before["response_vs_baseline"],
                   "passed": before["passed"]},
        "after": {"parameters": candidate_objects[body],
                  "response_vs_baseline": after["response_vs_baseline"],
                  "passed": after["passed"]},
        "calibration": "unverified_without_measured_target",
        "passed": bool(before["passed"] and after["passed"]),
    }
    _write_report(report, output, print_stdout=False)
    return report


def _find_body_element(root: ET.Element, body_name: str) -> ET.Element:
    matches = [body for body in root.findall(".//body") if body.get("name") == body_name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one XML body named {body_name}, found {len(matches)}")
    return matches[0]


def _friction_value(value: Any, current: Any) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        values = (float(value), float(current[1]), float(current[2]))
    else:
        values = tuple(float(item) for item in value)
        if len(values) == 1:
            values = (values[0], float(current[1]), float(current[2]))
    if len(values) != 3 or not np.isfinite(values).all() or any(item < 0 for item in values):
        raise ValueError("friction must be one or three finite nonnegative values")
    return values


def _rebase_scene_assets(root: ET.Element, scene: Path, output: Path) -> None:
    if scene.expanduser().resolve().parent == output.expanduser().resolve().parent:
        return
    source_dir = scene.expanduser().resolve().parent
    compiler = root.find("compiler")
    asset_dir = source_dir
    if compiler is not None:
        configured = compiler.get("assetdir")
        if configured:
            asset_dir = (source_dir / configured).resolve()
            compiler.set("assetdir", str(asset_dir))
        for name in ("meshdir", "texturedir"):
            configured = compiler.get(name)
            if configured:
                compiler.set(name, str((asset_dir / configured).resolve()))
    for include in root.iter("include"):
        source = include.get("file")
        if source:
            include.set("file", str((source_dir / source).resolve()))
    for asset in root.findall("./asset/*"):
        source = asset.get("file")
        if not source:
            continue
        base = asset_dir
        if compiler is not None:
            if asset.tag == "mesh" and compiler.get("meshdir"):
                base = Path(compiler.get("meshdir"))
            elif asset.tag == "texture" and compiler.get("texturedir"):
                base = Path(compiler.get("texturedir"))
        asset.set("file", str((base / source).resolve()))


def patch_scene(scene: Path, output: Path, patches: dict[str, dict[str, Any]],
                attach_piper: str = "auto") -> dict[str, Any]:
    if scene.expanduser().resolve() == output.expanduser().resolve():
        raise ValueError("output must differ from the source scene")
    if not patches:
        raise ValueError("patch must contain at least one body")
    handle = load_scene(scene, attach_piper=attach_piper)
    try:
        root = ET.parse(scene).getroot()
        changed = []
        initial_key = root.find("./keyframe/key[@name='stage2_initial']")
        key_id = (_mujoco().mj_name2id(handle.model, _mujoco().mjtObj.mjOBJ_KEY, "stage2_initial")
                  if initial_key is not None else -1)
        initial_qpos = handle.model.key_qpos[key_id].copy() if key_id >= 0 else None
        for body_name, values in patches.items():
            if not isinstance(values, dict) or not values:
                raise ValueError(f"patch for {body_name} must contain parameters")
            unknown = set(values) - {"mass_kg", "mass", "friction", "damping", "position", "quaternion",
                                     "joint_axis", "center_of_mass", "collision"}
            if unknown:
                raise ValueError(f"unknown parameters for {body_name}: {', '.join(sorted(unknown))}")
            body = _find_body_element(root, body_name)
            body_id = _body_id(handle.model, body_name)
            if _robot_body(handle.model, body_id):
                raise ValueError(f"cannot patch robot body: {body_name}")
            free_joints = [joint_id for joint_id in range(handle.model.body_jntadr[body_id],
                           handle.model.body_jntadr[body_id] + handle.model.body_jntnum[body_id])
                           if handle.model.jnt_type[joint_id] == _mujoco().mjtJoint.mjJNT_FREE]
            free_address = int(handle.model.jnt_qposadr[free_joints[0]]) if free_joints else None
            if "joint_axis" in values:
                axis = _finite_vector(values["joint_axis"], 3, "joint_axis")
                length = float(np.linalg.norm(axis))
                joint = body.find("joint")
                if length < 1e-9 or joint is None or joint.get("type") not in {"slide", "hinge"}:
                    raise ValueError(f"joint_axis requires one hinge/slide joint: {body_name}")
                joint.set("axis", _numbers(axis / length))
                changed.append({"body": body_name, "parameter": "joint_axis",
                                "new": (axis / length).tolist()})
            if "position" in values:
                position = _finite_vector(values["position"], 3, "position")
                body.set("pos", _numbers(position))
                if initial_qpos is not None and free_address is not None:
                    initial_qpos[free_address:free_address + 3] = position
                changed.append({"body": body_name, "parameter": "position",
                                "new": position.tolist()})
            if "quaternion" in values:
                quaternion = _normalize_quaternion(values["quaternion"])
                body.set("quat", _numbers(quaternion))
                if initial_qpos is not None and free_address is not None:
                    initial_qpos[free_address + 3:free_address + 7] = quaternion
                changed.append({"body": body_name, "parameter": "quaternion",
                                "new": quaternion.tolist()})
            if "mass_kg" in values or "mass" in values or "center_of_mass" in values:
                mass_value = values.get("mass_kg", values.get("mass", handle.model.body_mass[body_id]))
                mass = float(mass_value) if "mass_kg" in values or "mass" in values else float(handle.model.body_mass[body_id])
                if not np.isfinite(mass) or mass <= 0:
                    raise ValueError(f"mass for {body_name} must be positive and finite")
                old_mass = float(handle.model.body_mass[body_id])
                if old_mass <= 0:
                    raise ValueError(f"body {body_name} has no positive compiled mass")
                inertial = body.find("inertial")
                if inertial is None:
                    inertial = ET.Element("inertial")
                    body.insert(0, inertial)
                ratio = mass / old_mass
                inertial.attrib.clear()
                center_of_mass = (_finite_vector(values["center_of_mass"], 3, "center_of_mass")
                                  if "center_of_mass" in values else handle.model.body_ipos[body_id])
                inertial.attrib.update({
                    "pos": _numbers(center_of_mass),
                    "quat": _numbers(handle.model.body_iquat[body_id]),
                    "mass": f"{mass:.9g}",
                    "diaginertia": _numbers(handle.model.body_inertia[body_id] * ratio),
                })
                for geom in body.findall("geom"):
                    geom.set("mass", "0")
                if "mass_kg" in values or "mass" in values:
                    changed.append({"body": body_name, "parameter": "mass_kg",
                                    "old": old_mass, "new": mass})
                if "center_of_mass" in values:
                    changed.append({"body": body_name, "parameter": "center_of_mass",
                                    "old": handle.model.body_ipos[body_id].tolist(), "new": center_of_mass.tolist()})
            if "collision" in values:
                patch = values["collision"]
                if not isinstance(patch, dict) or set(patch) not in ({"geom", "mode"},
                                                                     {"geom", "mode", "pos", "size"}):
                    raise ValueError("collision requires geom, mode and (for box) pos and size")
                name = patch["geom"]
                geoms = [geom for geom in body.findall("geom") if geom.get("name") == name]
                if (len(geoms) != 1 or not name.endswith("_collision") or
                        name not in {handle.model.geom(geom_id).name for geom_id in range(handle.model.ngeom)
                                     if handle.model.geom_bodyid[geom_id] == body_id}):
                    raise ValueError(f"collision must name a direct task collider: {name}")
                geom = geoms[0]
                if patch["mode"] == "none" and set(patch) == {"geom", "mode"}:
                    if (int(handle.model.body_jntnum[body_id]) != 0 or
                            not re.search(r"backdrop|cyclorama|background|curtain", body_name, re.I)):
                        raise ValueError("collision removal is restricted to static backdrops")
                    body.remove(geom)
                elif patch["mode"] == "box" and set(patch) == {"geom", "mode", "pos", "size"}:
                    size = _finite_vector(patch["size"], 3, "collision.size")
                    if np.any(size <= 0):
                        raise ValueError("collision box size must be positive")
                    geom.attrib.pop("mesh", None)
                    geom.set("type", "box")
                    geom.set("pos", _numbers(_finite_vector(patch["pos"], 3, "collision.pos")))
                    geom.set("size", _numbers(size))
                else:
                    raise ValueError("collision mode must be none or box")
                changed.append({"body": body_name, "parameter": "collision", "new": patch})
            if "friction" in values:
                geoms = body.findall("geom")
                if not geoms:
                    raise ValueError(f"body {body_name} has no direct geoms to patch")
                geom_ids = [geom_id for geom_id in range(handle.model.ngeom)
                            if int(handle.model.geom_bodyid[geom_id]) == body_id]
                if len(geoms) != len(geom_ids):
                    raise ValueError(f"body {body_name} geometry differs from compiled scene")
                frictions = []
                for geom, geom_id in zip(geoms, geom_ids):
                    friction = _friction_value(values["friction"], handle.model.geom_friction[geom_id])
                    geom.set("friction", _numbers(friction))
                    frictions.append(list(friction))
                changed.append({"body": body_name, "parameter": "friction",
                                "new": frictions, "geom_count": len(geoms)})
            if "damping" in values:
                damping = float(values["damping"])
                if not np.isfinite(damping) or damping < 0:
                    raise ValueError(f"damping for {body_name} must be finite and nonnegative")
                joints = body.findall("joint")
                if not joints:
                    raise ValueError(f"body {body_name} has no direct joints to patch")
                for joint in joints:
                    joint.set("damping", f"{damping:.9g}")
                changed.append({"body": body_name, "parameter": "damping",
                                "new": damping, "joint_count": len(joints)})
        if initial_qpos is not None:
            initial_key.set("qpos", _numbers(initial_qpos))
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        _rebase_scene_assets(root, scene, output)
        ET.indent(root, space="  ")
        with tempfile.NamedTemporaryFile(prefix=".stage3_patch_", suffix=".xml",
                                         dir=output.parent, delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            ET.ElementTree(root).write(staged, encoding="utf-8", xml_declaration=True)
            validation = load_scene(staged, attach_piper=attach_piper)
            validation.close()
            os.replace(staged, output)
        finally:
            staged.unlink(missing_ok=True)
        return {"source_scene": str(scene.expanduser().resolve()), "output_scene": str(output),
                "changed": changed, "passed": True}
    finally:
        handle.close()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_report(report: dict[str, Any], output: Path | None,
                  print_stdout: bool = True) -> None:
    serialized = json.dumps({key: value for key, value in report.items() if key != "samples"},
                            ensure_ascii=False, indent=2, allow_nan=False)
    if print_stdout:
        print(serialized)
    if output is not None:
        output = output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(serialized + "\n", encoding="utf-8")
        if "samples" in report:
            (output / "trajectory.json").write_text(
                json.dumps(report["samples"], ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe and patch Stage 3 MuJoCo scenes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("objects", "list only task objects and physical parameters"),
                            ("inspect", "list all bodies, joints, masses and friction"),
                            ("static", "check spontaneous drift over several seconds"),
                            ("grasp", "pinch and lift with two actuated frictional box jaws"),
                            ("run", "run a time-scheduled force/torque probe"),
                            ("auto", "probe movable task bodies independently"),
                            ("try", "compare one candidate mass/friction against the original"),
                            ("patch", "write a patched copy of a scene")):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--scene", required=True, type=Path)
        subparser.add_argument("--attach-piper", choices=("auto", "never", "always"), default="auto")
    subparsers.choices["run"].add_argument("--script", type=Path)
    subparsers.choices["run"].add_argument("--action", action="append",
                                             help="inline action JSON; may be repeated")
    subparsers.choices["run"].add_argument("--duration", type=float, default=1.0)
    subparsers.choices["run"].add_argument("--settle", type=float, default=0.2)
    subparsers.choices["run"].add_argument("--output", type=Path)
    subparsers.choices["objects"].add_argument("--output", type=Path,
                                                help="also render a third-person overview PNG here")
    subparsers.choices["objects"].add_argument("--settle", type=float, default=0.1)
    static = subparsers.choices["static"]
    static.add_argument("--seconds", type=float, default=3.0)
    static.add_argument("--settle", type=float, default=0.1)
    static.add_argument("--max-drift-m", type=float, default=0.01)
    static.add_argument("--max-rotation-rad", type=float, default=0.1)
    static.add_argument("--output", type=Path)
    grasp = subparsers.choices["grasp"]
    grasp.add_argument("--body", required=True)
    grasp.add_argument("--normal-forces", nargs="+", required=True, type=float)
    grasp.add_argument("--seconds", type=float, default=0.4)
    grasp.add_argument("--settle", type=float, default=0.1)
    grasp.add_argument("--jaw-friction", type=float, default=1.0)
    grasp.add_argument("--jaw-size", type=float, default=0.006)
    grasp.add_argument("--grasp-axis", choices=("x", "y"), default="x")
    grasp.add_argument("--jaw-damping", type=float, default=2.0)
    grasp.add_argument("--min-lift-m", type=float, default=0.01)
    grasp.add_argument("--output", type=Path)
    subparsers.choices["auto"].add_argument("--duration", type=float, default=0.2)
    subparsers.choices["auto"].add_argument("--settle", type=float, default=0.1)
    subparsers.choices["auto"].add_argument("--force-n", type=float, default=0.02)
    subparsers.choices["auto"].add_argument("--torque-nm", type=float, default=0.001)
    subparsers.choices["auto"].add_argument("--min-displacement-m", type=float, default=1e-6)
    subparsers.choices["auto"].add_argument("--min-rotation-rad", type=float, default=1e-4)
    subparsers.choices["auto"].add_argument("--output", type=Path)
    subparsers.choices["auto"].add_argument("--body", help="probe only this movable task body")
    trial = subparsers.choices["try"]
    trial.add_argument("--body", required=True)
    trial.add_argument("--mass-kg", type=float)
    trial.add_argument("--friction", type=float, help="sliding friction; preserves torsional/rolling friction")
    trial.add_argument("--damping", type=float)
    trial.add_argument("--duration", type=float, default=0.2)
    trial.add_argument("--settle", type=float, default=0.1)
    trial.add_argument("--force-n", type=float, default=0.02)
    trial.add_argument("--torque-nm", type=float, default=0.001)
    trial.add_argument("--output", type=Path, required=True)
    subparsers.choices["patch"].add_argument("--patch-file", type=Path, required=True)
    subparsers.choices["patch"].add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "objects":
        report = (observe_scene(args.scene, args.output, args.settle, args.attach_piper) if args.output else
                  task_objects(args.scene, args.attach_piper, args.settle))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "static":
        report = check_static(args.scene, args.output, args.seconds, args.settle,
                              args.max_drift_m, args.max_rotation_rad)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 2
    if args.command == "grasp":
        report = check_grasp(args.scene, args.body, args.normal_forces, args.output,
                             args.seconds, args.settle, args.min_lift_m,
                             args.jaw_friction, args.jaw_size, args.attach_piper,
                             args.grasp_axis, args.jaw_damping)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 2
    if args.command == "inspect":
        handle = load_scene(args.scene, args.attach_piper)
        try:
            print(json.dumps(inspect_scene(handle), ensure_ascii=False, indent=2))
        finally:
            handle.close()
        return 0
    if args.command == "patch":
        report = patch_scene(args.scene, args.output, _load_json(args.patch_file), args.attach_piper)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "auto":
        report = auto_probe(args.scene, args.duration, args.settle, args.force_n,
                            args.torque_nm, args.min_displacement_m,
                            args.min_rotation_rad, args.attach_piper, args.body)
        _write_report(report, args.output)
        return 0 if report["passed"] else 2
    if args.command == "try":
        report = try_parameters(args.scene, args.body, args.output,
                                args.mass_kg, args.friction, args.damping,
                                args.duration, args.settle, args.force_n,
                                args.torque_nm, args.attach_piper)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if report["passed"] else 2
    if args.script is not None:
        script = _load_json(args.script)
    else:
        script = {"duration": args.duration, "settle": args.settle,
                  "actions": [json.loads(item) for item in (args.action or [])]}
    report = run_probe(args.scene, script, args.attach_piper)
    _write_report(report, args.output)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
