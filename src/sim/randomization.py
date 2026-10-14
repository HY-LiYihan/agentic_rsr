"""Reproducible, bounded randomization for simulation episodes.

The sampler keeps physical scene variation, camera calibration error, and
control execution error separate. This lets a future harness render with the
physical camera pose while giving the policy only the perturbed calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


def _finite_vector(value, size: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return tuple(float(item) for item in array)


def _nonnegative_vector(value, size: int, name: str) -> tuple[float, ...]:
    result = _finite_vector(value, size, name)
    if any(item < 0 for item in result):
        raise ValueError(f"{name} must be non-negative")
    return result


def _uniform_offset(rng: np.random.Generator, bounds: tuple[float, ...]) -> np.ndarray:
    return rng.uniform(-np.asarray(bounds), np.asarray(bounds))


def _normalize_quaternion(value) -> tuple[float, float, float, float]:
    quaternion = np.asarray(_finite_vector(value, 4, "quaternion"), dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("quaternion must be non-zero")
    return tuple(float(item) for item in quaternion / norm)


def _quaternion_multiply(left, right) -> tuple[float, float, float, float]:
    left_w, left_x, left_y, left_z = left
    right_w, right_x, right_y, right_z = right
    return _normalize_quaternion((
        left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
        left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
        left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
        left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
    ))


def _quaternion_from_rotation_vector(rotation_vector) -> tuple[float, float, float, float]:
    vector = np.asarray(_finite_vector(rotation_vector, 3, "rotation vector"), dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    axis = vector / angle
    half_angle = angle / 2.0
    sine = math.sin(half_angle)
    return _normalize_quaternion((math.cos(half_angle), *(axis * sine)))


def _apply_delta(pose: "Pose3D", position_offset, rotation_offset) -> "Pose3D":
    position = np.asarray(pose.position, dtype=float) + np.asarray(position_offset, dtype=float)
    rotation = _quaternion_multiply(
        _quaternion_from_rotation_vector(rotation_offset), pose.quaternion_wxyz,
    )
    return Pose3D(tuple(float(item) for item in position), rotation)


@dataclass(frozen=True)
class Pose3D:
    """A metric pose using metres and a `(w, x, y, z)` quaternion."""

    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _finite_vector(self.position, 3, "position"))
        object.__setattr__(self, "quaternion_wxyz", _normalize_quaternion(self.quaternion_wxyz))

    def to_manifest(self) -> dict[str, list[float]]:
        return {
            "position_m": list(self.position),
            "quaternion_wxyz": list(self.quaternion_wxyz),
        }


@dataclass(frozen=True)
class RandomizationConfig:
    """Upper bounds for one simulation randomization policy.

    Bounds are symmetric and sampled independently from uniform distributions.
    Object and camera mount perturbations describe physical scene variation;
    camera calibration perturbations describe the pose supplied to perception.
    """

    object_position_jitter_m: tuple[float, float, float] = (0.01, 0.01, 0.0)
    object_rotation_jitter_rad: tuple[float, float, float] = (0.0, 0.0, math.radians(5.0))
    camera_mount_position_jitter_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_mount_rotation_jitter_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_calibration_position_error_m: tuple[float, float, float] = (0.005, 0.005, 0.005)
    camera_calibration_rotation_error_rad: tuple[float, float, float] = (
        math.radians(1.0), math.radians(1.0), math.radians(1.0),
    )
    control_position_jitter_m: tuple[float, float, float] = (0.005, 0.005, 0.005)
    control_rotation_jitter_rad: tuple[float, float, float] = (
        math.radians(2.0), math.radians(2.0), math.radians(2.0),
    )
    workspace_min_xy_m: tuple[float, float] | None = None
    workspace_max_xy_m: tuple[float, float] | None = None
    min_object_separation_m: float | None = None
    max_object_sampling_attempts: int = 100

    def __post_init__(self) -> None:
        vector_fields = (
            ("object_position_jitter_m", 3),
            ("object_rotation_jitter_rad", 3),
            ("camera_mount_position_jitter_m", 3),
            ("camera_mount_rotation_jitter_rad", 3),
            ("camera_calibration_position_error_m", 3),
            ("camera_calibration_rotation_error_rad", 3),
            ("control_position_jitter_m", 3),
            ("control_rotation_jitter_rad", 3),
        )
        for field_name, size in vector_fields:
            object.__setattr__(self, field_name, _nonnegative_vector(getattr(self, field_name), size, field_name))
        if self.workspace_min_xy_m is not None:
            object.__setattr__(self, "workspace_min_xy_m",
                               _finite_vector(self.workspace_min_xy_m, 2, "workspace_min_xy_m"))
        if self.workspace_max_xy_m is not None:
            object.__setattr__(self, "workspace_max_xy_m",
                               _finite_vector(self.workspace_max_xy_m, 2, "workspace_max_xy_m"))
        if (self.workspace_min_xy_m is None) != (self.workspace_max_xy_m is None):
            raise ValueError("workspace_min_xy_m and workspace_max_xy_m must be provided together")
        if self.workspace_min_xy_m is not None and np.any(
                np.asarray(self.workspace_min_xy_m) > np.asarray(self.workspace_max_xy_m)):
            raise ValueError("workspace minimum must not exceed workspace maximum")
        if self.min_object_separation_m is not None and (
                not math.isfinite(self.min_object_separation_m) or self.min_object_separation_m < 0):
            raise ValueError("min_object_separation_m must be non-negative")
        if self.max_object_sampling_attempts < 1:
            raise ValueError("max_object_sampling_attempts must be positive")

    def to_manifest(self) -> dict[str, object]:
        return {
            "object_position_jitter_m": list(self.object_position_jitter_m),
            "object_rotation_jitter_rad": list(self.object_rotation_jitter_rad),
            "camera_mount_position_jitter_m": list(self.camera_mount_position_jitter_m),
            "camera_mount_rotation_jitter_rad": list(self.camera_mount_rotation_jitter_rad),
            "camera_calibration_position_error_m": list(self.camera_calibration_position_error_m),
            "camera_calibration_rotation_error_rad": list(self.camera_calibration_rotation_error_rad),
            "control_position_jitter_m": list(self.control_position_jitter_m),
            "control_rotation_jitter_rad": list(self.control_rotation_jitter_rad),
            "workspace_min_xy_m": None if self.workspace_min_xy_m is None else list(self.workspace_min_xy_m),
            "workspace_max_xy_m": None if self.workspace_max_xy_m is None else list(self.workspace_max_xy_m),
            "min_object_separation_m": self.min_object_separation_m,
            "max_object_sampling_attempts": self.max_object_sampling_attempts,
        }


@dataclass(frozen=True)
class CameraRandomization:
    name: str
    nominal_pose: Pose3D
    physical_pose: Pose3D
    estimated_pose: Pose3D
    mount_position_offset_m: tuple[float, float, float]
    mount_rotation_offset_rad: tuple[float, float, float]
    calibration_position_error_m: tuple[float, float, float]
    calibration_rotation_error_rad: tuple[float, float, float]

    def to_manifest(self) -> dict[str, object]:
        return {
            "nominal_pose": self.nominal_pose.to_manifest(),
            "physical_pose": self.physical_pose.to_manifest(),
            "estimated_pose": self.estimated_pose.to_manifest(),
            "mount_position_offset_m": list(self.mount_position_offset_m),
            "mount_rotation_offset_rad": list(self.mount_rotation_offset_rad),
            "calibration_position_error_m": list(self.calibration_position_error_m),
            "calibration_rotation_error_rad": list(self.calibration_rotation_error_rad),
        }


@dataclass(frozen=True)
class ControlRandomization:
    label: str
    position_offset_m: tuple[float, float, float]
    rotation_offset_rad: tuple[float, float, float]

    def to_manifest(self) -> dict[str, object]:
        return {
            "label": self.label,
            "position_offset_m": list(self.position_offset_m),
            "rotation_offset_rad": list(self.rotation_offset_rad),
        }


class EpisodeRandomizer:
    """Sample and record all stochastic variables for one episode."""

    def __init__(self, seed: int, config: RandomizationConfig | None = None):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.seed = seed
        self.config = config or RandomizationConfig()
        self._rng = np.random.default_rng(seed)
        self._objects: dict[str, Pose3D] = {}
        self._cameras: dict[str, CameraRandomization] = {}
        self._controls: list[ControlRandomization] = []

    def sample_object_poses(self, nominal_poses: Mapping[str, Pose3D]) -> dict[str, Pose3D]:
        if not nominal_poses:
            self._objects = {}
            return {}
        items = tuple(nominal_poses.items())
        for _ in range(self.config.max_object_sampling_attempts):
            candidate = {
                name: _apply_delta(
                    pose,
                    _uniform_offset(self._rng, self.config.object_position_jitter_m),
                    _uniform_offset(self._rng, self.config.object_rotation_jitter_rad),
                )
                for name, pose in items
            }
            if self._objects_are_valid(candidate):
                self._objects = candidate
                return dict(candidate)
        raise RuntimeError("unable to sample object poses within workspace and separation constraints")

    def _objects_are_valid(self, poses: Mapping[str, Pose3D]) -> bool:
        positions = np.asarray([pose.position for pose in poses.values()], dtype=float)
        if self.config.workspace_min_xy_m is not None:
            if np.any(positions[:, :2] < self.config.workspace_min_xy_m):
                return False
            if np.any(positions[:, :2] > self.config.workspace_max_xy_m):
                return False
        if self.config.min_object_separation_m is not None and len(positions) > 1:
            distances = np.linalg.norm(positions[:, None, :2] - positions[None, :, :2], axis=-1)
            distances[np.diag_indices_from(distances)] = np.inf
            if float(np.min(distances)) < self.config.min_object_separation_m:
                return False
        return True

    def sample_camera(self, name: str, nominal_pose: Pose3D) -> CameraRandomization:
        mount_position = _uniform_offset(self._rng, self.config.camera_mount_position_jitter_m)
        mount_rotation = _uniform_offset(self._rng, self.config.camera_mount_rotation_jitter_rad)
        calibration_position = _uniform_offset(self._rng, self.config.camera_calibration_position_error_m)
        calibration_rotation = _uniform_offset(self._rng, self.config.camera_calibration_rotation_error_rad)
        physical_pose = _apply_delta(nominal_pose, mount_position, mount_rotation)
        estimated_pose = _apply_delta(physical_pose, calibration_position, calibration_rotation)
        result = CameraRandomization(
            name=name,
            nominal_pose=nominal_pose,
            physical_pose=physical_pose,
            estimated_pose=estimated_pose,
            mount_position_offset_m=tuple(float(item) for item in mount_position),
            mount_rotation_offset_rad=tuple(float(item) for item in mount_rotation),
            calibration_position_error_m=tuple(float(item) for item in calibration_position),
            calibration_rotation_error_rad=tuple(float(item) for item in calibration_rotation),
        )
        self._cameras[name] = result
        return result

    def sample_control_error(self, label: str = "") -> ControlRandomization:
        position = _uniform_offset(self._rng, self.config.control_position_jitter_m)
        rotation = _uniform_offset(self._rng, self.config.control_rotation_jitter_rad)
        result = ControlRandomization(
            label=label,
            position_offset_m=tuple(float(item) for item in position),
            rotation_offset_rad=tuple(float(item) for item in rotation),
        )
        self._controls.append(result)
        return result

    @staticmethod
    def apply_control_error(target_pose: Pose3D, error: ControlRandomization) -> Pose3D:
        return _apply_delta(target_pose, error.position_offset_m, error.rotation_offset_rad)

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema": "sim-randomization-v1",
            "seed": self.seed,
            "config": self.config.to_manifest(),
            "objects": {name: pose.to_manifest() for name, pose in self._objects.items()},
            "cameras": {name: sample.to_manifest() for name, sample in self._cameras.items()},
            "control_errors": [sample.to_manifest() for sample in self._controls],
        }
