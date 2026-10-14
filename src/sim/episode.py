"""Episode orchestration between a nominal Stage 3 scene and Sim runtime noise."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from .randomization import (
    CameraRandomization,
    ControlRandomization,
    EpisodeRandomizer,
    Pose3D,
    RandomizationConfig,
)


class EpisodeScene(Protocol):
    """Minimal scene boundary required by the randomization layer."""

    def reset_nominal(self) -> None:
        """Restore the immutable Stage 3 baseline in the runtime instance."""

    def set_object_pose(self, name: str, pose: Pose3D) -> None:
        """Apply one randomized object pose to the runtime scene."""

    def set_camera_pose(self, name: str, pose: Pose3D) -> None:
        """Apply the physical camera pose used for rendering."""

    def move_pose(self, pose: Pose3D) -> None:
        """Execute a pose command in the runtime scene."""


@dataclass(frozen=True)
class ControlExecution:
    label: str
    commanded_pose: Pose3D
    arrived_pose: Pose3D
    error: ControlRandomization

    def to_manifest(self) -> dict[str, object]:
        return {
            "label": self.label,
            "commanded_pose": self.commanded_pose.to_manifest(),
            "arrived_pose": self.arrived_pose.to_manifest(),
            "error": self.error.to_manifest(),
        }


@dataclass(frozen=True)
class EpisodeContext:
    """Internal episode record containing randomized runtime truth."""

    stage3_scene: str
    seed: int
    objects: dict[str, Pose3D]
    cameras: dict[str, CameraRandomization]

    def estimated_camera_pose(self, name: str) -> Pose3D:
        try:
            return self.cameras[name].estimated_pose
        except KeyError as exc:
            raise KeyError(f"unknown randomized camera: {name}") from exc


class SimEpisode:
    """Apply per-episode randomization to a runtime copy of a Stage 3 scene.

    The supplied scene adapter owns the MuJoCo model/data instance. This class
    never edits the Stage 3 source file and never exposes randomized truth to a
    policy-facing observation interface.
    """

    def __init__(self, scene: EpisodeScene, stage3_scene: Path, seed: int,
                 config: RandomizationConfig | None = None):
        self.scene = scene
        self.stage3_scene = Path(stage3_scene)
        self.randomizer = EpisodeRandomizer(seed, config)
        self.context: EpisodeContext | None = None
        self.executions: list[ControlExecution] = []

    def reset(self, nominal_objects: Mapping[str, Pose3D],
              nominal_cameras: Mapping[str, Pose3D]) -> EpisodeContext:
        self.scene.reset_nominal()
        objects = self.randomizer.sample_object_poses(nominal_objects)
        cameras = {
            name: self.randomizer.sample_camera(name, pose)
            for name, pose in nominal_cameras.items()
        }
        for name, pose in objects.items():
            self.scene.set_object_pose(name, pose)
        for name, sample in cameras.items():
            self.scene.set_camera_pose(name, sample.physical_pose)
        self.context = EpisodeContext(
            stage3_scene=str(self.stage3_scene),
            seed=self.randomizer.seed,
            objects=objects,
            cameras=cameras,
        )
        self.executions = []
        return self.context

    def move_pose(self, commanded_pose: Pose3D, label: str = "") -> ControlExecution:
        if self.context is None:
            raise RuntimeError("reset() must be called before move_pose()")
        error = self.randomizer.sample_control_error(label)
        arrived_pose = self.randomizer.apply_control_error(commanded_pose, error)
        self.scene.move_pose(arrived_pose)
        execution = ControlExecution(label, commanded_pose, arrived_pose, error)
        self.executions.append(execution)
        return execution

    def to_manifest(self) -> dict[str, object]:
        if self.context is None:
            raise RuntimeError("reset() must be called before creating an episode manifest")
        manifest = self.randomizer.to_manifest()
        manifest.update({
            "stage3_scene": self.context.stage3_scene,
            "control_executions": [execution.to_manifest() for execution in self.executions],
        })
        return manifest
