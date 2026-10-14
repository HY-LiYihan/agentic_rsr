from pathlib import Path

from sim.episode import SimEpisode
from sim.randomization import Pose3D, RandomizationConfig


class FakeScene:
    def __init__(self):
        self.calls = []

    def reset_nominal(self):
        self.calls.append(("reset_nominal",))

    def set_object_pose(self, name, pose):
        self.calls.append(("object", name, pose))

    def set_camera_pose(self, name, pose):
        self.calls.append(("camera", name, pose))

    def move_pose(self, pose):
        self.calls.append(("move", pose))


def pose(x, y, z=0.75):
    return Pose3D((x, y, z), (1.0, 0.0, 0.0, 0.0))


def test_episode_resets_nominal_scene_then_applies_runtime_randomization():
    scene = FakeScene()
    episode = SimEpisode(
        scene,
        Path("runs/stage3/mujoco/scene.xml"),
        seed=42,
        config=RandomizationConfig(
            object_position_jitter_m=(0.01, 0.01, 0.0),
            workspace_min_xy_m=(-0.2, -0.2),
            workspace_max_xy_m=(0.1, 0.2),
            min_object_separation_m=0.1,
        ),
    )
    context = episode.reset(
        {"red block": pose(-0.15, -0.1), "blue block": pose(-0.02, 0.12)},
        {"wrist": pose(0.0, 0.0, 1.0)},
    )
    assert context.stage3_scene.endswith("scene.xml")
    assert scene.calls[0] == ("reset_nominal",)
    assert [call[0] for call in scene.calls[1:]] == ["object", "object", "camera",]


def test_episode_applies_control_error_only_at_execution_boundary():
    scene = FakeScene()
    episode = SimEpisode(
        scene,
        Path("scene.xml"),
        seed=3,
        config=RandomizationConfig(control_position_jitter_m=(0.01, 0.01, 0.01)),
    )
    episode.reset({"red block": pose(0.0, 0.0)}, {})
    commanded = pose(-0.05, -0.05, 0.9)
    execution = episode.move_pose(commanded, "pick")
    assert execution.commanded_pose == commanded
    assert execution.arrived_pose != commanded
    assert scene.calls[-1][0] == "move"
    assert episode.to_manifest()["control_executions"][0]["label"] == "pick"


def test_estimated_camera_pose_is_available_to_perception_but_physical_pose_is_applied_to_scene():
    scene = FakeScene()
    episode = SimEpisode(
        scene,
        Path("scene.xml"),
        seed=3,
        config=RandomizationConfig(camera_calibration_position_error_m=(0.01, 0.01, 0.01)),
    )
    context = episode.reset({}, {"wrist": pose(0.0, 0.0, 1.0)})
    assert context.estimated_camera_pose("wrist") != context.cameras["wrist"].physical_pose
    assert scene.calls[-1][0] == "camera"
    assert scene.calls[-1][2] == context.cameras["wrist"].physical_pose
