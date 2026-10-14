import math

import numpy as np

from sim.randomization import EpisodeRandomizer, Pose3D, RandomizationConfig


def pose(x: float, y: float, z: float = 0.75) -> Pose3D:
    return Pose3D((x, y, z), (1.0, 0.0, 0.0, 0.0))


def test_object_randomization_is_reproducible_and_bounded():
    config = RandomizationConfig(
        object_position_jitter_m=(0.01, 0.01, 0.0),
        object_rotation_jitter_rad=(0.0, 0.0, math.radians(5)),
        workspace_min_xy_m=(-0.2, -0.2),
        workspace_max_xy_m=(0.1, 0.2),
        min_object_separation_m=0.1,
    )
    nominal = {"red block": pose(-0.15, -0.1), "blue block": pose(-0.02, 0.12)}
    first = EpisodeRandomizer(17, config).sample_object_poses(nominal)
    second = EpisodeRandomizer(17, config).sample_object_poses(nominal)
    assert first == second
    for item in first.values():
        assert -0.2 <= item.position[0] <= 0.1
        assert -0.2 <= item.position[1] <= 0.2
        assert item.position[2] == 0.75


def test_camera_keeps_physical_and_estimated_poses_separate():
    config = RandomizationConfig(
        camera_mount_position_jitter_m=(0.01, 0.01, 0.01),
        camera_mount_rotation_jitter_rad=(0.02, 0.02, 0.02),
        camera_calibration_position_error_m=(0.005, 0.005, 0.005),
        camera_calibration_rotation_error_rad=(0.01, 0.01, 0.01),
    )
    sample = EpisodeRandomizer(3, config).sample_camera("wrist", pose(0.0, 0.0, 1.0))
    assert sample.physical_pose != sample.nominal_pose
    assert sample.estimated_pose != sample.physical_pose
    assert np.all(np.abs(sample.mount_position_offset_m) <= 0.01)
    assert np.all(np.abs(sample.calibration_position_error_m) <= 0.005)


def test_control_error_applies_to_commanded_pose_and_is_recorded():
    config = RandomizationConfig(
        control_position_jitter_m=(0.01, 0.02, 0.03),
        control_rotation_jitter_rad=(0.1, 0.2, 0.3),
    )
    randomizer = EpisodeRandomizer(9, config)
    error = randomizer.sample_control_error("pick")
    target = pose(0.0, 0.0, 0.9)
    arrived = randomizer.apply_control_error(target, error)
    assert arrived.position != target.position
    assert np.all(np.abs(error.position_offset_m) <= (0.01, 0.02, 0.03))
    assert randomizer.to_manifest()["control_errors"][0]["label"] == "pick"


def test_object_sampling_rejects_impossible_constraints():
    config = RandomizationConfig(
        object_position_jitter_m=(0.0, 0.0, 0.0),
        workspace_min_xy_m=(-0.1, -0.1),
        workspace_max_xy_m=(0.1, 0.1),
        min_object_separation_m=0.2,
        max_object_sampling_attempts=2,
    )
    nominal = {"a": pose(0.0, 0.0), "b": pose(0.05, 0.0)}
    randomizer = EpisodeRandomizer(1, config)
    try:
        randomizer.sample_object_poses(nominal)
    except RuntimeError as error:
        assert "unable to sample" in str(error)
    else:
        raise AssertionError("impossible object constraints must fail closed")
