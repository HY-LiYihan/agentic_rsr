import json

import numpy as np
import pytest

from real2sim.stage1.metric_calibration import (
    export_metric_scene, metric_transforms, rebase_to_robot_base,
)


def inputs():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    result = {"robot": "franka", "method": "franka_48_top1_sim3_rotation_v1",
              "rotation": rotation.tolist(), "translation_vggt_world": [4., 6., 8.],
              "scale_vggt_units_per_meter": 2.}
    extrinsics = np.tile(np.eye(4)[:3], (32, 1, 1))
    extrinsics[:, :, 3] = [2., -4., 6.]
    camera = {"intrinsics": np.tile([[100., 0., 40.], [0., 100., 40.],
                                     [0., 0., 1.]], (32, 1, 1)).tolist(),
              "extrinsics": extrinsics.tolist(),
              "extrinsics_direction": "world_to_camera"}
    return result, camera


def test_franka_metric_transform_preserves_reprojection_and_robot_size():
    result, camera = inputs()
    calibration, metric_camera = metric_transforms(result, camera)
    assert calibration["status"] == "urdf_anchored_metric_scale"
    assert calibration["scale_source"] == "fitted Franka URDF surface (meters)"
    assert calibration["meters_per_vggt_unit"] == .5
    robot_pose = np.asarray(calibration["robot_pose_metric_world_from_base"])
    np.testing.assert_allclose(robot_pose[:3, :3], result["rotation"])
    np.testing.assert_allclose(robot_pose[:3, 3], [2., 3., 4.])
    np.testing.assert_allclose(np.asarray(metric_camera["extrinsics"])[0, :, 3], [1., -2., 3.])

    point_base_m = np.array([.2, .3, .4])
    world_vggt = (2. * np.asarray(result["rotation"]) @ point_base_m +
                  np.asarray(result["translation_vggt_world"]))
    world_m = robot_pose[:3, :3] @ point_base_m + robot_pose[:3, 3]
    np.testing.assert_allclose(world_m, world_vggt / 2.)
    camera_vggt = np.asarray(camera["extrinsics"])[0, :, :3] @ world_vggt + \
        np.asarray(camera["extrinsics"])[0, :, 3]
    metric_extrinsic = np.asarray(metric_camera["extrinsics"])[0]
    camera_m = metric_extrinsic[:, :3] @ world_m + metric_extrinsic[:, 3]
    np.testing.assert_allclose(camera_m, camera_vggt / 2.)
    np.testing.assert_allclose(camera_m[:2] / camera_m[2], camera_vggt[:2] / camera_vggt[2])


def test_piper_uses_the_same_metric_conversion_and_keeps_urdf_scale():
    result, camera = inputs()
    result.update(robot="piper", method="piper_128_top1_sim3_rotation_v1")
    calibration, metric_camera = metric_transforms(result, camera)
    assert calibration["meters_per_vggt_unit"] == .5
    assert calibration["robot_pose_metric_world_from_base"][0][3] == 2.
    assert metric_camera["scale_source"] == "fitted piper URDF surface (meters)"


@pytest.mark.parametrize("robot,method", [
    ("franka", "franka_48_top1_sim3_rotation_v1"),
    ("piper", "piper_128_top3_sim3_rotation_v1"),
])
def test_robot_base_is_scene_origin_without_changing_camera_projections(robot, method):
    result, camera = inputs()
    result.update(robot=robot, method=method)
    calibration, metric_camera = metric_transforms(result, camera)
    rebased, rebased_camera = rebase_to_robot_base(calibration, metric_camera)
    np.testing.assert_array_equal(rebased["robot_pose_metric_world_from_base"], np.eye(4))
    np.testing.assert_array_equal(rebased["robot_pose_original_metric_world_from_base"],
                                  calibration["robot_pose_metric_world_from_base"])
    assert rebased["world_frame"] == rebased_camera["world_frame"] == "robot_base"
    assert rebased_camera["intrinsics"] == metric_camera["intrinsics"]
    assert metric_camera.get("world_frame") is None
    point_base = np.array([.2, .3, .4])
    original_pose = np.asarray(calibration["robot_pose_metric_world_from_base"])
    world_point = original_pose[:3, :3] @ point_base + original_pose[:3, 3]
    for original_extrinsic, rebased_extrinsic in zip(metric_camera["extrinsics"],
                                                     rebased_camera["extrinsics"]):
        original_extrinsic = np.asarray(original_extrinsic)
        rebased_extrinsic = np.asarray(rebased_extrinsic)
        np.testing.assert_allclose(rebased_extrinsic[:, :3] @ point_base + rebased_extrinsic[:, 3],
                                   original_extrinsic[:, :3] @ world_point + original_extrinsic[:, 3])


@pytest.mark.parametrize("value", [0., -1., float("nan")])
def test_invalid_scale_refused(value):
    result, camera = inputs()
    result["scale_vggt_units_per_meter"] = value
    with pytest.raises(ValueError, match="positive and finite"):
        metric_transforms(result, camera)


def test_export_writes_32_meter_depths_and_camera(tmp_path):
    result, camera = inputs()
    source = tmp_path / "source"
    depth_directory = source / "depth"
    depth_directory.mkdir(parents=True)
    (source / "camera.json").write_text(json.dumps(camera))
    for index in range(32):
        np.save(depth_directory / f"depth_{index:02d}.npy",
                np.full((3, 4), index + 2., dtype=np.float32))
    output = tmp_path / "output"
    calibration = export_metric_scene(source, output, result)
    assert calibration["depth_frames"] == list(range(32))
    assert len(list((output / "depth_metric").glob("*.npy"))) == 32
    np.testing.assert_array_equal(np.load(output / "depth_metric/depth_00.npy"),
                                  np.ones((3, 4), dtype=np.float32))
    metric_camera = json.loads((output / "camera_metric.json").read_text())
    assert metric_camera["depth_unit"] == "m"
    assert len(metric_camera["extrinsics"]) == 32
