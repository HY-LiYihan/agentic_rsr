"""Check metric backprojection, frame-zero viewpoint and robot-base Z direction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/visualize_stage1_arm_cloud.py"
SPEC = importlib.util.spec_from_file_location("visualize_stage1_arm_cloud", SCRIPT)
visualization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(visualization)


def test_metric_cloud_uses_world_to_camera_and_positive_camera_depth(tmp_path):
    (tmp_path / "depth_metric").mkdir()
    (tmp_path / "sam3_masks").mkdir()
    (tmp_path / "frames").mkdir()
    intrinsic = [[100, 0, 2], [0, 100, 1], [0, 0, 1]]
    rotation = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    extrinsic = [row + [value] for row, value in zip(rotation, [0, 0, 1])]
    (tmp_path / "camera_metric.json").write_text(json.dumps({
        "intrinsics": [intrinsic], "extrinsics": [extrinsic],
        "world_frame": "robot_base", "depth_unit": "m", "extrinsics_direction": "world_to_camera"}))
    depth = np.zeros((3, 5), dtype=np.float32)
    depth[1, 2] = 2
    depth[1, 3] = 2
    depth[2, 1] = -1
    mask = depth != 0
    np.save(tmp_path / "depth_metric/depth_00.npy", depth)
    np.save(tmp_path / "sam3_masks/mask_00.npy", mask)
    cv2.imwrite(str(tmp_path / "frames/00.jpg"), np.zeros((3, 5, 3), dtype=np.uint8))

    points, colors, loaded_intrinsic, loaded_extrinsic, image = visualization.load_cloud(tmp_path)
    np.testing.assert_allclose(points, [[0, 1, 0], [.02, 1, 0]], atol=1e-6)
    np.testing.assert_allclose(visualization.project(points, loaded_intrinsic, loaded_extrinsic),
                               [[2, 1], [3, 1]], atol=1e-6)
    assert len(colors) == 2 and image.shape == (3, 5, 3)


def test_camera_view_aligns_with_first_camera_up_and_optical_axis():
    rotation, _ = cv2.Rodrigues(np.array([1.2, -.8, .3]))
    extrinsic = np.column_stack((rotation, [0., 0., 1.]))
    elevation, azimuth, roll = visualization.camera_view(extrinsic)
    elev, azim, bank = np.deg2rad([elevation, azimuth, roll])
    eye_direction = np.array([np.cos(elev) * np.cos(azim),
                              np.cos(elev) * np.sin(azim), np.sin(elev)])
    screen_right = np.cross([0., 0., 1.], eye_direction)
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(eye_direction, screen_right)
    adjusted_up = screen_up * np.cos(bank) + screen_right * np.sin(bank)
    np.testing.assert_allclose(eye_direction, -rotation.T[:, 2], atol=1e-8)
    np.testing.assert_allclose(adjusted_up, -rotation.T[:, 1], atol=1e-8)
