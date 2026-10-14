import numpy as np
import pytest
import trimesh

from real2sim.stage1.robot_surface import BASE_LINKS, FRANKA_URDF, POSES, colored_visual_meshes, generate, generate_pyramid, link_poses, sample_surface, visual_meshes
from real2sim.stage1.piper_wrist import assembly_sha256, piper_description, stand_mesh_parts, stand_mesh_correction, STAND_MESH
from real2sim.stage1.geometry import MODEL, description
import xml.etree.ElementTree as ET


@pytest.mark.parametrize("robot", ("piper", "franka"))
def test_fixed_joint_pose_and_open_gripper(robot):
    root = description() if robot == "piper" else ET.parse(FRANKA_URDF).getroot()
    frames, values = link_poses(root, BASE_LINKS[robot], POSES[robot])
    np.testing.assert_allclose(frames[BASE_LINKS[robot]], np.eye(4), atol=1e-8)
    if robot == "piper":
        np.testing.assert_allclose([values[f"joint{index}"] for index in range(1, 7)],
                                   np.deg2rad([0, 30, -45, 0, 60, 0]))
        assert values["gripper_joint1"] == pytest.approx(0.035)
        assert values["gripper_joint2"] == pytest.approx(-0.035)
    else:
        assert values["fr3_finger_joint1"] == pytest.approx(0.04)
        assert values["fr3_finger_joint2"] == pytest.approx(0.04)
        np.testing.assert_allclose(
            [values[f"fr3_joint{index}"] for index in range(1, 8)],
            np.deg2rad([0, -45, 0, -135, 0, 90, 45]))
        alternate = {**POSES[robot], "fr3_joint4": -1.0}
        alternate_frames, _ = link_poses(root, BASE_LINKS[robot], alternate)
        assert not np.allclose(frames["fr3_hand"], alternate_frames["fr3_hand"])


def test_area_weighted_surface_and_voxel_deduplication():
    small = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    large = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    large.apply_translation((0.2, 0, 0))
    points = sample_surface([small, large], count=1500, voxel=0.002)
    assert points.shape == (1500, 3)
    assert (points[:, 0] > 0.1).mean() > 0.95
    assert len(np.unique(np.floor(points / 0.002).astype(int), axis=0)) == len(points)
    np.testing.assert_array_equal(points, sample_surface([small, large], 1500, 0.002))


def test_piper_camera_and_stand_are_physical_visuals_in_link6_frame():
    root = piper_description()
    frames, _ = link_poses(root, BASE_LINKS["piper"], POSES["piper"])
    assert "d435i_camera_link" in frames and "camera_stand_link" in frames
    stand = frames["camera_stand_link"]
    camera = frames["d435i_camera_link"]
    np.testing.assert_allclose(np.linalg.inv(frames["link6"]) @ stand,
                               np.array([[-0.000796, -1., 0., -0.032],
                                         [-1., 0.000796, 0.001593, -0.002],
                                         [-0.001593, 0., -1., 0.025],
                                         [0., 0., 0., 1.]]), atol=2e-3)
    assert np.linalg.norm(camera[:3, 3] - stand[:3, 3]) < 0.2
    meshes = visual_meshes(root, MODEL, "piper", frames)
    assert len(meshes) >= 13
    assert max(mesh.area for mesh in meshes) > 0.01


def test_piper_stand_visual_matches_robot_control_raw_dae_mesh():
    scene = trimesh.load(STAND_MESH, force="scene")
    original = scene.dump(concatenate=False)[0]
    corrected = original.copy()
    corrected.apply_transform(stand_mesh_correction())
    np.testing.assert_allclose(corrected.vertices, stand_mesh_parts()[0].vertices, atol=1e-7)
    np.testing.assert_allclose(stand_mesh_parts()[0].extents, [0.0969936, 0.0741501, 0.0484897], atol=1e-5)


@pytest.mark.parametrize("robot", ("piper", "franka"))
def test_colored_visuals_preserve_original_urdf_geometry(robot):
    root = piper_description() if robot == "piper" else ET.parse(FRANKA_URDF).getroot()
    urdf = MODEL if robot == "piper" else FRANKA_URDF
    frames, _ = link_poses(root, BASE_LINKS[robot], POSES[robot])
    original = visual_meshes(root, urdf, robot, frames)
    colored = colored_visual_meshes(root, urdf, robot, frames)
    assert sum(len(mesh.faces) for mesh in original) == sum(len(mesh.faces) for mesh, _ in colored)
    np.testing.assert_allclose(trimesh.util.concatenate(original).bounds,
                               trimesh.util.concatenate([mesh for mesh, _ in colored]).bounds)
    assert len({rgba for _, rgba in colored}) >= 3
    assert all(len(rgba) == 4 and all(0 <= value <= 1 for value in rgba)
               for _, rgba in colored)
    if robot == "piper":
        assert any(max(rgba[:3]) < 0.15 for _, rgba in colored)
    else:
        assert any(rgba[:3] == (1., .072272, .039546) for _, rgba in colored)


def test_reject_joint_outside_urdf_limit():
    root = ET.parse(FRANKA_URDF).getroot()
    with pytest.raises(ValueError, match="fr3_finger_joint1"):
        link_poses(root, "fr3_link0", {**POSES["franka"], "fr3_finger_joint1": 0.05})


@pytest.mark.parametrize("robot", ("piper", "franka"))
def test_generate_visual_surface(robot, tmp_path):
    report = generate(robot, tmp_path, count=1000)
    with np.load(tmp_path / "surface.npz") as data:
        points = data["points"]
    saved = trimesh.load(tmp_path / "surface.ply")
    assert len(points) == len(saved.vertices) == report["sampled_points"] == 1000
    assert report["base_frame"] == BASE_LINKS[robot]
    assert report["visual_mesh_count"] > 10
    assert np.isfinite(points).all()


@pytest.mark.parametrize("robot", ("piper", "franka"))
def test_generate_nested_asset_resolutions(robot, tmp_path):
    report = generate_pyramid(robot, tmp_path, sizes=(1000, 500, 250))
    clouds = {}
    for size in (1000, 500, 250):
        with np.load(tmp_path / f"surface_{size}.npz") as data:
            clouds[size] = data["points"]
        assert len(trimesh.load(tmp_path / f"surface_{size}.ply").vertices) == size
        assert clouds[size].shape == (size, 3)
    assert set(map(tuple, clouds[250])) <= set(map(tuple, clouds[500])) <= set(map(tuple, clouds[1000]))
    assert report["resolutions"] == [1000, 500, 250]
    assert not (tmp_path / "surface.npz").exists()
