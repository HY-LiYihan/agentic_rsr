import json

import numpy as np

from real2sim.stage1.sim3_registration import apply_pose, load_observations, search_rotations, similarity_fit, refine, score_views, TRAIN_VIEWS


def test_similarity_recovers_translation_orientation_and_scale():
    rng = np.random.default_rng(25)
    model = rng.uniform(-0.2, 0.2, size=(180, 3))
    rotation = search_rotations(4)[2]
    expected = (rotation, np.array([0.5, -0.2, 1.3]), 1.45)
    observed = apply_pose(model, *expected)
    fitted = similarity_fit(model, observed, (0.4, 2.0))
    np.testing.assert_allclose(fitted[0], expected[0], atol=1e-10)
    np.testing.assert_allclose(fitted[1], expected[1], atol=1e-10)
    np.testing.assert_allclose(fitted[2], expected[2], atol=1e-10)


def test_multiscale_icp_refines_center_seed():
    rng = np.random.default_rng(26)
    model = rng.uniform(-0.2, 0.2, size=(500, 3))
    observed = apply_pose(model, np.eye(3), np.array([0.5, -0.2, 1.2]), 1.3)
    seed = (np.eye(3), np.array([0.51, -0.19, 1.21]), 1.25)
    fitted = refine(model, observed, seed, (0.5, 2.0), 10)
    np.testing.assert_allclose(apply_pose(model, *fitted), observed, atol=1e-5)


def test_training_views_exclude_held_out_depth_and_masks():
    points = np.array([[0., 0., 2.], [.02, 0., 2.], [0., .02, 2.]])
    camera = {"intrinsics": np.tile([[100., 0., 40.], [0., 100., 40.], [0., 0., 1.]], (32, 1, 1)),
              "extrinsics": np.tile(np.column_stack((np.eye(3), [0, 0, 0])), (32, 1, 1))}
    depths = {index: np.full((80, 80), 2., np.float32) for index in range(32)}
    masks = {index: np.zeros((80, 80), bool) for index in range(32)}
    for index in TRAIN_VIEWS:
        masks[index][38:44, 38:44] = True
    original = score_views(points, (np.eye(3), np.zeros(3), 1.), camera,
                           depths, masks, TRAIN_VIEWS)[0]
    for index in (0, 10, 21, 31):
        depths[index][:] = 8.
        masks[index][:] = True
    changed = score_views(points, (np.eye(3), np.zeros(3), 1.), camera,
                          depths, masks, TRAIN_VIEWS)[0]
    assert original == changed


def test_initialization_observations_include_all_32_frames(tmp_path):
    directory = tmp_path / "scene"
    (directory / "depth").mkdir(parents=True)
    (directory / "sam3_masks").mkdir()
    (directory / "manifest.json").write_text(json.dumps({"status": "inference_complete"}))
    camera = {"extrinsics_direction": "world_to_camera",
              "intrinsics": np.tile(np.eye(3), (32, 1, 1)).tolist(),
              "extrinsics": np.tile(np.eye(4)[:3], (32, 1, 1)).tolist()}
    (directory / "camera.json").write_text(json.dumps(camera))
    for index in range(32):
        np.save(directory / "depth" / f"depth_{index:02}.npy",
                np.full((8, 8), index + 1, dtype=np.float32))
        np.save(directory / "sam3_masks" / f"mask_{index:02}.npy",
                np.ones((8, 8), dtype=bool))
    _, _, _, _, observed, counts = load_observations(directory, held_out=())
    assert [entry["frame"] for entry in counts] == list(range(32))
    assert all(entry["points"] == 4 for entry in counts)
    assert set(observed[:, 2]) == set(range(1, 33))
