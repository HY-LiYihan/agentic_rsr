import json

import numpy as np
import pytest

from real2sim.stage1 import franka_rotation_refine as rotation_refine


def test_rotation_levels_repeat_until_no_improvement_and_recenter():
    target = np.array([[np.cos(np.deg2rad(30)), -np.sin(np.deg2rad(30)), 0.],
                       [np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30)), 0.],
                       [0., 0., 1.]])
    center = np.array([1., 0., 0.])
    def score(pose):
        rot, translation, scale = pose
        loss = float(np.linalg.norm(rot - target) ** 2 +
                     np.linalg.norm(translation + (rot - np.eye(3)) @ center) ** 2 +
                     (scale - 1.) ** 2)
        return loss, {"loss": loss}

    pose, loss, metrics, history = rotation_refine.refine_rotation(
        (np.eye(3), np.zeros(3), 1.), center, 1., (.5, 1.6), score,
        steps=(15., 8., 4., 2.))
    np.testing.assert_allclose(pose[0], target, atol=1e-5)
    np.testing.assert_allclose(pose[1], (np.eye(3) - target) @ center, atol=1e-5)
    assert loss < 1e-8 and metrics["loss"] == loss
    assert history[0]["accepted_moves"] == 2
    assert [entry["rotation_step_degrees"] for entry in history] == [15., 8., 4., 2.]


def test_rotation_move_can_improve_after_translation_scale_readjustment():
    target = np.array([[np.cos(np.deg2rad(15)), -np.sin(np.deg2rad(15)), 0.],
                       [np.sin(np.deg2rad(15)), np.cos(np.deg2rad(15)), 0.],
                       [0., 0., 1.]])

    def score(pose):
        rot, translation, scale = pose
        angle = np.arctan2(rot[1, 0], rot[0, 0])
        loss = float((angle - np.deg2rad(15)) ** 2 +
                     500 * (translation[0] - .025) ** 2 +
                     500 * (scale - np.exp(.025)) ** 2)
        return loss, {}

    initial = (np.eye(3), np.zeros(3), 1.)
    adjusted, loss, _ = rotation_refine.adjust_translation_scale(
        (target, initial[1], initial[2]), np.zeros(3), 1., (.5, 1.6), score)
    assert loss < score((target, initial[1], initial[2]))[0]
    np.testing.assert_allclose(adjusted[1], [.025, 0., 0.], atol=1e-9)
    np.testing.assert_allclose(adjusted[2], np.exp(.025), atol=1e-9)


def test_run_rejects_mismatched_baseline_before_loading_data(tmp_path):
    baseline = tmp_path / "baseline.json"
    initial = tmp_path / "initial.json"
    initial.write_text(json.dumps({"robot": "franka", "status": "initialization_only",
                                   "rotation_count": 48, "pointcloud_frames": list(range(32)),
                                   "scoring_frames": list(range(32)),
                                   "candidates": [{"index": index, "loss": index / 100}
                                                  for index in range(48)]}))
    baseline.write_text(json.dumps({"robot": "franka", "scene": "other_scene", "index": 0,
                                    "selected_seed_rank": 1, "status": "candidate_unverified",
                                    "scoring_frames": list(range(32))}))
    with pytest.raises(ValueError, match="same scene"):
        rotation_refine.run("close_drawer", baseline, initial, tmp_path, tmp_path / "output")


def test_no_accepted_move_keeps_initial_score_even_with_scoring_jitter():
    calls = 0

    def noisy_score(_pose):
        nonlocal calls
        calls += 1
        return 1. - calls * 1e-6, {"call": calls}

    pose, loss, metrics, history = rotation_refine.refine_rotation(
        (np.eye(3), np.zeros(3), 1.), np.zeros(3), 1., (.5, 1.6), noisy_score,
        steps=(15.,), initial_score=(1., {"call": 0}))
    assert loss == 1. and metrics == {"call": 0}
    assert history[0]["accepted_moves"] == 0
    np.testing.assert_array_equal(pose[0], np.eye(3))
