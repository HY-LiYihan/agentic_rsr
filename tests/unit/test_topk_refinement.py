import json

import numpy as np
import pytest

from real2sim.stage1 import topk_refinement


def test_minimum_k_covers_both_rotations(tmp_path):
    for name, ranking in (("one", [42, 221, 6]), ("two", [6, 221, 42])):
        directory = tmp_path / name
        directory.mkdir()
        report = {"robot": "piper", "status": "initialization_only",
                  "candidates": [{"index": index, "loss": position} for position, index in enumerate(ranking)]}
        (directory / "initialization.json").write_text(json.dumps(report))
    k, ranks = topk_refinement.minimum_k(tmp_path)
    assert k == 3
    assert ranks == {"one": {"221": 2, "42": 1}, "two": {"221": 2, "42": 3}}


def test_minimum_k_rejects_missing_rotation(tmp_path):
    directory = tmp_path / "scene"
    directory.mkdir()
    (directory / "initialization.json").write_text(json.dumps({
        "robot": "piper", "status": "initialization_only",
        "candidates": [{"index": 221, "loss": 0.1}]}))
    with pytest.raises(ValueError, match="missing required rotation"):
        topk_refinement.minimum_k(tmp_path)


def test_optimizing_each_branch_keeps_best_all_frame_loss(monkeypatch):
    model = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    models = {size: model for size, _ in topk_refinement.MODEL_STAGES}
    seed = {"index": 221, "rotation": np.eye(3).tolist(),
            "translation_vggt_world": [0., 0., 2.],
            "scale_vggt_units_per_meter": 1., "loss": 0.5}
    calls = []

    def score(_, __, pose, ___, ____, _____, indices):
        calls.append(indices)
        return float(np.linalg.norm(pose[1] - [0., 0., 2.])), {}

    def worsen(_, __, pose, ___, ____):
        return pose[0], pose[1] + [1., 0., 0.], pose[2]

    monkeypatch.setattr(topk_refinement, "ranking", score)
    monkeypatch.setattr(topk_refinement, "refine", worsen)
    output = topk_refinement.optimize_seed(seed, models, model, {}, {}, {},
                                            np.zeros(3), (.5, 1.6), 1., passes=0)
    assert output["index"] == 221
    assert output["loss"] == 0.
    assert output["scale_vggt_units_per_meter"] == 1.
    assert all(not stage["accepted"] for stage in output["stages"])
    assert all(indices == tuple(range(32)) for indices in calls)


def test_local_search_can_optimize_translation_rotation_and_scale():
    initial = np.eye(3), np.array([0., 0., 0.]), 1.
    target_rotation = np.array([[np.cos(.1), -np.sin(.1), 0.],
                                [np.sin(.1), np.cos(.1), 0.], [0., 0., 1.]])

    def score(pose):
        rotation, translation, scale = pose
        return (float(np.linalg.norm(rotation - target_rotation) ** 2 +
                      np.linalg.norm(translation - [.025, 0., 0.]) ** 2 +
                      (scale - np.exp(.05)) ** 2), {})

    result, loss, _, changes = topk_refinement.local_search(
        initial, np.zeros(3), (.5, 1.6), 1., score, passes=1)
    np.testing.assert_allclose(result[0], target_rotation, atol=1e-12)
    np.testing.assert_allclose(result[1], [.025, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(result[2], np.exp(.05), atol=1e-12)
    assert loss < 1e-10 and changes >= 3
