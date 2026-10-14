import json

import numpy as np
import pytest

from real2sim.stage1 import franka_final


def initialization():
    return {"robot": "franka", "status": "initialization_only",
            "rotation_count": 48, "pointcloud_frames": list(range(32)),
            "scoring_frames": list(range(32)),
            "candidates": [{"index": index, "loss": 1 - index / 100,
                            "scale_vggt_units_per_meter": 1.}
                           for index in range(48)]}


def test_select_seed_uses_only_best_of_48():
    seed = franka_final.select_seed(initialization())
    assert seed["index"] == 47


def test_piper_selects_best_of_128_rotations_without_forcing_r42():
    report = initialization()
    report["robot"] = "piper"
    report["rotation_count"] = 128
    report["candidates"] = [{"index": index, "loss": abs(index - 60) / 128}
                            for index in range(128)]
    assert franka_final.select_seed(report, robot="piper")["index"] == 60
    assert [franka_final.select_seed(report, robot="piper", seed_rank=rank)["index"]
            for rank in (1, 2, 3)] == [60, 59, 61]


@pytest.mark.parametrize("field,value", [
    ("robot", "piper"), ("rotation_count", 256),
    ("scoring_frames", list(range(31))),
    ("pointcloud_frames", list(range(1, 32))),
])
def test_select_seed_rejects_mismatched_inputs(field, value):
    report = initialization()
    report[field] = value
    with pytest.raises(ValueError):
        franka_final.select_seed(report)


def test_refine_scene_sends_only_top_seed_and_all_frames(tmp_path, monkeypatch):
    initializations = tmp_path / "initializations"
    scene = initializations / "task"
    scene.mkdir(parents=True)
    (scene / "initialization.json").write_text(json.dumps(initialization()))
    processed = tmp_path / "processed"
    models = {20000: np.zeros((3, 3))}
    masks = {index: np.ones((8, 8), dtype=bool) for index in range(32)}
    cameras = {"intrinsics": np.tile(np.eye(3), (32, 1, 1)),
               "extrinsics": np.tile(np.eye(4)[:3], (32, 1, 1))}
    observation = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])

    def load(source, *, held_out):
        assert source == processed / "franka" / "stage1" / "task"
        assert held_out == ()
        return ({"robot": "franka"}, cameras, {}, masks, observation,
                [{"frame": index, "points": 1} for index in range(32)])

    calls = []

    def optimize(seed, *_args):
        calls.append(seed["index"])
        return {"index": seed["index"], "initial_loss_recomputed": 0.53,
                "loss": 0.3, "rotation": np.eye(3).tolist(),
                "translation_vggt_world": [0., 0., 1.],
                "scale_vggt_units_per_meter": 1., "metrics": {}}

    scoring_frames = []

    def rank(_, __, ___, ____, _____, ______, indices):
        scoring_frames.append(indices)
        return 0.2, {"per_frame": [{"frame": index} for index in indices]}

    def rotate(pose, _, __, ___, score, *, initial_score):
        assert initial_score == (0.3, {})
        loss, metrics = score(pose)
        return pose, loss, metrics, [{"rotation_step_degrees": 15., "accepted_moves": 0}]

    monkeypatch.setattr(franka_final, "load_observations", load)
    monkeypatch.setattr(franka_final, "optimize_seed", optimize)
    monkeypatch.setattr(franka_final, "ranking", rank)
    monkeypatch.setattr(franka_final, "refine_rotation", rotate)
    monkeypatch.setattr(franka_final, "ranked_sheet", lambda *args, **kwargs: None)
    monkeypatch.setattr(franka_final, "export_metric_scene", lambda *args: {
        "status": "urdf_anchored_metric_scale", "meters_per_vggt_unit": 1.})
    output = tmp_path / "output"
    result = franka_final.refine_scene("task", processed, initializations, output,
                                       models, observation)
    assert calls == [47]
    assert scoring_frames == [tuple(range(32))]
    assert result["selected_seed_rank"] == 1
    assert result["baseline_loss"] == 0.3 and result["loss"] == 0.2
    assert result["method"] == franka_final.METHOD
    assert result["metric_calibration"]["meters_per_vggt_unit"] == 1.
    assert result["scoring_frames"] == result["pointcloud_frames"] == list(range(32))
    assert result["status"] == "candidate_unverified" and not result["accepted"]
    assert json.loads((output / "result.json").read_text()) == result
