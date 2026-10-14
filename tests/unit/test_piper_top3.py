import json
from pathlib import Path

import pytest

from real2sim.stage1.metric_calibration import PIPER_TOP3_METHOD, valid_metric_method
from real2sim.stage1.piper_top3 import cached_branch, publish_winner, ranked_seeds


def test_ranked_seeds_uses_loss_not_fixed_identifier():
    seeds = [{"index": index, "loss": float(index)} for index in range(128)]
    for index, loss in ((111, -3.), (60, -2.), (42, -1.)):
        seeds[index]["loss"] = loss
    init = {"robot": "piper", "status": "initialization_only", "rotation_count": 128,
            "candidates": seeds, "pointcloud_frames": list(range(32)),
            "scoring_frames": list(range(32))}
    assert [seed["index"] for seed in ranked_seeds(init)] == [111, 60, 42]
    init["rotation_count"] = 48
    with pytest.raises(ValueError, match="128-seed"):
        ranked_seeds(init)


def test_publish_selects_best_refined_loss_not_initial_rank(tmp_path, monkeypatch):
    init = tmp_path / "initialization.json"
    init.write_bytes(b"initialization")
    from real2sim.stage1.pipeline import sha256

    branches = []
    for rank, index, loss in ((1, 111, .30), (2, 60, .23), (3, 42, .10)):
        folder = tmp_path / f"rank{rank:02d}"
        (folder / "depth_metric").mkdir(parents=True)
        for name in ("camera_metric.json", "before_after_32.png", "overlay_32.png"):
            (folder / name).write_bytes(f"{index}".encode())
        for frame in range(32):
            (folder / "depth_metric" / f"depth_{frame:02d}.npy").write_bytes(f"{index}".encode())
        branches.append(({"index": index, "selected_seed_rank": rank,
                          "initial_loss_saved": .4 + .01 * rank, "loss": loss,
                          "initialization_sha256": sha256(init), "sam3_masks_sha256": "masks",
                          "metrics": {"scored_views": {"mean_iou": .9}}}, folder))
    result = publish_winner("scene", branches, tmp_path / "output", init,
                            {"sam3_masks_sha256": "masks"})
    assert result["index"] == 42 and result["selected_seed_rank"] == 3
    assert result["method"] == PIPER_TOP3_METHOD
    assert len(result["branch_results"]) == 3
    assert (tmp_path / "output/depth_metric/depth_31.npy").read_bytes() == b"42"
    assert json.loads((tmp_path / "output/result.json").read_text()) == result
    assert valid_metric_method({"robot": "piper", "method": result["method"]})


def test_cached_branch_accepts_actual_processed_source(tmp_path):
    from real2sim.stage1.metric_calibration import metric_method
    from real2sim.stage1.pipeline import sha256

    source = tmp_path / "fresh-processed/piper/stage1/task"
    init = tmp_path / "initialization.json"
    init.write_bytes(b"fresh initialization")
    folder = tmp_path / "branch"
    (folder / "depth_metric").mkdir(parents=True)
    for name in ("camera_metric.json", "before_after_32.png", "overlay_32.png"):
        (folder / name).touch()
    for index in range(32):
        (folder / "depth_metric" / f"depth_{index:02d}.npy").touch()
    result = {"robot": "piper", "scene": "task", "source": str(source),
              "method": metric_method("piper"), "initialization_sha256": sha256(init),
              "sam3_masks_sha256": "masks", "selected_seed_rank": 1, "index": 42,
              "pointcloud_frames": list(range(32)), "scoring_frames": list(range(32)),
              "status": "candidate_unverified", "accepted": False}
    (folder / "result.json").write_text(json.dumps(result))
    arguments = (folder, "task", 1, {"index": 42}, init, {"sam3_masks_sha256": "masks"})
    assert cached_branch(*arguments, source=source) == result
    assert cached_branch(*arguments, source=tmp_path / "other") is None
    assert cached_branch(*arguments) is None
