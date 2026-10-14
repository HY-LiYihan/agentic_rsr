import hashlib
import json

import pytest

from real2sim.stage1 import piper_final


def test_reuse_requires_matching_video_and_128_seed_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(piper_final, "ROOT", tmp_path)
    raw = tmp_path / "data/raw/piper"
    raw.mkdir(parents=True)
    video = raw / "task.mp4"
    video.write_bytes(b"updated task")
    processed = tmp_path / "data/processed"
    scene = processed / "piper/stage1/task"
    scene.mkdir(parents=True)
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    (scene / "manifest.json").write_text(json.dumps({"video_sha256": digest,
                                                        "sam3_masks_sha256": "new-masks"}))
    initializations = tmp_path / "runs/piper_initial_128_review/piper"
    seed = initializations / "task"
    seed.mkdir(parents=True)
    (seed / "initialization.json").write_bytes(b"128 scored seeds")
    output = tmp_path / "runs/piper_stage1_final_128"
    result = output / "task"
    result.mkdir(parents=True)
    cached = {"scene": "task", "robot": "piper", "method": "piper_128_top1_sim3_rotation_v1",
              "initialization_sha256": hashlib.sha256(b"128 scored seeds").hexdigest(),
              "sam3_masks_sha256": "new-masks",
              "status": "candidate_unverified", "index": 6, "baseline_loss": .2, "loss": .1,
              "scale_vggt_units_per_meter": 1.}
    (result / "result.json").write_text(json.dumps(cached))
    assert piper_final.batch(initializations, processed, output, workers=1,
                             progress=lambda _: None) == [cached]
    video.write_bytes(b"replaced again")
    with pytest.raises(ValueError, match="active Piper scenes"):
        piper_final.batch(initializations, processed, output, workers=1,
                          progress=lambda _: None)
