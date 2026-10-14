import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "organize_stage2_results.py"
SPEC = spec_from_file_location("stage2_organizer", SCRIPT)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
organize = MODULE.organize


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_organizer_only_prunes_finished_round_previews(tmp_path):
    done = tmp_path / "franka" / "stage2" / "done" / "geometry"
    active = tmp_path / "piper" / "stage2" / "active" / "geometry"
    for geometry in (done, active):
        round_dir = geometry / "round_001"
        round_dir.mkdir(parents=True)
        for filename in ("candidate.blend", "candidate.blend1", "agent_preview.blend"):
            (round_dir / filename).write_bytes(b"scene")
        _write_json(round_dir / "evaluation" / "report.json", {
            "depth_mae_mean_m": 0.1, "lab_delta_e76_mean": 12.0,
        })
    (done / "final.blend").write_bytes(b"scene")
    _write_json(done / "stage2_manifest.json", {"accepted": True, "evaluation": {
        "depth_mae_mean_m": 0.1, "lab_delta_e76_mean": 12.0,
    }})
    _write_json(done / "session.json", {"status": "passed"})
    _write_json(active / "session.json", {"status": "running"})
    unfinished = done / "round_002"
    unfinished.mkdir()
    (unfinished / "agent_preview.blend").write_bytes(b"unfinished")

    output = active.parent.resolve()
    metadata_root = tmp_path / "runs" / "stage2" / "metadata"
    result = organize(tmp_path, prune=True,
                      commands=f"python -m real2sim.stage2.agent_runner --output {output}\n",
                      metadata_root=metadata_root)
    assert result["counts"] == {"passed": 1, "running": 1}
    assert result["pruned_files"] == 2
    assert (done / "round_001" / "candidate.blend").exists()
    assert not (done / "round_001" / "candidate.blend1").exists()
    assert not (done / "round_001" / "agent_preview.blend").exists()
    assert (unfinished / "agent_preview.blend").exists()
    assert (active / "round_001" / "candidate.blend1").exists()
    assert (metadata_root / "results_index.json").exists()
    repeated = organize(tmp_path, prune=True, commands="", metadata_root=metadata_root)
    assert repeated["pruned_files"] == 0
    assert repeated["total_pruned_files"] == 2
