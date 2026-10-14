import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/archive_stage3_runs.py"
SPEC = importlib.util.spec_from_file_location("archive_stage3_runs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_archive_moves_runs_without_changing_stage2(tmp_path):
    source = tmp_path / "data/processed/piper/stage3/task"
    source.mkdir(parents=True)
    stage2 = tmp_path / "data/processed/piper/stage2/task/geometry/final.blend"
    stage2.parent.mkdir(parents=True)
    stage2.write_bytes(b"blend")
    (source / "scene.xml").write_text("<mujoco/>")
    (source / "session.json").write_text(json.dumps({"scene": str(source / "scene.xml")}))
    prior = tmp_path / "runs/stage3/batch_old/batch_manifest.json"
    prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps({"scene": str(source / "scene.xml")}))

    preview = MODULE.archive(tmp_path, dry_run=True)
    assert len(preview["moved"]) == 1
    assert source.is_dir()
    MODULE.archive(tmp_path)
    destination = tmp_path / "runs/stage3/workspace/piper/task"
    assert not source.exists()
    assert stage2.read_bytes() == b"blend"
    assert json.loads((destination / "session.json").read_text())["scene"] == str(destination / "scene.xml")
    assert json.loads(prior.read_text())["scene"] == str(destination / "scene.xml")
    assert MODULE.archive(tmp_path)["moved"] == []


def test_archive_refuses_to_overwrite_runs(tmp_path):
    source = tmp_path / "data/processed/franka/stage3/task"
    destination = tmp_path / "runs/stage3/workspace/franka/task"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        MODULE.archive(tmp_path)
    assert source.is_dir()
