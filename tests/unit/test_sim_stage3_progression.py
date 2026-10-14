"""Stage ordering, privilege isolation and randomized-scene regression tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_stage3_codex_batch as batch
from sim.stage3_harness import Stage3Tools
from sim.stage3_scene import Stage3Scene
from sim.stage3_verification import verify_task


def test_three_stage_order_and_no_privileged_policy_leak(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "SCENE_ROOT", tmp_path)
    calls = []

    def fake_run(robot, task, model, timeout, effort, run_name, max_iterations,
                 total_timeout, *, stage, previous_policy, prior_experience):
        calls.append((stage, previous_policy, prior_experience))
        policy = tmp_path / f"{stage[0]}.py"
        policy.write_text(f"def run(api):\n    return '{stage[0]}'\n")
        stage_output = tmp_path / "piper" / "task" / "new-run" / stage[0]
        stage_output.mkdir(parents=True, exist_ok=True)
        (stage_output / "experience.md").write_text(
            f"experience from {stage[0]}")
        return {"success": True, "verification": "physical_check_passed",
                "policy": str(policy)}

    monkeypatch.setattr(batch, "run_one", fake_run)
    result = batch.run_three_stages("piper", "task", "model", 10, "medium",
                                    "new-run", random_seed=123)
    assert [stage[0] for stage, _, _ in calls] == [
        "privileged_exact", "visual_exact", "visual_randomized"]
    assert [stage[1:] for stage, _, _ in calls] == [(True, None), (False, None),
                                                    (False, 123)]
    assert calls[0][1] is None and calls[0][2] is None
    assert "privileged_exact" in calls[1][1]
    assert "privileged_exact" in calls[1][2]
    assert "visual_exact" in calls[2][1]
    assert "visual_exact" in calls[2][2]
    assert result["success"] is True
    assert len(json.loads((tmp_path / "piper/task/new-run/report.json").read_text())["stages"]) == 3


def test_progression_stops_at_unverified_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "SCENE_ROOT", tmp_path)
    calls = []

    def fake_run(*args, stage, previous_policy, prior_experience, **kwargs):
        calls.append(stage[0])
        return {"success": False, "verification": "pending_video_review"}

    monkeypatch.setattr(batch, "run_one", fake_run)
    report = batch.run_three_stages("piper", "task", "model", 10, "medium")
    assert calls == ["privileged_exact"]
    assert report["success"] is False


def test_exact_stage_blocks_truth_in_policy_and_cli():
    scene = SimpleNamespace(observe=lambda: {"objects": {"Object": {"xyz_m": [1, 2, 3]}}})
    perception = SimpleNamespace(names=("Object",))
    exact = Stage3Tools(scene, "task", "instruction", perception=perception)
    with pytest.raises(ValueError, match="unavailable"):
        exact.dispatch("object_poses")
    privileged = Stage3Tools(scene, "task", "instruction", perception=perception,
                             privileged=True)
    assert privileged.dispatch("object_poses")["Object"]["xyz_m"] == [1, 2, 3]
    prompt = batch.task_prompt("piper", "task", "description", ["Object"],
                               stage="visual_exact", previous_policy="def run(api):\n    pass",
                               prior_experience="previous stage notes")
    assert "Object ground truth is unavailable" in prompt
    assert "PRIOR STAGE POLICY" in prompt
    assert "previous stage notes" in prompt


def test_randomized_stage_resets_and_exact_restores_nominal():
    source = (Path(__file__).resolve().parents[2] / "runs/sim_processed_stage3_all/piper"
              / "put_emergency_stop_button_in_green_storage_box/scene_for_sim.xml")
    if not source.exists():
        pytest.skip("Stage 3 emergency-stop fixture is not installed")
    scene = Stage3Scene(source, "piper")
    try:
        nominal = scene.body_xyz("Stage2_EStopHousing")
        assert verify_task(scene, "put_emergency_stop_button_in_green_storage_box") == "failed"
        scene.set_randomization(123)
        randomized = scene.body_xyz("Stage2_EStopHousing")
        assert randomized != nominal
        assert scene.randomizer.to_manifest()["seed"] == 123
        scene.reset()
        assert scene.randomizer.to_manifest()["seed"] == 124
        scene.set_randomization(None)
        assert scene.body_xyz("Stage2_EStopHousing") == pytest.approx(nominal)
        assert scene.randomizer is None
    finally:
        scene.close()
