import hashlib
import json
import sys
from types import ModuleType

import cv2
import numpy as np
import pytest

from real2sim.stage2 import phase_runner
from real2sim.stage2.evaluator import evaluate_bundle
from real2sim.stage3.mujoco_export import _stage2_delivery


def _baseline(tmp_path, monkeypatch, depth_errors=(0.0, 0.0, 0.0, 0.0)):
    source = tmp_path / "processed/piper/stage1/example"
    output = tmp_path / "processed/piper/stage2/example"
    (source / "reference").mkdir(parents=True)
    (source / "depth_metric").mkdir()
    (source / "initial.blend").write_bytes(b"original frozen scene")
    (source / "stage1_delivery.json").write_text(json.dumps({
        "accepted": False, "reference_indices": [0, 10, 21, 31],
    }))
    (output / "plan").mkdir(parents=True)
    (output / "plan/plan.md").write_text("Reviewed plan")
    rgb = np.full((4, 6, 3), 128, dtype=np.uint8)
    for ordinal, frame in enumerate((0, 10, 21, 31)):
        cv2.imwrite(str(source / "reference" / f"frame_{ordinal:02d}.jpg"), rgb)
        np.save(source / "depth_metric" / f"depth_{frame:02d}.npy",
                np.ones((4, 6), dtype=np.float32))

    def prepare(_blender, original, target):
        assert original == source / "initial.blend"
        target.write_bytes(b"prepared frozen scene with lighting")
        return {"source": str(original), "target": str(target)}

    def blender(_blender, scene, command, target):
        assert scene.read_bytes() == b"prepared frozen scene with lighting"
        if command == "inspect":
            target.write_text(json.dumps({"frozen_signature": {"robot": "unchanged"}}))
        elif command == "render":
            target.mkdir()
            for ordinal, error in enumerate(depth_errors):
                cv2.imwrite(str(target / f"camera_{ordinal}_rgb.png"), rgb)
                np.save(target / f"camera_{ordinal}_depth.npy",
                        np.full((4, 6), 1.0 + error, dtype=np.float32))
            (target / "render_manifest.json").write_text("{}")
        else:
            raise AssertionError(f"Unexpected Blender command: {command}")

    monkeypatch.setattr(phase_runner, "_blender_path", lambda _: "test-blender")
    monkeypatch.setattr(phase_runner, "_prepare_initial_scene", prepare)
    monkeypatch.setattr(phase_runner, "_run_blender", blender)
    # Baseline-only completion must not create a Codex session.
    monkeypatch.setitem(sys.modules, "openai_codex", ModuleType("openai_codex"))
    return source, output


@pytest.mark.parametrize("max_rounds", [0, 10])
def test_passing_baseline_produces_stage3_readable_delivery(tmp_path, monkeypatch, max_rounds):
    source, output = _baseline(tmp_path, monkeypatch)
    result = phase_runner.run_geometry(source, output, max_rounds=max_rounds)

    assert result["status"] == "passed"
    final = output / "geometry/final.blend"
    assert result["final_scene"] == str(final)
    assert final.read_bytes() == b"prepared frozen scene with lighting"
    assert (source / "initial.blend").read_bytes() == b"original frozen scene"
    delivery = json.loads((output / "geometry/stage2_manifest.json").read_text())
    assert delivery["scene_sha256"] == hashlib.sha256(final.read_bytes()).hexdigest()
    assert delivery["thread_id"] is None
    assert delivery["evaluation"] == result["final_evaluation"]
    assert delivery["evaluation"]["passed"]
    assert len(delivery["evaluation"]["views"]) == 4
    assert json.loads((output / "geometry/frozen_signature.json").read_text()) == {
        "robot": "unchanged",
    }
    accepted = _stage2_delivery(output)
    assert accepted["robot"] == "piper"
    assert accepted["scene"] == final
    assert accepted["plan"] == output / "plan/plan.md"
    assert len(accepted["rgb_paths"]) == 4


@pytest.mark.parametrize("depth_errors", [(1.0, 1.0, 1.0, 1.0), (0.3, 0.0, 0.0, 0.0)])
def test_rejected_zero_round_baseline_has_no_delivery(tmp_path, monkeypatch, depth_errors):
    source, output = _baseline(tmp_path, monkeypatch, depth_errors)
    result = phase_runner.run_geometry(source, output, max_rounds=0)

    assert result["status"] == "failed_baseline"
    assert result["final_scene"] is None
    assert not result["baseline"]["passed"]
    assert not (output / "geometry/final.blend").exists()
    assert not (output / "geometry/stage2_manifest.json").exists()


def _completed_round(source, output, depth_error=1.0):
    round_dir = output / "geometry/round_001"
    render = round_dir / "render"
    render.mkdir(parents=True)
    for ordinal in range(4):
        rgb = np.full((4, 6, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(render / f"camera_{ordinal}_rgb.png"), rgb)
        np.save(render / f"camera_{ordinal}_depth.npy",
                np.full((4, 6), 1.0 + depth_error, dtype=np.float32))
    report = evaluate_bundle(source, render, round_dir / "evaluation")
    (round_dir / "candidate.blend").write_bytes(b"cached candidate")
    files = [round_dir / "candidate.blend", render / "render_manifest.json"]
    (render / "render_manifest.json").write_text("{}")
    files += [render / f"camera_{ordinal}_{suffix}" for ordinal in range(4)
              for suffix in ("rgb.png", "depth.npy")]
    (round_dir / "round_report.json").write_text(json.dumps({
        "round": 1, "evaluation": report,
        "artifact_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in files},
    }))
    return round_dir


def test_resumed_passing_baseline_delivers_evaluated_scene_not_failed_candidate(tmp_path, monkeypatch):
    source, output = _baseline(tmp_path, monkeypatch)
    phase_runner.run_geometry(source, output, max_rounds=0)
    _completed_round(source, output)

    result = phase_runner.run_geometry(source, output, max_rounds=1, resume=True)

    assert result["status"] == "passed"
    accepted = _stage2_delivery(output)
    assert accepted["scene"].read_bytes() == b"prepared frozen scene with lighting"
    assert result["final_evaluation"] == result["baseline"]
    assert result["rounds"][0]["evaluation"]["passed"] is False


@pytest.mark.parametrize("changed", ["depth", "rgb", "scene", "plan", "manifest", "threshold"])
def test_resume_rejects_changed_inputs_before_reusing_acceptance(tmp_path, monkeypatch, changed):
    source, output = _baseline(tmp_path, monkeypatch)
    phase_runner.run_geometry(source, output, max_rounds=0)
    original_delivery = (output / "geometry/stage2_manifest.json").read_bytes()
    if changed == "depth":
        np.save(source / "depth_metric/depth_00.npy", np.full((4, 6), 2.0, dtype=np.float32))
    elif changed == "rgb":
        cv2.imwrite(str(source / "reference/frame_00.jpg"), np.zeros((4, 6, 3), dtype=np.uint8))
    elif changed == "scene":
        (source / "initial.blend").write_bytes(b"changed robot or cameras")
    elif changed == "plan":
        (output / "plan/plan.md").write_text("Different reviewed plan")
    elif changed == "manifest":
        (source / "stage1_delivery.json").write_text(json.dumps({
            "reference_indices": [31, 21, 10, 0],
        }))
    else:
        monkeypatch.setattr(phase_runner, "DEPTH_MAE_THRESHOLD_M", 0.05)
    with pytest.raises(ValueError, match="provenance.*new output directory"):
        phase_runner.run_geometry(source, output, max_rounds=0, resume=True)
    assert (output / "geometry/stage2_manifest.json").read_bytes() == original_delivery


@pytest.mark.parametrize("artifact", ["initial_with_default_light.blend", "inspect.json",
                                     "baseline_render/camera_0_rgb.png",
                                     "baseline_render/camera_0_depth.npy"])
def test_resume_rejects_changed_baseline_artifacts(tmp_path, monkeypatch, artifact):
    source, output = _baseline(tmp_path, monkeypatch)
    phase_runner.run_geometry(source, output, max_rounds=0)
    (output / "geometry" / artifact).write_bytes(b"modified cached artifact")
    with pytest.raises(ValueError, match="provenance.*new output directory"):
        phase_runner.run_geometry(source, output, max_rounds=0, resume=True)


def test_resume_recomputes_baseline_instead_of_trusting_cached_report(tmp_path, monkeypatch):
    source, output = _baseline(tmp_path, monkeypatch, depth_errors=(1.0,) * 4)
    assert phase_runner.run_geometry(source, output, max_rounds=0)["status"] == "failed_baseline"
    path = output / "geometry/baseline_evaluation/report.json"
    report = json.loads(path.read_text())
    report["depth_mae_mean_m"] = 0
    report["lab_delta_e76_mean"] = 0
    report["passed"] = True
    for view in report["views"]:
        view.update(depth_passed=True, lab_passed=True)
    path.write_text(json.dumps(report))
    result = phase_runner.run_geometry(source, output, max_rounds=0, resume=True)
    assert result["status"] == "failed_baseline"
    assert result["baseline"]["depth_mae_mean_m"] == pytest.approx(1.0)
    assert not (output / "geometry/stage2_manifest.json").exists()


def test_unchanged_resume_preserves_valid_delivery(tmp_path, monkeypatch):
    source, output = _baseline(tmp_path, monkeypatch)
    first = phase_runner.run_geometry(source, output, max_rounds=0)
    result = phase_runner.run_geometry(source, output, max_rounds=0, resume=True)
    assert result["status"] == "passed"
    assert result["final_evaluation"] == first["final_evaluation"]
    assert _stage2_delivery(output)["scene"].is_file()


def test_existing_geometry_requires_resume_and_legacy_cache_requires_new_directory(tmp_path, monkeypatch):
    source, output = _baseline(tmp_path, monkeypatch)
    phase_runner.run_geometry(source, output, max_rounds=0)
    with pytest.raises(FileExistsError, match="--resume"):
        phase_runner.run_geometry(source, output, max_rounds=0)
    (output / "geometry/input_provenance.json").unlink()
    assert _stage2_delivery(output)["scene"].is_file()
    with pytest.raises(ValueError, match="provenance.*new output directory"):
        phase_runner.run_geometry(source, output, max_rounds=0, resume=True)


@pytest.mark.parametrize("artifact", ["candidate.blend", "render/camera_0_rgb.png", "render/camera_0_depth.npy"])
def test_resume_rejects_changed_candidate_artifacts(tmp_path, monkeypatch, artifact):
    source, output = _baseline(tmp_path, monkeypatch)
    phase_runner.run_geometry(source, output, max_rounds=0)
    round_dir = _completed_round(source, output, depth_error=0)
    (round_dir / artifact).write_bytes(b"changed candidate artifact")
    with pytest.raises(ValueError, match="provenance.*new output directory"):
        phase_runner.run_geometry(source, output, max_rounds=1, resume=True)


def test_resume_rechecks_round_metrics_before_delivery(tmp_path, monkeypatch):
    source, output = _baseline(tmp_path, monkeypatch, depth_errors=(1.0,) * 4)
    phase_runner.run_geometry(source, output, max_rounds=0)
    round_dir = _completed_round(source, output)
    report_path = round_dir / "evaluation/report.json"
    report = json.loads(report_path.read_text())
    report["depth_mae_mean_m"] = 0
    report["lab_delta_e76_mean"] = 0
    report["passed"] = True
    for view in report["views"]:
        view.update(depth_passed=True, lab_passed=True)
    report_path.write_text(json.dumps(report))
    result = phase_runner.run_geometry(source, output, max_rounds=1, resume=True)
    assert result["status"] == "max_rounds_exceeded"
    assert not result["rounds"][0]["evaluation"]["passed"]
    assert not (output / "geometry/stage2_manifest.json").exists()


def test_resume_delivers_verified_passing_candidate(tmp_path, monkeypatch):
    source, output = _baseline(tmp_path, monkeypatch, depth_errors=(1.0,) * 4)
    phase_runner.run_geometry(source, output, max_rounds=0)
    _completed_round(source, output, depth_error=0)
    result = phase_runner.run_geometry(source, output, max_rounds=1, resume=True)
    assert result["status"] == "passed"
    assert _stage2_delivery(output)["scene"].read_bytes() == b"cached candidate"
    assert result["final_evaluation"]["depth_mae_mean_m"] == pytest.approx(0)


def test_all_phases_rejects_existing_geometry_before_calling_plan_agent(tmp_path, monkeypatch):
    source, output = _baseline(tmp_path, monkeypatch)
    phase_runner.run_geometry(source, output, max_rounds=0)

    def forbidden_agent(*args, **kwargs):
        pytest.fail("Existing geometry must be rejected before starting the planning agent")

    monkeypatch.setattr(phase_runner, "run_plan", forbidden_agent)
    with pytest.raises(FileExistsError, match="--phase geometry --resume"):
        phase_runner.run_phases(source, output)
