import json
from pathlib import Path

import pytest

from sim.policy_loop_controller import CodexPolicyLoopController
from sim.policy_bundle import PolicyBundle
from sim.policy_refinement import run_policy_refinement


def make_bundle(root, marker):
    root.mkdir(parents=True)
    (root / "policy.py").write_text(
        f'def run(api):\n    return {{"status": "{marker}"}}\n')
    return PolicyBundle(root)


class FakeRefinementController:
    def __init__(self, refined_root):
        self.refined_root = refined_root
        self.calls = []

    def refine_bundle(self, failure_report, previous_bundle, output_dir):
        self.calls.append((failure_report, previous_bundle.hash, output_dir))
        bundle = make_bundle(output_dir / "bundle", "refined")
        return bundle.root, bundle


def test_refinement_resets_between_candidates_and_keeps_previous_bundle(tmp_path):
    initial = make_bundle(tmp_path / "initial", "initial")
    controller = FakeRefinementController(tmp_path / "unused")
    attempts = []

    def run_episode(*args, **kwargs):
        bundle = kwargs["bundle"]
        profile = args[3]
        attempts.append((bundle.hash, profile, args[5]))
        passed = bundle.hash != initial.hash
        return {
            "profile": profile,
            "seed": args[4],
            "success": passed,
            "policy_bundle": bundle.manifest(),
            "program": {"status": "finished", "history": []},
            "calls": [],
            "before_xyz": {"hidden": [0, 0, 0]},
            "after_xyz": {"hidden": [1, 1, 1]},
            "distance_xy_m": 99,
        }

    def run_both(*args, **kwargs):
        assert kwargs["bundle"].hash != initial.hash
        return {"accepted": True, "results": []}

    report = run_policy_refinement(
        tmp_path / "scene.xml", object(), initial, tmp_path / "run", "host", 123,
        controller, max_iterations=2, dev_seed=17, heldout_seeds=(29,),
        run_episode_fn=run_episode, run_both_fn=run_both)

    assert report["accepted"] is True
    assert report["locked"] is True
    assert report["reset_count"] == 5
    assert len(controller.calls) == 1
    assert len(attempts) == 3
    assert attempts[0][1:] == ("exact", tmp_path / "run" / "episodes" / "iteration_000" / "exact")
    assert attempts[1][1:] == ("exact", tmp_path / "run" / "episodes" / "iteration_001" / "exact")
    assert attempts[2][1:] == (
        "randomized", tmp_path / "run" / "episodes" / "iteration_001" / "randomized" / "17")
    assert (tmp_path / "run" / "policy" / "iteration_000" / "bundle" / "policy.py").is_file()
    assert (tmp_path / "run" / "policy" / "iteration_001" / "bundle" / "policy.py").is_file()
    failure = json.loads(
        (tmp_path / "run" / "refinement" / "iteration_000_failure.json").read_text())
    assert "before_xyz" not in failure
    assert "after_xyz" not in failure
    assert "distance_xy_m" not in failure


def test_refinement_stops_at_max_iterations_without_final_lock(tmp_path):
    initial = make_bundle(tmp_path / "initial", "initial")
    controller = FakeRefinementController(tmp_path / "unused")
    calls = []

    def run_episode(*args, **kwargs):
        calls.append(kwargs["bundle"].hash)
        return {"success": False, "program": {"status": "turn_limit", "history": []},
                "calls": [], "policy_bundle": kwargs["bundle"].manifest()}

    def fail_if_called(*args, **kwargs):
        raise AssertionError("held-out evaluation must not run before development passes")

    report = run_policy_refinement(
        tmp_path / "scene.xml", object(), initial, tmp_path / "run", "host", 123,
        controller, max_iterations=1, run_episode_fn=run_episode,
        run_both_fn=fail_if_called)

    assert report["accepted"] is False
    assert report["locked"] is False
    assert len(calls) == 2
    assert len(controller.calls) == 1
    assert report["refinement_errors"] == []


def test_development_seed_cannot_be_held_out(tmp_path):
    bundle = make_bundle(tmp_path / "initial", "initial")
    with pytest.raises(ValueError, match="Development seed"):
        run_policy_refinement(
            tmp_path / "scene.xml", object(), bundle, tmp_path / "run", "host", 123,
            FakeRefinementController(tmp_path / "unused"),
            dev_seed=101, heldout_seeds=(101,))


def test_refinement_attaches_public_images_to_same_session(tmp_path):
    bundle = make_bundle(tmp_path / "initial", "initial")
    images = []
    for index in range(3):
        image = tmp_path / f"observation_{index}.jpg"
        image.write_bytes(b"public frame")
        images.append(image)
    class FakeSession:
        id = "same"

        def __init__(self):
            self.calls = []

        def run(self, message, schema, output, images=()):
            self.calls.append((message, schema, images))
            return {"files": [{"path": "policy.py", "source":
                               'def run(api):\n    return {"status": "completed"}\n'}]}

    session = FakeSession()
    controller = CodexPolicyLoopController("task", session, tmp_path, generated=True)
    failure = {"calls": [{"tool": "observe", "result": {"image": str(image)}}
                         for image in images]}
    root, refined = controller.refine_bundle(failure, bundle, tmp_path / "iteration_001")
    assert refined.root == root
    assert refined.hash != bundle.hash
    assert controller.session_id == "same"
    assert len(session.calls) == 1
    assert session.calls[0][2] == (images[1], images[2])
