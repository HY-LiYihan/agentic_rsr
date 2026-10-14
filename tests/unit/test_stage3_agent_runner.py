import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from real2sim.stage3 import agent_runner
from real2sim.stage3.agent_runner import _command_trace, _validate_visual_prior, verify_run, _sha256
from real2sim.stage3.physics_probe import (_grasp_geometry, check_grasp, check_static,
                                           patch_scene, run_probe, task_objects)
from test_stage3_physics_probe import PAPER_SCENE, write_probe_scene


def test_trace_includes_executed_commands():
    items = [SimpleNamespace(root=SimpleNamespace(
        type="commandExecution", command="python -m real2sim.stage3.physics_probe static",
        exit_code=2, status="completed", aggregated_output="failed"))]
    assert _command_trace(items)[0]["exit_code"] == 2
    wrapped = [SimpleNamespace(root=SimpleNamespace(
        type="commandExecution", command="python -m real2sim.stage3.physics_probe grasp",
        exit_code=1, status="failed", aggregated_output="probe output\n__RC__=0\nzsh: no matches found"))]
    assert _command_trace(wrapped)[0]["probe_exit_code"] == 0


def test_runner_requires_actual_commands_and_reports(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>", encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    (output / "static").mkdir()
    (output / "grasp").mkdir()
    for name in ("static", "grasp"):
        (output / name / "report.json").write_text(json.dumps({
            "scene": str(scene), "passed": True,
            **({"objects": [{"body": "cube", "passed": True}]} if name == "static" else
               {"method": "two_frictional_actuated_box_jaws",
                "trials": [{"target_normal_force_per_jaw_n": 1.0, "lifted": True}]}),
        }), encoding="utf-8")
    result = {"static_report": str(output / "static/report.json"),
              "grasp_report": str(output / "grasp/report.json"),
              "candidate_scene": None, "status": "provisional", "real_world_verified": False}
    trace = [{"command": f"python -m real2sim.stage3.physics_probe {name} --scene {scene}",
              "exit_code": 0} for name in ("objects", "static", "grasp")]
    checked = verify_run(output, _sha256(scene), scene, {"real_world_measurements": []}, trace, result)
    assert checked["passed"] is True
    assert checked["real_world_verified"] is False
    result["grasp_report"] += ": passed=true; force=0.6 N"
    trace[-1]["exit_code"] = 1
    trace[-1]["probe_exit_code"] = 0
    assert verify_run(output, _sha256(scene), scene, {}, trace, result)["passed"] is True
    trace[1]["exit_code"] = 1
    trace[1]["output_tail"] = (output / "static/report.json").read_text(encoding="utf-8")
    checked = verify_run(output, _sha256(scene), scene, {}, trace, result)
    assert checked["passed"] is True
    assert "shell wrapper" in checked["warnings"][0]
    assert any("grasp" in error for error in verify_run(
        output, _sha256(scene), scene, {}, trace[:-1], result)["errors"])
    result["real_world_verified"] = True
    assert verify_run(output, _sha256(scene), scene, {}, trace, result)["passed"] is False


def test_stage3_static_detects_falling_body_and_pose_patch(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene.xml"
    write_probe_scene(scene)
    report = check_static(scene, duration=0.3, settle=0.0, max_drift_m=0.01)
    assert report["passed"] is False
    assert report["objects"][0]["drift_m"] > 0.01
    with pytest.raises(ValueError, match="duration"):
        check_static(scene, duration=float("nan"))
    target = tmp_path / "patched.xml"
    patch_scene(scene, target, {"cube": {"position": [0.1, 0, 0.3],
                                          "quaternion": [2, 0, 0, 0]}}, attach_piper="never")
    assert task_objects(target, attach_piper="never", settle=0)["objects"][0]["position_m"][0] == pytest.approx(0.1)
    assert scene.read_text() != target.read_text()


def test_fixture_two_pad_grasp_depends_on_contact_friction(tmp_path):
    pytest.importorskip("mujoco")
    report = check_grasp(PAPER_SCENE, "blue_block", [0.2, 0.6], output=tmp_path)
    assert report["method"] == "two_frictional_actuated_box_jaws"
    assert report["trials"][1]["lifted"] is True
    assert report["trials"][1]["two_sided_contact_fraction"] > 0.5
    assert all(force > 0 for force in report["trials"][1]["peak_normal_force_per_jaw_n"])
    slippery = check_grasp(PAPER_SCENE, "blue_block", [0.2, 0.6], jaw_friction=0.01)
    assert all(not trial["lifted"] for trial in slippery["trials"])
    with pytest.raises(ValueError, match="force limits"):
        check_grasp(PAPER_SCENE, "blue_block", [-1])
    with pytest.raises(ValueError, match="force limits"):
        check_grasp(PAPER_SCENE, "blue_block", [10.01])
    assert (tmp_path / "report.json").is_file()


def test_grasp_jaws_are_identical_and_symmetric(tmp_path):
    pytest.importorskip("mujoco")
    root = _grasp_geometry(PAPER_SCENE, "blue_block", 0.1, 0.6, 0.006, "auto")
    jaws = [root.find(f"worldbody/body[@name='stage3_probe_jaw_{side}']")
            for side in ("left", "right")]
    centers = [[float(value) for value in jaw.get("pos").split()] for jaw in jaws]
    assert centers[0][0] + centers[1][0] == pytest.approx(-0.1, abs=1e-6)
    assert centers[0][1:] == pytest.approx(centers[1][1:])
    assert [jaw.find("geom").get("size") for jaw in jaws] == [jaws[0].find("geom").get("size")] * 2
    assert [jaw.find("geom").get("friction") for jaw in jaws] == ["0.6 0.005 0.0001"] * 2
    assert [jaw.find("geom").get("mass") for jaw in jaws] == ["0.025"] * 2
    assert [jaw.find("joint").get("range") for jaw in jaws] == ["0 0.05"] * 2
    assert [jaw.find("joint").get("actuatorfrcrange") for jaw in jaws] == ["-10 10"] * 2
    equality = root.find("equality/joint[@name='stage3_probe_mimic']")
    assert equality.get("polycoef") == "0 1 0 0 0"
    assert [act.get("kp") for act in root.findall("actuator/position")
            if act.get("name", "").startswith("stage3_probe_close_motor_")] == ["200"] * 2
    assert [[float(value) for value in jaw.find("joint").get("axis").split()]
            for jaw in jaws] == [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]


def test_grasp_jaws_center_on_offset_collider_not_body_origin(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "offset.xml"
    write_probe_scene(scene)
    root = ET.parse(scene).getroot()
    root.find("worldbody/body[@name='cube']/geom").set("pos", "0.5 0 0")
    ET.ElementTree(root).write(scene, encoding="utf-8")
    augmented = _grasp_geometry(scene, "cube", 0, 1, 0.006, "never")
    jaws = [augmented.find(f"worldbody/body[@name='stage3_probe_jaw_{side}']")
            for side in ("left", "right")]
    x_coordinates = [float(jaw.get("pos").split()[0]) for jaw in jaws]
    assert sum(x_coordinates) / 2 == pytest.approx(0.5)
    assert x_coordinates[1] - x_coordinates[0] < 0.1


def test_grasp_pad_height_does_not_follow_long_object_length(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "elongated.xml"
    write_probe_scene(scene)
    root = ET.parse(scene).getroot()
    root.find("worldbody/body[@name='cube']/geom").set("size", "0.12 0.02 0.02")
    ET.ElementTree(root).write(scene, encoding="utf-8")
    augmented = _grasp_geometry(scene, "cube", 0, 1, 0.006, "never")
    pad = augmented.find("worldbody/body[@name='stage3_probe_jaw_left']/geom")
    dimensions = [float(value) for value in pad.get("size").split()]
    assert dimensions[1:] == pytest.approx([0.018, 0.018])


def test_runner_independently_replays_probe_reports(tmp_path):
    pytest.importorskip("mujoco")
    output = tmp_path / "run"
    output.mkdir()
    static = check_static(PAPER_SCENE, output=output / "static", duration=0.2)
    grasp = check_grasp(PAPER_SCENE, "blue_block", [0.2, 0.6], output=output / "grasp")
    trace = [{"command": f"python -m real2sim.stage3.physics_probe {name}", "exit_code": 0}
             for name in ("objects", "static", "grasp")]
    result = {"status": "provisional", "real_world_verified": False,
              "static_report": str(output / "static/report.json"),
              "grasp_report": str(output / "grasp/report.json"), "candidate_scene": None}
    checked = verify_run(output, _sha256(PAPER_SCENE), PAPER_SCENE, {}, trace, result,
                         replay_physics=True)
    assert static["passed"] and grasp["passed"] and checked["passed"]
    assert checked["independent_replay"] == {"static_reproduced": True,
                                             "grasp_reproduced": True}
    report_path = output / "grasp/report.json"
    forged = json.loads(report_path.read_text())
    forged["trials"][0]["rise_m"] += 0.02
    report_path.write_text(json.dumps(forged), encoding="utf-8")
    rejected = verify_run(output, _sha256(PAPER_SCENE), PAPER_SCENE, {}, trace, result,
                          replay_physics=True)
    assert rejected["passed"] is False
    assert rejected["independent_replay"]["grasp_reproduced"] is False


def test_runner_replays_grasp_direction_and_damping(tmp_path):
    pytest.importorskip("mujoco")
    output = tmp_path / "run"
    output.mkdir()
    check_static(PAPER_SCENE, output=output / "static", duration=0.2)
    check_grasp(PAPER_SCENE, "blue_block", [0.6], output=output / "grasp",
                grasp_axis="y", jaw_damping=30)
    trace = [{"command": f"python -m real2sim.stage3.physics_probe {name}", "exit_code": 0}
             for name in ("objects", "static", "grasp")]
    result = {"status": "provisional", "real_world_verified": False,
              "static_report": str(output / "static/report.json"),
              "grasp_report": str(output / "grasp/report.json"), "candidate_scene": None}
    verified = verify_run(output, _sha256(PAPER_SCENE), PAPER_SCENE, {}, trace, result,
                          replay_physics=True)
    assert verified["independent_replay"]["grasp_reproduced"] is True


def test_runner_audits_articulated_scene_without_inventing_grasp(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "drawer.xml"
    write_probe_scene(scene)
    root = ET.parse(scene).getroot()
    body = root.find("worldbody/body[@name='cube']")
    body.set("pos", "0 0 0.02")
    body.remove(body.find("freejoint"))
    ET.SubElement(body, "joint", {"name": "drawer_slide", "type": "slide", "axis": "1 0 0"})
    ET.ElementTree(root).write(scene, encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    script = {"duration": 0.1, "settle": 0.01, "actions": [
        {"body": "cube", "type": "force", "duration": 0.08, "vector": [1, 0, 0]}],
        "checks": [{"body": "cube", "type": "move", "min_m": 1e-5}]}
    (output / "articulation_script.json").write_text(json.dumps(script), encoding="utf-8")
    static = check_static(scene, duration=0.15)
    articulate = run_probe(scene, script)
    assert static["passed"] and articulate["passed"]
    for directory, report in (("static", static), ("articulation", articulate)):
        folder = output / directory
        folder.mkdir()
        (folder / "report.json").write_text(json.dumps({key: value for key, value in report.items()
                                                          if key != "samples"}), encoding="utf-8")
    trace = [{"command": f"python -m real2sim.stage3.physics_probe {name} --scene {scene}",
              "exit_code": 0} for name in ("objects", "static", "run")]
    result = {"status": "provisional", "real_world_verified": False,
              "static_report": str(output / "static/report.json"), "grasp_report": None,
              "articulation_report": str(output / "articulation/report.json"), "candidate_scene": None}
    verified = verify_run(output, _sha256(scene), scene, {}, trace, result, replay_physics=True)
    assert verified["passed"] is True
    assert verified["sim_grasp_passed"] is None
    assert verified["sim_articulation_passed"] is True
    assert verified["independent_replay"]["articulation_reproduced"] is True


def test_runner_compares_failed_input_with_repaired_candidate(tmp_path):
    pytest.importorskip("mujoco")
    faulty = tmp_path / "floating.xml"
    output = tmp_path / "run"
    output.mkdir()
    candidate = output / "candidate.xml"
    patch_scene(PAPER_SCENE, faulty, {"blue_block": {"position": [-0.05, -0.12, 0.9]}})
    patch_scene(faulty, candidate, {"blue_block": {"position": [-0.05, -0.12, 0.773]}})
    static_before = check_static(faulty, output=output / "before_static", duration=0.3)
    grasp_before = check_grasp(faulty, "blue_block", [0.4], output=output / "before_grasp")
    check_grasp(faulty, "blue_block", [1, 3, 10], output=output / "before_grasp_higher_forces")
    static_after = check_static(candidate, output=output / "after_static", duration=0.3)
    grasp_after = check_grasp(candidate, "blue_block", [0.4], output=output / "after_grasp")
    assert not static_before["passed"] and not grasp_before["passed"]
    assert static_after["passed"] and grasp_after["passed"]
    trace = [{"command": f"python -m real2sim.stage3.physics_probe {name} --scene {faulty}",
              "exit_code": 0} for name in ("objects", "static", "grasp")]
    trace.extend({"command": f"python -m real2sim.stage3.physics_probe {name} --scene {candidate}",
                  "exit_code": 0} for name in ("static", "grasp"))
    trace.append({"command": f"python -m real2sim.stage3.physics_probe patch --output {candidate}",
                  "exit_code": 0})
    agent_result = {"status": "provisional", "real_world_verified": False,
                    "static_report": str(output / "after_static/report.json"),
                    "grasp_report": str(output / "after_grasp/report.json"),
                    "candidate_scene": str(candidate)}
    verified = verify_run(output, _sha256(faulty), faulty, {}, trace, agent_result,
                          replay_physics=True)
    assert verified["passed"] is True
    assert not verified["sim_static_passed"] and not verified["sim_grasp_passed"]
    assert verified["candidate_recheck"]["static_passed"]
    assert verified["candidate_recheck"]["grasp_passed"]
    assert all(verified["independent_replay"].values())
    assert not verify_run(output, _sha256(faulty), faulty, {}, trace[:-1], agent_result)["passed"]


def test_runner_merges_separate_baseline_force_ranges_for_candidate(tmp_path):
    pytest.importorskip("mujoco")
    output = tmp_path / "run"
    output.mkdir()
    candidate = output / "candidate.xml"
    patch_scene(PAPER_SCENE, candidate, {"blue_block": {"friction": 0.6}})
    check_static(PAPER_SCENE, output=output / "before_static", duration=0.2)
    check_grasp(PAPER_SCENE, "blue_block", [0.2], output=output / "before_low")
    check_grasp(PAPER_SCENE, "blue_block", [0.6], output=output / "before_high")
    check_static(candidate, output=output / "after_static", duration=0.2)
    check_grasp(candidate, "blue_block", [0.2, 0.6], output=output / "after_grasp")
    trace = [{"command": f"python -m real2sim.stage3.physics_probe {name} --scene {PAPER_SCENE}",
              "exit_code": 0} for name in ("objects", "static", "grasp")]
    trace.extend({"command": f"python -m real2sim.stage3.physics_probe {name} --scene {candidate}",
                  "exit_code": 0} for name in ("static", "grasp"))
    trace.append({"command": f"python -m real2sim.stage3.physics_probe patch --output {candidate}",
                  "exit_code": 0})
    result = {"status": "provisional", "real_world_verified": False,
              "static_report": str(output / "after_static/report.json"),
              "grasp_report": str(output / "after_grasp/report.json"),
              "candidate_scene": str(candidate)}
    verified = verify_run(output, _sha256(PAPER_SCENE), PAPER_SCENE, {}, trace, result,
                          replay_physics=True)
    assert verified["passed"] is True
    assert len(verified["baseline_report_paths"]["grasp"]) == 2


@pytest.mark.parametrize("failed", [False, True])
def test_runner_deletes_only_temporary_overview_after_sdk_consumes_it(tmp_path, monkeypatch, failed):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>", encoding="utf-8")
    hypotheses = tmp_path / "hypotheses.json"
    hypotheses.write_text('{"hypotheses": []}', encoding="utf-8")
    unrelated = tmp_path / "keep.png"
    unrelated.write_bytes(b"untouched")
    output = tmp_path / "output"

    def observe(_, destination):
        destination.mkdir(parents=True)
        photo = destination / "overview.png"
        photo.write_bytes(b"temporary image")
        (destination / "observation.json").write_text("{}", encoding="utf-8")
        return {"image": str(photo), "objects": []}

    class FakeThread:
        id = "fake-thread"

        def run(self, inputs, **kwargs):
            assert (output / "observe/overview.png").is_file()
            if failed:
                raise RuntimeError("SDK failure")
            return SimpleNamespace(id="fake-turn", status="completed", items=[],
                                   final_response='{"status": "provisional"}')

    class FakeCodex:
        archived = []

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def thread_start(self, **kwargs):
            assert kwargs["ephemeral"] is True
            assert kwargs["model"] == "gpt-6-astra"
            return FakeThread()

        def thread_archive(self, thread_id):
            self.archived.append(thread_id)

    fake_sdk = SimpleNamespace(ApprovalMode=SimpleNamespace(deny_all="deny_all"),
                               Codex=FakeCodex,
                               LocalImageInput=lambda path: path,
                               Sandbox=SimpleNamespace(full_access="full_access"),
                               TextInput=lambda text: text)
    monkeypatch.setitem(sys.modules, "openai_codex", fake_sdk)
    monkeypatch.setattr(agent_runner, "observe_scene", observe)
    monkeypatch.setattr(agent_runner, "verify_run", lambda *args, **kwargs: {
        "passed": True, "sim_static_passed": True, "sim_grasp_passed": True})

    result = agent_runner.run_agent(scene, hypotheses, output)
    assert (output / "observe/overview.png").exists() is False
    assert unrelated.read_bytes() == b"untouched"
    assert json.loads((output / "observe/observation.json").read_text())["overview_consumed_and_deleted"]
    assert "image" not in result["observation"]
    assert result["thread_id"] == "fake-thread"
    assert result["thread_cleanup"] == "archived"
    assert ("SDK failure" in result.get("error", "")) is failed


@pytest.mark.parametrize("invalid", [
    {"mass_range_kg": [False, 1]},
    {"mass_range_kg": [float("nan"), 1]},
    {"grasp_force_range_per_jaw_n": [4, 1]},
    {"sliding_friction_range": [-1, 2]},
    {"body": "not_a_free_body"},
])
def test_visual_force_prior_rejects_invalid_ranges(invalid):
    prior = {"body": "button", "material_guess": "plastic", "mass_range_kg": [0.1, 0.5],
             "sliding_friction_range": [0, 1], "grasp_force_range_per_jaw_n": [1, 5],
             "confidence": "low", "rationale": "uncertain visual size"}
    prior.update(invalid)
    with pytest.raises(ValueError):
        _validate_visual_prior(prior, ["button"])


def test_runner_retries_failed_physics_in_same_ephemeral_thread(tmp_path, monkeypatch):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>", encoding="utf-8")
    hypotheses = tmp_path / "hypotheses.json"
    hypotheses.write_text('{"hypotheses": []}', encoding="utf-8")
    output = tmp_path / "output"
    turns = []

    def observe(_, destination):
        destination.mkdir(parents=True)
        image = destination / "overview.png"
        image.write_bytes(b"image")
        return {"image": str(image), "objects": [{"body": "tissue", "joint": "free"}]}

    prior = {"body": "tissue", "material_guess": "paper", "mass_range_kg": [0.001, 0.01],
             "sliding_friction_range": [0, 1], "grasp_force_range_per_jaw_n": [0.2, 3],
             "confidence": "low", "rationale": "visible paper"}

    class FakeThread:
        id = "ephemeral"

        def run(self, inputs, **kwargs):
            turns.append((inputs, kwargs))
            assert kwargs["model"] == "gpt-6-astra" and kwargs["effort"] == "high"
            if len(turns) == 1:
                response = prior
            else:
                response = {"candidate_scene": None, "status": "needs_repair" if len(turns) == 2 else "provisional"}
            return SimpleNamespace(id=f"turn-{len(turns)}", status="completed", items=[],
                                   final_response=json.dumps(response))

    class FakeCodex:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def thread_start(self, **kwargs):
            assert kwargs["sandbox"] == "full_access"
            assert kwargs["ephemeral"] is True
            return FakeThread()

        def thread_archive(self, thread_id):
            assert thread_id == "ephemeral"

    monkeypatch.setitem(sys.modules, "openai_codex", SimpleNamespace(
        ApprovalMode=SimpleNamespace(deny_all="deny_all"), Codex=FakeCodex,
        LocalImageInput=lambda path: path, Sandbox=SimpleNamespace(full_access="full_access"),
        TextInput=lambda text: text))
    monkeypatch.setattr(agent_runner, "observe_scene", observe)
    checks = iter([{"passed": True, "sim_static_passed": False, "sim_grasp_passed": False},
                   {"passed": True, "sim_static_passed": True, "sim_grasp_passed": True}])
    monkeypatch.setattr(agent_runner, "verify_run", lambda *args, **kwargs: next(checks))

    result = agent_runner.run_agent(scene, hypotheses, output)
    assert len(turns) == 3
    assert "thin-sheet collision proxy" in turns[2][0][0]
    assert [turn["physics_ready"] for turn in result["turns"]] == [False, True]
    assert result["thread_cleanup"] == "archived"
    assert not (output / "observe/overview.png").exists()


@pytest.mark.parametrize("prior_runs_command", [False, True])
def test_runner_estimates_force_without_simulator_inputs(tmp_path, monkeypatch, prior_runs_command):
    scene = tmp_path / "scene.xml"
    scene.write_text('<mujoco mass="secret_simulator_mass"/>', encoding="utf-8")
    hypotheses = tmp_path / "hypotheses.json"
    hypotheses.write_text('{"hypotheses": []}', encoding="utf-8")
    output = tmp_path / "output"
    calls = []
    prior = {"body": "button", "material_guess": "plastic", "mass_range_kg": [0.1, 0.5],
             "sliding_friction_range": [0.2, 1.2], "grasp_force_range_per_jaw_n": [1, 5],
             "confidence": "low", "rationale": "uncertain visual size"}

    def observe(_, destination):
        destination.mkdir(parents=True)
        photo = destination / "overview.png"
        photo.write_bytes(b"temporary image")
        return {"image": str(photo), "objects": [{"body": "button", "joint": "free",
                 "mass_kg": "secret_simulator_mass"}]}

    class FakeThread:
        id = "fake-thread"

        def run(self, inputs, **kwargs):
            calls.append((inputs, kwargs))
            if len(calls) == 1:
                return SimpleNamespace(id="prior-turn", items=[SimpleNamespace(root=SimpleNamespace(
                    type="commandExecution", command="cat scene.xml", exit_code=0,
                    status="completed", aggregated_output=""))] if prior_runs_command else [],
                    final_response=json.dumps(prior))
            return SimpleNamespace(id="physics-turn", status="completed", items=[],
                                   final_response='{"status":"provisional"}')

    class FakeCodex:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def thread_start(self, **kwargs):
            assert kwargs["ephemeral"] is True
            assert kwargs["model"] == "gpt-6-astra"
            return FakeThread()

        def thread_archive(self, thread_id):
            assert thread_id == "fake-thread"

    monkeypatch.setitem(sys.modules, "openai_codex", SimpleNamespace(
        ApprovalMode=SimpleNamespace(deny_all="deny_all"), Codex=FakeCodex,
        LocalImageInput=lambda path: path, Sandbox=SimpleNamespace(full_access="full_access"),
        TextInput=lambda text: text))
    monkeypatch.setattr(agent_runner, "observe_scene", observe)
    monkeypatch.setattr(agent_runner, "verify_run", lambda *args, **kwargs: {
        "passed": True, "sim_static_passed": True, "sim_grasp_passed": True})

    result = agent_runner.run_agent(scene, hypotheses, output, task_context="grasp a button")
    prior_inputs, prior_kwargs = calls[0]
    assert prior_kwargs["effort"] == "high"
    assert prior_kwargs["output_schema"] == agent_runner.PRIOR_SCHEMA
    assert "secret_simulator_mass" not in prior_inputs[0]
    assert str(scene) not in prior_inputs[0]
    if prior_runs_command:
        assert len(calls) == 1
        assert "must not execute commands" in result["error"]
        assert not (output / "visual_force_prior.json").exists()
    else:
        assert len(calls) == 2
        assert json.loads((output / "visual_force_prior.json").read_text()) == prior
        assert "secret_simulator_mass" not in (output / "visual_force_prior_prompt.md").read_text()
        assert str(output / "visual_force_prior.json") in calls[1][0][0]
        assert result["verification"]["passed"]
        assert "image" not in result["observation"]
