import numpy as np
import pytest

from sim.codex_controller import CodexDecisionController
from sim.harness import EpisodeTools, TabletopTask, near_release_pose, prompt_for
from sim.observation import ExactObservation, SAM3InstanceSegmenter
from sim.program_runner import run_program, validate_program


def dispatch(name, *args):
    if name == "observe":
        return {"objects": {"blue block": {"xy_m": [-0.05, -0.12]},
                            "yellow bowl": {"xy_m": [0.05, 0.08]}}}
    if name == "robot.pick_place":
        assert args == ("blue block", [-0.05, -0.12], [0.05, 0.08])
        return {"status": "completed"}
    if name == "task.check_success":
        return True
    raise ValueError(name)


def test_prompt_advertises_tools_and_both_profiles():
    task = TabletopTask("blue block", "yellow bowl", "Pick up the blue block")
    prompt = prompt_for(task)
    assert "api.observe()" in prompt
    assert "api.robot.pick_place" in prompt
    assert "api.robot.survey" in prompt
    assert "api.task.check_success" in prompt
    assert "api.task.checkpoint" not in prompt
    assert "api.agent.ask" in prompt
    assert "SAME conversation" in prompt
    assert "observe repeatedly after ANY movement" in prompt
    assert "EXACT" in prompt and "RANDOMIZED" in prompt
    assert "IDENTICAL file" in prompt and "Prefer uninterrupted execution" in prompt
    assert "WORLD metres" in prompt and "z=0.75" in prompt


def test_program_runs_through_public_rpc():
    source = '''def run(api):
    objects = api.observe()["objects"]
    api.robot.pick_place("blue block", objects["blue block"]["xy_m"],
                         objects["yellow bowl"]["xy_m"])
    return {"success": api.task.check_success()}
'''
    result = run_program(source, dispatch, timeout=10)
    assert result == {"status": "finished", "return": {"success": True}}


def test_program_can_reobserve_without_pausing():
    observed = []

    def observation_dispatch(name, *args):
        observed.append((name, args))
        if name == "observe":
            if len(observed) == 1:
                return {"objects": {"blue block": {"xy_m": [0.0, 0.0]}}}
            return {"objects": {"blue block": {"xy_m": [0.02, 0.0]}},
                    "image": None}
        raise ValueError(name)

    source = '''def run(api):
    before = api.observe()["objects"]["blue block"]["xy_m"]
    after = api.observe()["objects"]["blue block"]["xy_m"]
    return {"status": "completed" if after != before else "failed"}
'''
    result = run_program(source, observation_dispatch, timeout=10)
    assert result == {"status": "finished", "return": {"status": "completed"}}
    assert observed == [("observe", ()), ("observe", ())]


def test_exact_observation_is_fresh_and_does_not_expose_scene_handles():
    class Scene:
        positions = {"blue block": (-0.05, -0.12, 0.77),
                     "yellow bowl": (0.05, 0.08, 0.75)}

        def object_xyz(self, name):
            return self.positions[name]

    scene = Scene()
    perception = ExactObservation(scene, ("blue block", "yellow bowl"))
    first = perception.observe()
    assert first["objects"]["blue block"]["xy_m"] == [-0.05, -0.12]
    assert first["image"] is None
    assert "scene" not in first and "ground_truth" not in first
    scene.positions["blue block"] = (0.05, 0.08, 0.77)
    assert perception.observe()["objects"]["blue block"]["xy_m"] == [0.05, 0.08]


def test_agent_ask_suspends_worker_and_resumes_named_branch():
    class Perception:
        def observe(self):
            return {"objects": {"blue block": {"xy_m": [0.0, 0.0]}}, "image": None}

    tools = EpisodeTools(None, None, Perception(), None)
    requests = []

    def decide(request):
        requests.append(request)
        return "retry_grasp"

    source = '''def run(api):
    api.observe()
    api.observe()
    choice = api.agent.ask("grasp_review", "Is it grasped?", ["retry_grasp", "continue", "abort"])
    if choice == "retry_grasp":
        return {"status": "completed", "branch": choice, "next_step": "retry"}
    return {"status": "failed"}
'''
    result = run_program(source, tools.dispatch, timeout=10, on_pause=decide)
    assert result == {"status": "finished", "return": {
        "status": "completed", "branch": "retry_grasp", "next_step": "retry"}}
    assert len(requests) == 1
    assert requests[0]["label"] == "grasp_review"
    assert requests[0]["observation"]["objects"]["blue block"]["xy_m"] == [0.0, 0.0]
    assert [call["tool"] for call in tools.calls] == [
        "observe", "observe", "agent.ask", "observe"]


def test_abort_stops_further_actions():
    tools = EpisodeTools(None, None, None, None)
    source = '''def run(api):
    api.agent.ask("review", "Continue?", ["continue", "abort"])
    api.robot.gripper(0.08)
    return {"status": "completed"}
'''
    tools.perception = type("Perception", (), {"observe": lambda self: {"objects": {}}})()
    result = run_program(source, tools.dispatch, timeout=10, on_pause=lambda request: "abort")
    assert result["status"] == "failed"
    assert "Episode stopped" in result["error"]
    assert not any(call["tool"] == "robot.gripper" for call in tools.calls)


def test_stop_remains_available_after_tool_call_limit():
    tools = EpisodeTools(None, None, None, None)
    tools.calls = [{"tool": "observe"}] * 32
    with pytest.raises(RuntimeError, match="Tool-call limit exceeded"):
        tools.dispatch("observe")
    tools.dispatch("task.stop")
    assert tools.stopped
    assert tools.calls[-1]["tool"] == "task.stop"


def test_agent_can_retry_earlier_stage_in_same_worker():
    class Perception:
        def observe(self):
            return {"objects": {"blue block": {"xy_m": [0.0, 0.0]}}, "image": None}

    tools = EpisodeTools(None, None, Perception(), None)
    choices = iter(("retry_grasp", "continue"))
    steps = []

    def dispatch(name, *args):
        if name == "robot.gripper":
            steps.append(args[0])
            return {"status": "completed"}
        return tools.dispatch(name, *args)

    source = '''def run(api):
    for attempt in range(2):
        api.robot.gripper(0.08)
        api.observe()
        choice = api.agent.ask("grasp_review", "Grasped?", ["retry_grasp", "continue", "abort"])
        if choice == "continue":
            return {"status": "completed", "attempts": attempt + 1}
        if choice == "abort":
            return {"status": "failed"}
    return {"status": "failed"}
'''
    result = run_program(source, dispatch, timeout=10, on_pause=lambda request: next(choices))
    assert result == {"status": "finished", "return": {"status": "completed", "attempts": 2}}
    assert steps == [0.08, 0.08]
    assert [call["tool"] for call in tools.calls].count("agent.ask") == 2


def test_agent_ask_requires_controller_and_valid_branches():
    tools = EpisodeTools(None, None, None, None)
    with pytest.raises(ValueError, match="including abort"):
        tools.agent_ask("review", "Continue?", ["retry", "continue"])
    result = run_program('''def run(api):
    api.agent.ask("review", "Continue?", ["continue", "abort"])
''', tools.dispatch, timeout=10)
    assert result["status"] == "failed"
    assert "No Codex decision controller" in result["error"]


def test_codex_controller_reuses_sdk_session_with_public_request(tmp_path):
    import json

    class FakeSession:
        id = "sdk-thread"

        def __init__(self):
            self.calls = []

        def run(self, message, schema, output, images):
            self.calls.append((message, schema, images))
            return {"choice": "retry", "reason": "object seems missing"}

    session = FakeSession()
    controller = CodexDecisionController("pick the block", session, tmp_path)
    request = {"profile": "randomized", "label": "grasp_review",
               "question": "Is it grasped?", "options": ["retry", "abort"],
               "observation": {"objects": {}, "image": None}, "recent_actions": []}
    assert controller(request) == "retry"
    assert controller(request) == "retry"
    assert len(session.calls) == 2
    assert all("Current public state:" in message for message, _, _ in session.calls)
    assert "pick the block" in session.calls[0][0]
    assert "pick the block" not in session.calls[1][0]
    assert json.loads((tmp_path / "decisions/002/decision.json").read_text())["session_id"] == session.id


@pytest.mark.parametrize("source", [
    "import os\ndef run(api): pass",
    "def run(api): return api._connection",
    "def run(api): return api.__class__",
    "def other(api): pass",
])
def test_program_rejects_private_or_unstructured_code(source):
    with pytest.raises(ValueError):
        validate_program(source)


def test_sam3_selects_small_block_among_high_scoring_instances(monkeypatch, tmp_path):
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, timeout):
            pass

    first = np.zeros((2, 3), dtype=np.uint8)
    second = first.copy()
    first[0, 0] = 1
    second[1, :] = 1
    monkeypatch.setattr("sim.observation.socket.create_connection", lambda *args, **kwargs: Connection())
    monkeypatch.setattr("sim.observation.send_frame", lambda *args: None)
    monkeypatch.setattr("sim.observation.receive_arrays", lambda *args, **kwargs: (
        {"masks": np.stack((first, second)), "scores": np.array([0.92, 0.94])}, {},
    ))
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")
    mask = SAM3InstanceSegmenter("example", 42).segment(image, "yellow block", (2, 3))
    np.testing.assert_array_equal(mask, first.astype(bool))


def test_survey_merges_only_observed_objects():
    class Perception:
        calls = 0

        def observe(self):
            self.calls += 1
            objects = {"yellow block": {"xy_m": [0.0, 0.17]}} if self.calls == 1 else {
                "red bowl": {"xy_m": [-0.18, 0.17]}}
            return {"objects": objects, "camera": "wrist"}

    class Episode:
        def move_pose(self, *args, **kwargs):
            pass

    tools = EpisodeTools(None, Episode(), Perception(),
                         TabletopTask("yellow block", "red bowl", "move block"))
    assert list(tools.dispatch("observe")["objects"]) == ["yellow block"]
    surveyed = tools.dispatch("robot.survey")
    assert set(surveyed["objects"]) == {"yellow block", "red bowl"}
    assert sum(call["tool"] == "robot.move_pose" for call in tools.calls) == 1


def test_release_contact_only_accepts_near_target_and_height():
    assert near_release_pose((0.052, 0.08, 0.94), (0.05, 0.08), 0.93)
    assert not near_release_pose((0.07, 0.08, 0.94), (0.05, 0.08), 0.93)
    assert not near_release_pose((0.05, 0.08, 0.96), (0.05, 0.08), 0.93)
