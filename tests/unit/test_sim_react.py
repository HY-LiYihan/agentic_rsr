import pytest

from sim.policy_bundle import PolicyBundle
from sim.policy_loop import run_policy_loop
from sim.policy_terms import ADAPTATION_MODE, LOOP_NAME, POLICY_TYPE
from sim.react_runtime import run_react


class FakeTools:
    stopped = False

    def __init__(self):
        self.calls = []
        self.observe_count = 0

    def dispatch(self, name, *args):
        self.calls.append((name, args))
        if name == "observe":
            self.observe_count += 1
            return {"objects": {"blue block": {"xy_m": [0.01 * self.observe_count, 0.0]}},
                    "image": None}
        if name == "robot.gripper":
            return {"status": "completed", "width_m": args[0]}
        if name == "task.check_success":
            return True
        if name == "task.stop":
            self.stopped = True
            return None
        raise ValueError(name)


def write_bundle(tmp_path):
    root = tmp_path / "bundle"
    (root / "steps").mkdir(parents=True)
    (root / "policy.py").write_text('''def run(api):
    observation = api.observe()
    return {"status": "needs_agent", "state": {"phase": "located",
            "x": observation["objects"]["blue block"]["xy_m"][0]}}
''')
    (root / "steps" / "grasp.py").write_text('''def run(api, state):
    api.robot.gripper(0.08)
    return {"status": "completed", "state": {"phase": "grasped",
            "previous": state["phase"]}}
''')
    return root


def test_policy_bundle_requires_policy_and_exposes_named_steps(tmp_path):
    root = write_bundle(tmp_path)
    bundle = PolicyBundle(root)
    assert set(bundle.sources) == {"run", "step_grasp"}
    assert len(bundle.hash) == 64
    assert bundle.manifest()["policy_type"] == POLICY_TYPE
    assert bundle.manifest()["loop"] == LOOP_NAME
    assert bundle.manifest()["adaptation_mode"] == ADAPTATION_MODE
    assert bundle.manifest()["files"] == ["policy.py", "steps/grasp.py"]

    (root / "policy.py").unlink()
    with pytest.raises(ValueError, match="policy.py"):
        PolicyBundle(root)


def test_policy_execution_loop_returns_every_step_and_direct_tool_to_controller(tmp_path):
    bundle = PolicyBundle(write_bundle(tmp_path))
    tools = FakeTools()
    commands = iter((
        {"action": "run_policy", "name": "step_grasp", "args": [], "reason": "grasp"},
        {"action": "call_tool", "name": "task.check_success", "args": [], "reason": "verify"},
        {"action": "finish", "name": "", "args": [], "reason": "done"},
    ))
    requests = []

    def decide(request):
        requests.append(request)
        return next(commands)

    result = run_policy_loop(bundle, tools, decide, "exact")
    assert result["status"] == "finished"
    assert result["return"]["status"] == "completed"
    assert [item["action"] for item in result["history"]] == [
        "run_policy", "run_policy", "call_tool"]
    assert len(requests) == 3
    assert requests[0]["last_result"]["name"] == "run"
    assert requests[0]["last_result"]["result"]["return"]["state"]["phase"] == "located"
    assert requests[1]["last_result"]["result"]["return"]["status"] == "completed"
    assert tools.calls.count(("robot.gripper", (0.08,))) == 1
    assert any(call[0] == "task.check_success" for call in tools.calls)


def test_policy_execution_loop_rejects_unavailable_direct_tool(tmp_path):
    bundle = PolicyBundle(write_bundle(tmp_path))
    tools = FakeTools()
    result = run_policy_loop(bundle, tools,
                       lambda request: {"action": "call_tool", "name": "scene.truth",
                                        "args": [], "reason": "bad"}, "randomized",
                       max_turns=1)
    assert result["status"] == "turn_limit"
    assert "Unknown direct tool" in result["history"][1]["error"]
    assert result["loop"] == LOOP_NAME


def test_complete_policy_runs_without_codex_decision(tmp_path):
    root = write_bundle(tmp_path)
    (root / "policy.py").write_text('''def run(api):
    api.observe()
    api.robot.gripper(0.08)
    return {"status": "completed"}
''')
    tools = FakeTools()

    def no_decision(request):
        raise AssertionError("Uninterrupted policy must not call Codex")

    result = run_policy_loop(PolicyBundle(root), tools, no_decision, "exact")
    assert result["status"] == "finished"
    assert result["return"]["status"] == "completed"
    assert len(result["history"]) == 1
    assert tools.calls.count(("robot.gripper", (0.08,))) == 1


def test_policy_checkpoint_uses_public_result_and_resumes_same_run(tmp_path):
    root = write_bundle(tmp_path)
    (root / "policy.py").write_text('''def run(api):
    observation = api.observe()
    choice = api.agent.ask("grasp_review", "Grasp complete?", ["retry", "continue", "abort"])
    return {"status": "completed" if choice == "continue" else "failed",
            "seen_x": observation["objects"]["blue block"]["xy_m"][0]}
''')
    class CheckpointTools(FakeTools):
        def dispatch(self, name, *args):
            if name == "agent.ask":
                return {"label": args[0], "question": args[1], "options": args[2],
                        "observation": self.dispatch("observe"),
                        "recent_actions": [{"tool": "observe"}]}
            return super().dispatch(name, *args)

    class Decision:
        def __init__(self):
            self.checkpoints = []

        def checkpoint(self, request):
            self.checkpoints.append(request)
            return "continue"

        def __call__(self, request):
            raise AssertionError("Completed checkpoint policy must not need another turn")

    decide = Decision()
    result = run_policy_loop(PolicyBundle(root), CheckpointTools(), decide, "exact")
    assert result["return"]["status"] == "completed"
    assert decide.checkpoints[0]["label"] == "grasp_review"
    assert decide.checkpoints[0]["running_command"] == {"action": "run_policy", "name": "run"}
    assert decide.checkpoints[0]["recent_actions"] == [{"tool": "observe"}]


def test_legacy_react_name_is_only_a_compatibility_alias():
    assert run_react is run_policy_loop
