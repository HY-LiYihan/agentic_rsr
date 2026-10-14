import json
import sys
from types import SimpleNamespace

from sim.codex_sdk import SimCodexSession
from sim.policy_loop_controller import CodexPolicyLoopController, generate_bundle


def test_sdk_episode_forks_persistent_history_without_mutating_it(tmp_path, monkeypatch):
    calls = []

    class FakeThread:
        def __init__(self, thread_id):
            self.id = thread_id
            self.turns = 0

        def run(self, inputs, **kwargs):
            self.turns += 1
            calls.append((inputs, kwargs))
            properties = kwargs["output_schema"]["properties"]
            if "acknowledged" in properties:
                response = {"acknowledged": True}
            elif "files" in properties:
                response = {"files": [{"path": "policy.py", "source":
                                      'def run(api):\n    return {"status": "completed"}\n'}]}
            else:
                response = {"action": "finish", "name": "", "args_json": "[]", "reason": "done"}
            return SimpleNamespace(id=f"turn-{self.turns}", status="completed",
                                   final_response=json.dumps(response))

    class FakeCodex:
        def __init__(self, config=None):
            calls.append(("config", config.env if config is not None else {}))
            self.starts = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append(("closed", args))

        def thread_start(self, **kwargs):
            self.starts += 1
            calls.append(("started", kwargs))
            return FakeThread("private-sim-thread")

        def thread_fork(self, thread_id, **kwargs):
            calls.append(("forked", thread_id, kwargs))
            return FakeThread("temporary-evaluation-thread")

        def thread_resume(self, thread_id, **kwargs):
            calls.append(("resumed", thread_id, kwargs))
            return FakeThread(thread_id)

    monkeypatch.setitem(sys.modules, "openai_codex", SimpleNamespace(
        Codex=FakeCodex,
        CodexConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        Sandbox=SimpleNamespace(read_only="read-only"),
        ApprovalMode=SimpleNamespace(deny_all="deny_all"),
        TextInput=lambda text: SimpleNamespace(text=text),
        LocalImageInput=lambda path: SimpleNamespace(path=path),
    ))
    transcript = tmp_path / "authoring_conversation.jsonl"
    with SimCodexSession("test-model", tmp_path, transcript) as session:
        bundle_root, session_id = generate_bundle("make policy", session, tmp_path / "run")
        assert bundle_root.joinpath("policy.py").is_file()
        assert session_id == session.id == "private-sim-thread"
        controller = CodexPolicyLoopController("make policy", session, tmp_path / "run",
                                               generated=True)
        observation = tmp_path / "frame.png"
        observation.write_bytes(b"public image")
        with controller.episode(tmp_path / "run" / "exact") as decide:
            decision = decide({"available_policies": ["run"],
                               "available_tools": ["observe"],
                               "observation": {"image": str(observation)}})
        assert decision["action"] == "finish"
        assert controller.session_id == session_id
        assert calls[0] == ("config", {})
        assert session.codex_home is None
        assert calls[1][1]["ephemeral"] is False
        assert calls[1][1]["sandbox"] == "read-only"
        assert calls[1][1]["approval_mode"] == "deny_all"
        assert calls[2][1]["effort"] == calls[4][1]["effort"] == "high"
        assert calls[3][0:2] == ("forked", session_id)
        assert calls[3][2]["ephemeral"] is True
        assert "make policy" not in calls[4][0][0].text
        assert calls[4][0][1].path == str(observation)
        assert len(transcript.read_text().splitlines()) == 1
        assert json.loads(transcript.read_text())["prompt"] == "make policy"
    assert calls[-1][0] == "closed"
    assert session.id is None

    with SimCodexSession("test-model", tmp_path, tmp_path / "supplied_conversation.jsonl") as author:
        controller = CodexPolicyLoopController("original task", author, tmp_path / "supplied")
        with controller.episode(tmp_path / "supplied" / "exact") as decide:
            assert decide({"available_policies": ["run"], "available_tools": ["observe"],
                           "observation": {"image": None}})["action"] == "finish"
        assert author.id == "private-sim-thread"
        assert json.loads((tmp_path / "supplied_conversation.jsonl").read_text())["prompt"] == "original task"
        prompt = (tmp_path / "supplied/exact/codex/turns/001/prompt.md").read_text()
        assert "Original SIM authoring conversation" not in prompt
    assert sum(call[0] == "started" for call in calls if isinstance(call[0], str)) == 2
    assert sum(call[0] == "forked" for call in calls if isinstance(call[0], str)) == 2


def test_sdk_resumes_from_same_private_home(tmp_path, monkeypatch):
    calls = []

    class FakeCodex:
        def __init__(self, config):
            calls.append(config.env)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def thread_resume(self, thread_id, **options):
            calls.append((thread_id, options))
            return SimpleNamespace(id=thread_id)

    monkeypatch.setitem(sys.modules, "openai_codex", SimpleNamespace(
        Codex=FakeCodex, CodexConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        Sandbox=SimpleNamespace(read_only="read-only"),
        ApprovalMode=SimpleNamespace(deny_all="deny_all"),
    ))
    home = tmp_path / "runs" / "codex_home"
    with SimCodexSession("test-model", tmp_path, codex_home=home,
                         thread_id="author-thread") as resumed:
        assert resumed.id == "author-thread"
        assert calls[0] == {"CODEX_HOME": str(home.resolve())}
        assert calls[1][1]["sandbox"] == "read-only"
        assert home.stat().st_mode & 0o777 == 0o700


def test_sdk_bootstrap_keeps_logs_in_run_with_shared_home(tmp_path, monkeypatch):
    transcript = tmp_path / "run" / "authoring_conversation.jsonl"
    session = SimCodexSession("test-model", tmp_path, transcript,
                              codex_home=tmp_path / "runs" / "shared_home")
    calls = []
    monkeypatch.setattr(session, "run", lambda message, schema, output: calls.append(output))

    session.bootstrap("original task")

    assert calls == [transcript.parent / "bootstrap"]


def test_checkpoint_uses_temporary_thread_and_public_intermediate_state(tmp_path):
    class FakeChild:
        id = "temporary"
        discarded = False

        def run(self, message, schema, output, images=()):
            self.message = message
            self.images = images
            return {"choice": "continue", "reason": "visible block"}

        def discard(self):
            self.discarded = True

    class FakeAuthor:
        id = "author"

        def fork(self, bootstrap_prompt):
            self.child = FakeChild()
            return self.child

    author = FakeAuthor()
    controller = CodexPolicyLoopController("original task", author, tmp_path, generated=True)
    image = tmp_path / "checkpoint.png"
    image.write_bytes(b"frame")
    with controller.episode(tmp_path / "episode") as decide:
        choice = decide.checkpoint({"profile": "randomized",
                                    "running_command": {"action": "run_policy", "name": "step_grasp"},
                                    "label": "grasp_review",
                                    "question": "Is it held?", "options": ["continue", "abort"],
                                    "observation": {"image": str(image)},
                                    "recent_actions": [{"tool": "robot.gripper"}]})
        assert choice == "continue"
    assert author.id == "author"
    assert author.child.discarded is True
    assert '"action": "run_policy"' in author.child.message
    assert '"name": "step_grasp"' in author.child.message
    assert "robot.gripper" in author.child.message
    assert author.child.images == (image,)
