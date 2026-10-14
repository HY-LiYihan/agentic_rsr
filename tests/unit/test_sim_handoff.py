import importlib.util
import json
from pathlib import Path
import sys

import pytest

from sim.policy_bundle import PolicyBundle


def test_accepted_refinement_delivers_latest_policy_and_authoring_history(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "scripts" / "run_sim_codex_probe.py"
    spec = importlib.util.spec_from_file_location("run_sim_codex_probe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "policy.py").write_text('def run(api):\n    return {"status": "failed"}\n')
    output = tmp_path / "run"
    shared_home = tmp_path / "runs" / "sim_codex_home"

    class FakeSession:
        def __init__(self, model, cwd, transcript, codex_home=None):
            self.transcript = transcript
            self.id = "author-thread"
            self.codex_home = codex_home or output / "codex_home"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def bootstrap(self, prompt):
            self.prompt = prompt

    def fake_refinement(scene, task, bundle, run_output, host, port, controller, **kwargs):
        refined = run_output / "policy" / "iteration_001" / "bundle"
        refined.mkdir(parents=True)
        (refined / "policy.py").write_text('def run(api):\n    return {"status": "completed"}\n')
        return {"accepted": True, "locked": True, "reset_count": 4,
                "candidate_bundle": PolicyBundle(refined).manifest(),
                "iterations": [{"iteration": 0, "policy_hash": bundle.hash,
                                "exact": {"success": False}},
                               {"iteration": 1, "policy_hash": PolicyBundle(refined).hash,
                                "exact": {"success": True},
                                "randomized": {"success": True}}],
                "final_evaluation": {"accepted": True}}

    monkeypatch.setattr(module, "SimCodexSession", FakeSession)
    monkeypatch.setattr(module, "run_policy_refinement", fake_refinement)
    monkeypatch.setenv("SAM3_HOST", "test-host")
    monkeypatch.setenv("SAM3_PORT", "1234")
    monkeypatch.setattr(sys, "argv", ["run_sim_codex_probe.py", "--scene", str(tmp_path / "scene.xml"),
                                  "--source", "blue block", "--target", "yellow bowl",
                                  "--bundle", str(initial), "--output", str(output),
                                  "--codex-home", str(shared_home)])
    module.main()
    handoff = json.loads((output / "real_handoff.json").read_text())
    assert Path(handoff["policy"]).read_text() == (
        'def run(api):\n    return {"status": "completed"}\n')
    assert handoff["policy_hash"] == PolicyBundle(output / "final_policy").hash
    assert handoff["authoring_thread_id"] == "author-thread"
    assert handoff["codex_home"] == str(shared_home)
    assert handoff["model"] == "gpt-6-astra"
    assert json.loads((output / "authoring_thread.json").read_text()) == {
        "authoring_thread_id": handoff["authoring_thread_id"],
        "codex_home": handoff["codex_home"],
        "model": handoff["model"],
    }
    messages = [json.loads(line) for line in (output / "authoring_conversation.jsonl").read_text().splitlines()]
    assert [message["type"] for message in messages] == ["task", "supplied_bundle"]

    monkeypatch.setattr(sys, "argv", ["run_sim_codex_probe.py", "--scene", str(tmp_path / "scene.xml"),
                                  "--source", "blue block", "--target", "yellow bowl",
                                  "--output", str(tmp_path / "rejected"),
                                  "--codex-home", str(Path.home() / ".codex")])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
