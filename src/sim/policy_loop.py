"""Execution-grounded policy execution loop for one persistent SIM episode."""

from __future__ import annotations

import json

from .policy_terms import POLICY_TYPE, LOOP_NAME


DIRECT_TOOLS = frozenset({"observe", "robot.survey", "robot.move_pose", "robot.gripper",
                          "robot.pick_place", "task.check_success", "task.stop"})


def _json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


def run_policy_loop(bundle, tools, decide, profile: str, max_turns: int = 12) -> dict:
    """Orchestrate locked programmatic policies using public execution feedback."""
    decision_checkpoint = getattr(decide, "checkpoint", None)
    running_policy = "run"

    def handle_checkpoint(request):
        choice = decision_checkpoint({"profile": profile,
                                      "running_command": {"action": "run_policy",
                                                          "name": running_policy},
                                      **request})
        if hasattr(tools, "record_agent_choice"):
            tools.record_agent_choice(request["label"], choice)
        return choice

    checkpoint = handle_checkpoint if decision_checkpoint is not None else None
    initial = bundle.execute("run", {}, tools.dispatch, on_pause=checkpoint)
    history = [{"turn": 0, "action": "run_policy", "name": "run",
                "policy_type": POLICY_TYPE, "result": _json_value(initial)}]
    state = {}
    outcome = initial.get("return")
    if initial["status"] == "finished" and isinstance(outcome, dict):
        if isinstance(outcome.get("state"), dict):
            state = _json_value(outcome["state"])
        completed = (outcome.get("status") == "completed"
                     or outcome.get("success") is True)
        if completed:
            return {"status": "finished", "return": outcome, "history": history,
                    "loop": LOOP_NAME, "policy_type": POLICY_TYPE,
                    "policy_bundle_hash": bundle.hash}
    if tools.stopped:
        return {"status": "finished", "return": {"status": "failed", "reason": "Stopped"},
                "history": history, "loop": LOOP_NAME,
                "policy_type": POLICY_TYPE, "policy_bundle_hash": bundle.hash}
    observation = tools.dispatch("observe")
    for turn in range(max_turns):
        request = {"loop": LOOP_NAME, "policy_type": POLICY_TYPE,
                   "policy_bundle": bundle.manifest(), "profile": profile, "turn": turn,
                   "available_policies": list(bundle.sources),
                   "available_tools": sorted(DIRECT_TOOLS), "observation": observation,
                   "state": state, "last_result": history[-1] if history else None}
        command = decide(_json_value(request))
        if not isinstance(command, dict):
            raise ValueError("Codex turn must return a structured command")
        action, name = command.get("action"), command.get("name")
        if action in ("finish", "abort"):
            if action == "abort":
                tools.dispatch("task.stop")
            return {"status": "finished", "return": {"status": "completed" if action == "finish"
                                                     else "failed", "reason": command.get("reason", "")},
                    "history": history, "loop": LOOP_NAME,
                    "policy_type": POLICY_TYPE,
                    "policy_bundle_hash": bundle.hash}
        entry = {"turn": turn, "action": action, "name": name,
                 "policy_type": POLICY_TYPE}
        try:
            if action == "run_policy":
                if name not in bundle.sources:
                    raise ValueError("Unknown policy step")
                running_policy = name
                result = bundle.execute(name, _json_value(state), tools.dispatch,
                                        on_pause=checkpoint)
                entry["result"] = _json_value(result)
                outcome = result.get("return")
                if result["status"] == "finished" and isinstance(outcome, dict) and "state" in outcome:
                    if not isinstance(outcome["state"], dict):
                        raise ValueError("Step state must be a JSON object")
                    state = _json_value(outcome["state"])
            elif action == "call_tool":
                if name not in DIRECT_TOOLS:
                    raise ValueError("Unknown direct tool")
                args = command.get("args", [])
                if not isinstance(args, list):
                    raise ValueError("Tool arguments must be a JSON array")
                entry["result"] = _json_value(tools.dispatch(name, *args))
            else:
                raise ValueError("Unknown Codex action")
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        history.append(entry)
        if tools.stopped:
            return {"status": "finished", "return": {"status": "failed", "reason": "Stopped"},
                    "history": history, "loop": LOOP_NAME,
                    "policy_type": POLICY_TYPE,
                    "policy_bundle_hash": bundle.hash}
        try:
            observation = tools.dispatch("observe")
        except Exception as exc:
            observation = {"error": f"{type(exc).__name__}: {exc}"}
        entry["observation_after"] = _json_value(observation)
    return {"status": "turn_limit", "error": f"Exceeded {max_turns} policy turns",
            "history": history, "loop": LOOP_NAME,
            "policy_bundle_hash": bundle.hash}
