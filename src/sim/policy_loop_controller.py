"""Codex policy-loop turns for one locked programmatic policy bundle."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from .codex_sdk import SimCodexSession


def _bundle_schema() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {"files": {"type": "array", "minItems": 1, "maxItems": 17,
                                   "items": {"type": "object", "additionalProperties": False,
                                             "properties": {"path": {"type": "string"},
                                                            "source": {"type": "string"}},
                                             "required": ["path", "source"]}}},
        "required": ["files"],
    }


def _write_bundle_payload(payload: dict, output_dir: Path):
    files = payload.get("files")
    if not isinstance(files, list) or not any(item.get("path") == "policy.py" for item in files):
        raise ValueError("Codex bundle must include policy.py")
    bundle_root = output_dir / "bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)
    seen = set()
    for item in files:
        relative = Path(item.get("path", ""))
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts
                or relative.suffix != ".py" or relative.as_posix() != str(relative)):
            raise ValueError("Invalid generated policy path")
        if relative in seen or (relative != Path("policy.py")
                                and relative.parent != Path("steps")):
            raise ValueError("Generated bundle paths must be unique policy.py or steps/*.py")
        source = item.get("source")
        if not isinstance(source, str):
            raise ValueError("Generated policy source must be a string")
        seen.add(relative)
        destination = bundle_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
    from .policy_bundle import PolicyBundle
    return bundle_root, PolicyBundle(bundle_root)


def policy_loop_prompt_for(task) -> str:
    if task.scene_kind == "stage3_emergency":
        return f"""Task: {task.instruction}
Write a complete reusable SIM policy bundle. Return JSON with files containing path and source; at least policy.py is required and defines def run(api):. Optional steps/<name>.py define def run(api, state):. No imports, file access, simulator handles, hidden state, or identifiers beginning with _. You can observe repeatedly; api.agent.ask(label, question, options) suspends for Codex judgement when genuinely ambiguous. Return {{"status": "completed"}} only after verified success; do not call api.task.stop() on successful completion, and use it only for unrecoverable aborts. The same policy must pass exact and randomized episodes after reset.
Public tools: api.observe() returns task objects {{name: {{xy_m: [x,y], xyz_m: [x,y,z]}}}}, image path or null. Exact gives precise positions; randomized uses wrist RGB-D plus SAM3 and may miss detections. api.robot.survey() visits safe wrist viewpoints. api.robot.pick_place(source_name, source_xy, target_xy) contact-verifies grasp; positions must derive from recent observation. api.robot.move_pose(x,y,z,quaternion_wxyz) supports x 0.25..0.46, y -0.25..0.25, z 0.14..0.22 m, tool-down quaternion (0,0,1,0). api.robot.gripper(width_m) supports 0..0.08 m. api.task.check_success() provides a public observation-based assessment. api.task.stop() stops the episode.
Observe 'emergency stop button' and 'green storage box'. The grasp approach is 0.20 -> 0.16 -> 0.15 m with open fingers, then close and verify both-finger contact. Transport at z=0.20 m above the tray wall; lower to z=0.18 m, release and settle. These are TOOL heights, not object heights. Do not use old tabletop coordinates (z ~0.9 m). Do not claim success simply because motion returned completed. If uncertain observe again and use a bounded retry or checkpoint.
"""
    return f"""Task: {task.instruction}
Produce a reusable SIM policy bundle. A policy.py is REQUIRED; optional steps/<name>.py files are allowed (up to 16). Return JSON containing files with path and source, no Markdown. No imports or external files. Do not use identifiers beginning with `_`; the policy validator reserves private names. policy.py defines def run(api): and may define def step_<name>(api, state):. Each steps/<name>.py defines def run(api, state):. State is a JSON object returned in a step result as {{"state": {{...}}}}. Work in one step may move/observe repeatedly. A successful policy must return {{"status": "completed"}} and must not call api.task.stop(); call task.stop() only for an unrecoverable abort. The harness first runs policy.py without asking Codex. If it completes, the evaluator scores it without another agent turn. Only a pause, failure, or explicit need for a next step sends execution feedback and fresh observations to a temporary fork of this conversation. In that fork you may rerun a step, call a public tool directly, finish, or abort. The temporary evaluation turns do not become part of the original authoring conversation. Do not use source-line goto.

Public policy and direct-call tools:
- api.observe() -> {{"objects": {{name: {{"xy_m": [x,y], "xyz_m": [x,y,z]}}}}, "image": path or null}}. Exact: accurate task-object positions; randomized: SAM3 RGB-D estimates and missing detections possible. Use freely inside every step.
- api.robot.survey() -> observation from safe wrist viewpoints if objects are missing.
- api.robot.pick_place(source_name, source_xy, target_xy) -> status. Coordinates MUST come from a recent public observation; grasps are contact-verified.
- api.robot.move_pose(x,y,z,quaternion_wxyz) -> status; world metres, workspace x -0.3..0.1, y -0.3..0.3, z 0.89..1.08, downward quaternion (0,0,1,0).
- api.robot.gripper(width_m) -> status; width 0..0.08 m.
- api.task.check_success() -> public visual check (may be uncertain), NOT hidden truth.
- api.task.stop() -> stop safely.
Code cannot import packages, read files, access MuJoCo handles or ground truth. Never invent object coordinates. In direct tool calls, use exact tool name and JSON-array args. The same locked bundle must pass both exact and randomized episodes, with hidden evaluator scoring at the end. Do not claim completion only because a robot action returned completed. Keep retries bounded and use public observations to judge ambiguous grasps; if still uncertain choose a safe retry or abort. First build policy.py as a usable complete attempt, and optional named steps for recovery.
"""


def generate_bundle(prompt: str, session: SimCodexSession, output_dir: Path):
    """Ask Codex once for the locked bundle and return its root plus thread id."""
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "bundle_schema.json"
    schema_path.write_text(json.dumps(_bundle_schema()))
    payload = session.run(prompt, _bundle_schema(), output_dir / "generation")
    (output_dir / "bundle_response.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    bundle_root, bundle = _write_bundle_payload(payload, output_dir)
    return bundle_root, session.id


class CodexPolicyLoopController:
    def __init__(self, prompt: str, session: SimCodexSession, output_dir: Path,
                 generated: bool = False):
        self.prompt, self.session, self.output_dir = prompt, session, output_dir
        self.has_context, self.turn = generated, 0

    @property
    def session_id(self) -> str | None:
        return self.session.id

    @contextmanager
    def episode(self, output: Path):
        child = None
        try:
            def ensure_controller():
                nonlocal child
                if child is None:
                    child = CodexPolicyLoopController(
                        self.prompt, self.session.fork(
                            bootstrap_prompt=self.prompt),
                        output / "codex",
                        generated=self.has_context,
                    )
                return child

            class EpisodeDecision:
                def __call__(self, request):
                    return ensure_controller()(request)

                def checkpoint(self, request):
                    return ensure_controller().checkpoint(request)

            yield EpisodeDecision()
        finally:
            if child is not None:
                child.session.discard()

    def checkpoint(self, request: dict) -> str:
        output = self.output_dir / "checkpoints" / f"{self.turn + 1:03d}"
        self.turn += 1
        output.mkdir(parents=True, exist_ok=True)
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"choice": {"type": "string", "enum": request["options"]},
                                 "reason": {"type": "string"}},
                  "required": ["choice", "reason"]}
        (output / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2))
        message = (
            "Policy execution paused at api.agent.ask. Here are the running command, "
            "its public intermediate results and a fresh observation. Select one of "
            "the available branches; the policy resumes at this checkpoint. Do not "
            "inspect simulator truth or change the locked policy.\n\n"
            + (f"Original task:\n{self.prompt}\n\n" if not self.has_context else "")
            + "Running command: "
            + json.dumps(request.get("running_command"), ensure_ascii=False)
            + "\nCurrent public state:\n"
            + json.dumps(request, ensure_ascii=False)
        )
        image = request.get("observation", {}).get("image")
        images = (Path(image),) if isinstance(image, str) and Path(image).is_file() else ()
        answer = self.session.run(message, schema, output, images)
        self.has_context = True
        if answer["choice"] not in request["options"]:
            raise ValueError("Codex selected an unknown checkpoint branch")
        return answer["choice"]

    def refine_bundle(self, failure_report: dict, previous_bundle, output_dir: Path):
        """Ask the same Codex conversation for a complete replacement bundle."""
        output_dir.mkdir(parents=True, exist_ok=True)
        schema_path = output_dir / "schema.json"
        schema_path.write_text(json.dumps(_bundle_schema()))
        (output_dir / "request.json").write_text(json.dumps({
            "previous_bundle": previous_bundle.manifest(),
            "failure_report": failure_report,
        }, ensure_ascii=False, indent=2))
        message = (
            "SIM policy refinement turn. The previous policy bundle failed a fresh, "
            "reset episode. Use only the public failure report below; hidden simulator "
            "truth is not available. Return a COMPLETE replacement JSON bundle with "
            "policy.py and optional steps/*.py files, not a patch and not Markdown. "
            "Keep the public API contract from the original task. Fix the observed "
            "failure conservatively, keep retries bounded, and do not add imports, "
            "private identifiers beginning with '_', file access, simulator handles, "
            "or ground-truth shortcuts. The new bundle "
            "will be validated, hashed, and tested after another external reset.\n\n"
            + ("Original task contract:\n" + self.prompt + "\n\n"
               if not self.has_context else "")
            + "Previous bundle manifest:\n" + json.dumps(previous_bundle.manifest(), ensure_ascii=False) +
            "\n\nPublic failure report:\n" + json.dumps(failure_report, ensure_ascii=False)
        )
        images = []
        for call in failure_report.get("calls", []):
            if call.get("tool") != "observe":
                continue
            observation = call.get("result")
            image = observation.get("image") if isinstance(observation, dict) else None
            if isinstance(image, str) and image not in images and Path(image).is_file():
                images.append(Path(image))
        payload = self.session.run(message, _bundle_schema(), output_dir, tuple(images[-2:]))
        self.has_context = True
        bundle_root, bundle = _write_bundle_payload(payload, output_dir)
        (output_dir / "manifest.json").write_text(
            json.dumps(bundle.manifest(), ensure_ascii=False, indent=2))
        return bundle_root, bundle

    def __call__(self, request: dict) -> dict:
        self.turn += 1
        output = self.output_dir / "turns" / f"{self.turn:03d}"
        output.mkdir(parents=True, exist_ok=True)
        names = [*request["available_policies"], *request["available_tools"], ""]
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"action": {"type": "string", "enum": [
                      "run_policy", "call_tool", "finish", "abort"]},
                                 "name": {"type": "string", "enum": names},
                                 "args_json": {"type": "string"},
                                 "reason": {"type": "string"}},
                  "required": ["action", "name", "args_json", "reason"]}
        (output / "schema.json").write_text(json.dumps(schema))
        (output / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2))
        message = (
            "SIM Policy Execution Loop turn: choose one command using the schema. "
            "run_policy name must be one "
            "of available_policies; call_tool name must be one of available_tools. "
            "args_json is a JSON array for call_tool, otherwise '[]'. "
            "finish only when publicly supported, abort if unsafe. Every action result, "
            "including normal completion, returns to this SAME conversation. This is an "
            "execution-grounded policy decision, not a request to rewrite the policy bundle. "
            "Never inspect files or hidden simulator truth. Do not generate code now.\n\n"
            + (f"Original task:\n{self.prompt}\n\n" if not self.has_context else "")
            + "Current public episode report:\n" + json.dumps(request, ensure_ascii=False)
        )
        image = request.get("observation", {}).get("image")
        images = (Path(image),) if isinstance(image, str) and Path(image).is_file() else ()
        answer = self.session.run(message, schema, output, images)
        self.has_context = True
        action, name = answer["action"], answer["name"]
        allowed = request["available_policies"] if action == "run_policy" else request["available_tools"]
        if action not in ("finish", "abort") and name not in allowed:
            raise ValueError("Codex selected an unavailable policy or tool")
        args = json.loads(answer["args_json"])
        if not isinstance(args, list):
            raise ValueError("Codex tool arguments must be an array")
        decision = {"action": action, "name": name, "args": args,
                    "reason": answer["reason"]}
        (output / "decision.json").write_text(json.dumps(decision, indent=2))
        return decision
