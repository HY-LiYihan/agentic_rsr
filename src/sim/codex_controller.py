"""Continue one Codex conversation at explicit SIM policy decision points."""

from __future__ import annotations

import json
from pathlib import Path

from .codex_sdk import SimCodexSession


class CodexDecisionController:
    def __init__(self, prompt: str, session: SimCodexSession, output_dir: Path,
                 has_context: bool = False):
        self.prompt, self.session, self.output_dir = prompt, session, output_dir
        self.has_context = has_context
        self.decisions = 0

    def __call__(self, request: dict) -> str:
        self.decisions += 1
        output = self.output_dir / "decisions" / f"{self.decisions:03d}"
        output.mkdir(parents=True, exist_ok=True)
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"choice": {"type": "string", "enum": request["options"]},
                                 "reason": {"type": "string"}},
                  "required": ["choice", "reason"]}
        (output / "schema.json").write_text(json.dumps(schema))
        (output / "request.json").write_text(json.dumps(request, indent=2))
        message = (
            "The SIM Python program is PAUSED at api.agent.ask(). You are judging a "
            "single public-observation decision, NOT writing new Python. The robot and "
            "policy process are frozen. Reply in the required JSON schema with exactly "
            "one of the offered choice strings. Never infer hidden simulator truth from "
            "files or tools. If evidence is insufficient, prefer a safe retry or abort. "
            "The selected choice is returned to the SAME running program at the ask() "
            "call; you cannot jump to an arbitrary source line.\n\n"
            + (f"Original task and tool contract:\n{self.prompt}\n\n"
               if not self.has_context else "")
            + "Current public state:\n" + json.dumps(request, ensure_ascii=False)
        )
        image = request["observation"].get("image")
        images = (Path(image),) if isinstance(image, str) and Path(image).is_file() else ()
        response = self.session.run(message, schema, output, images)
        self.has_context = True
        choice = response.get("choice")
        if choice not in request["options"]:
            raise ValueError("Codex selected an unknown branch")
        (output / "decision.json").write_text(json.dumps({"choice": choice,
                                                         "reason": response.get("reason", ""),
                                                         "session_id": self.session.id}, indent=2))
        return choice
