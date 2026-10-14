"""Keep the SIM authoring thread in a run-local Codex home."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


class SimCodexSession:
    def __init__(self, model: str, cwd: Path, transcript: Path | None = None,
                 codex_home: Path | None = None, thread_id: str | None = None,
                 effort: str = "high", ephemeral: bool = False):
        if ephemeral and thread_id is not None:
            raise ValueError("Ephemeral authoring threads cannot be resumed")
        self.model = model
        self.cwd = cwd
        self.transcript = transcript
        # Match real2sim/stage2: the SDK's default Codex home carries the
        # authenticated native thread. A custom home is opt-in and must have
        # been authenticated separately before this process starts.
        self.codex_home = codex_home.expanduser().resolve() if codex_home else None
        self.thread_id = thread_id
        self.effort = effort
        self.ephemeral = ephemeral
        self.client = None
        self.thread = None
        self.turns = []

    @property
    def id(self) -> str | None:
        return self.thread.id if self.thread is not None else None

    def __enter__(self):
        from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

        if self.codex_home is None:
            self.client = Codex()
        else:
            self.codex_home.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.codex_home, 0o700)
            self.client = Codex(CodexConfig(env={"CODEX_HOME": str(self.codex_home)}))
        self.client.__enter__()
        try:
            options = dict(cwd=str(self.cwd), sandbox=Sandbox.read_only,
                           approval_mode=ApprovalMode.deny_all, model=self.model)
            if self.thread_id:
                self.thread = self.client.thread_resume(self.thread_id, **options)
            else:
                self.thread = self.client.thread_start(**options, ephemeral=self.ephemeral)
        except BaseException:
            self.client.__exit__(*sys.exc_info())
            self.client = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self.client.__exit__(exc_type, exc_value, traceback)
        finally:
            self.thread = None
            self.client = None

    def fork(self, bootstrap_prompt: str | None = None):
        """Fork the stored authoring history without saving evaluation turns."""
        if self.thread is None:
            raise RuntimeError("Start the parent Codex session before forking an episode")
        from openai_codex import ApprovalMode, Sandbox

        if not self.turns and not self.thread_id:
            if bootstrap_prompt is None:
                raise RuntimeError("A new authoring thread needs a prompt before evaluation")
            self.bootstrap(bootstrap_prompt)
        child = SimCodexSession(self.model, self.cwd, codex_home=self.codex_home,
                                effort=self.effort)
        child.client = self.client
        child.thread = self.client.thread_fork(
            self.id, cwd=str(self.cwd), sandbox=Sandbox.read_only, model=self.model,
            approval_mode=ApprovalMode.deny_all, ephemeral=True,
        )
        return child

    def bootstrap(self, prompt: str):
        """Give a supplied bundle's author thread a stored task context."""
        if self.turns or self.thread_id:
            return
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"acknowledged": {"type": "boolean"}},
                  "required": ["acknowledged"]}
        output = self.transcript.parent if self.transcript else (
            self.codex_home.parent if self.codex_home else self.cwd)
        self.run(prompt, schema, output / "bootstrap")

    def discard(self):
        self.thread = None
        self.client = None

    def run(self, message: str, schema: dict, output: Path,
            images: tuple[Path, ...] = ()) -> dict:
        if self.thread is None:
            raise RuntimeError("Start the SIM Codex SDK session before requesting a turn")
        from openai_codex import LocalImageInput, TextInput

        inputs = [TextInput(text=message)]
        inputs.extend(LocalImageInput(path=str(image)) for image in images)
        turn = self.thread.run(inputs, model=self.model, effort=self.effort, output_schema=schema)
        output.mkdir(parents=True, exist_ok=True)
        (output / "prompt.md").write_text(message)
        (output / "turn.json").write_text(json.dumps({
            "thread_id": self.id, "turn_id": turn.id, "status": str(turn.status),
            "final_response": turn.final_response,
        }, ensure_ascii=False, indent=2))
        if not turn.final_response:
            raise RuntimeError(f"Codex SDK turn returned no response: {turn.status}")
        answer = json.loads(turn.final_response)
        (output / "response.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2))
        record = {
            "turn_id": turn.id, "status": str(turn.status),
            "prompt": message, "images": [str(image) for image in images],
            "response": answer,
            "items": _jsonable(getattr(turn, "items", [])),
            "usage": _jsonable(getattr(turn, "usage", None)),
        }
        self.turns.append(record)
        if self.transcript is not None:
            self.transcript.parent.mkdir(parents=True, exist_ok=True)
            with self.transcript.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return answer
