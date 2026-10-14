"""A locked policy.py and optional reusable, import-free step files."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from .program_runner import run_program, validate_program
from .policy_terms import ADAPTATION_MODE, LOOP_NAME, POLICY_TYPE


def _step_names(source: str) -> tuple[str, ...]:
    tree = validate_program(source)
    return tuple(node.name for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name.startswith("step_")
                 and len(node.args.args) == 2 and not node.args.defaults)


class PolicyBundle:
    def __init__(self, root: Path):
        root = Path(root)
        if not root.is_dir() or root.is_symlink() or not (root / "policy.py").is_file():
            raise ValueError("Policy bundle requires a directory containing policy.py")
        steps_dir = root / "steps"
        if steps_dir.exists() and steps_dir.is_symlink():
            raise ValueError("Policy bundle steps directory cannot be a symlink")
        files = [root / "policy.py", *sorted(steps_dir.glob("*.py"))]
        if len(files) > 17 or any(file.is_symlink() or file.stat().st_size > 64_000 for file in files):
            raise ValueError("Policy bundle exceeds file limits or contains a symlink")
        self.root = root
        self.sources = {}
        self.files = [file.relative_to(root).as_posix() for file in files]
        for file in files:
            source = file.read_text()
            validate_program(source)
            if file.name == "policy.py":
                function = next(node for node in ast.parse(source).body
                                if isinstance(node, ast.FunctionDef) and node.name == "run")
                if len(function.args.args) != 1 or function.args.defaults:
                    raise ValueError("policy.py must define run(api)")
                self.sources["run"] = (source, "run", None)
                for step in _step_names(source):
                    self.sources[step] = (source, step, {})
            else:
                if file.parent != steps_dir:
                    raise ValueError("Only policy.py and steps/*.py are allowed")
                if not file.stem.isidentifier() or file.stem.startswith("_"):
                    raise ValueError("Invalid step filename")
                name = f"step_{file.stem}"
                if name in self.sources:
                    raise ValueError(f"Duplicate step: {name}")
                function = next(node for node in ast.parse(source).body
                                if isinstance(node, ast.FunctionDef) and node.name == "run")
                if len(function.args.args) != 2 or function.args.defaults:
                    raise ValueError(f"{file.name} must define run(api, state)")
                self.sources[name] = (source, "run", {})
        self.hash = hashlib.sha256(json.dumps(
            [(str(file.relative_to(root)), file.read_text()) for file in files],
            ensure_ascii=False).encode()).hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "policy_type": POLICY_TYPE,
            "loop": LOOP_NAME,
            "adaptation_mode": ADAPTATION_MODE,
            "required_entrypoint": "policy.py:run(api)",
            "policy_names": list(self.sources),
            "files": self.files,
            "file_count": len(self.files),
            "sha256": self.hash,
        }

    def execute(self, name: str, state: dict, dispatch, on_pause=None) -> dict:
        if name not in self.sources or not isinstance(state, dict):
            raise ValueError("Unknown policy step or invalid state")
        source, entry, default_state = self.sources[name]
        return run_program(
            source, dispatch, timeout=240, on_pause=on_pause,
            entry=entry, state=state if default_state is not None else None)
