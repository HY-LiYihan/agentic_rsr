"""Resolve the Blender executable consistently across reconstruction stages."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MACOS_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def resolve_blender(value: str | Path | None = None) -> Path:
    configured = value if value is not None else os.environ.get("BLENDER_BIN") or None
    if configured is not None:
        requested = str(Path(configured).expanduser())
        executable = shutil.which(requested)
        if executable is None:
            raise FileNotFoundError(
                f"Configured Blender executable is missing or not executable: {configured}. "
                "Set --blender or BLENDER_BIN to an executable path or command on PATH.")
        return Path(executable).resolve()

    executable = shutil.which("blender")
    if executable is None and sys.platform == "darwin":
        executable = shutil.which(str(MACOS_BLENDER))
    if executable is None:
        raise FileNotFoundError(
            "Blender executable not found. Install blender on PATH, "
            "or set --blender or BLENDER_BIN to your Blender executable.")
    return Path(executable).resolve()
