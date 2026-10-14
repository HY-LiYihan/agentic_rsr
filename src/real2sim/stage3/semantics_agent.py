"""Codex classification of Stage 2 meshes before MuJoCo scene compilation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PROMPT_FILE = Path(__file__).with_name("semantics_prompt.md")
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL = "gpt-6-astra"
EFFORT = "high"
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_name": {"type": "string"},
                    "role": {"type": "string", "enum": ["visual_only", "static", "dynamic", "operable", "compound"]},
                    "joint_type": {"type": "string", "enum": ["fixed", "free", "hinge", "slide"]},
                    "collision": {"type": "string", "enum": ["none", "box", "mesh", "open_container"]},
                    "attached_to": {"type": ["string", "null"]},
                    "reason": {"type": "string"},
                },
                "required": ["source_name", "role", "joint_type", "collision", "attached_to", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["objects"],
    "additionalProperties": False,
}


def validate_classification(result: dict[str, Any], objects: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    names = [item["name"] for item in objects]
    records = result.get("objects")
    if not isinstance(records, list) or len(records) != len(names):
        raise ValueError("semantics must classify every task mesh exactly once")
    mapping = {}
    for record in records:
        if not {"source_name", "role", "joint_type", "collision", "attached_to", "reason"} <= record.keys():
            raise ValueError("each classified mesh requires a complete physical decision")
        name = record["source_name"]
        if name not in names or name in mapping:
            raise ValueError(f"unexpected or duplicate classified mesh: {name}")
        role, joint, collision = record["role"], record["joint_type"], record["collision"]
        allowed = {
            "visual_only": ("fixed", {"none"}),
            "static": ("fixed", {"box", "mesh", "open_container"}),
            "dynamic": ("free", {"box", "open_container"}),
            "operable": ("hinge", {"box"}),
            "compound": ("fixed", {"box"}),
        }
        if role not in allowed or (joint != allowed[role][0] and
                                   not (role == "operable" and joint == "slide")) or collision not in allowed[role][1]:
            raise ValueError(f"inconsistent physical classification: {name}")
        if record.get("attached_to") is not None and role not in {"visual_only", "compound"}:
            raise ValueError(f"world-level physical mesh must not attach to another mesh: {name}")
        if role == "compound" and record.get("attached_to") is None:
            raise ValueError(f"compound collision needs a moving owner: {name}")
        mapping[name] = record
    for name, record in mapping.items():
        parent = record.get("attached_to")
        if parent is not None and (parent not in mapping or
                                   mapping[parent]["role"] not in {"dynamic", "operable"}):
            raise ValueError(f"visual attachment must refer to a moving physical mesh: {name}")
    return mapping


def object_inventory(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory = []
    for item in objects:
        points = np.asarray(item["vertices"], dtype=float)
        bounds = (points.max(axis=0) - points.min(axis=0)).tolist()
        inventory.append({
            "source_name": item["name"], "parent": item.get("parent"),
            "size_m": [round(value, 4) for value in bounds],
            "position_m": [round(value, 4) for value in item["location"]],
            "explicit_properties": {key: value for key, value in item.get("custom_properties", {}).items()
                                    if key.startswith("real2sim_")},
        })
    return inventory


def classify_objects(objects: list[dict[str, Any]], task: str, plan: str,
                     rgb_paths: list[Path], output: Path, source_hash: str) -> dict[str, dict[str, str]]:
    from openai_codex import ApprovalMode, Codex, LocalImageInput, Sandbox, TextInput

    output.mkdir(parents=True, exist_ok=True)
    inventory = object_inventory(objects)
    inventory_path = output / "semantics_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    prompt = PROMPT_FILE.read_text(encoding="utf-8").format(
        task_description=task, plan=plan, inventory=json.dumps(inventory, ensure_ascii=False))
    (output / "semantics_prompt.md").write_text(prompt, encoding="utf-8")
    record: dict[str, Any] = {"model": MODEL, "effort": EFFORT, "thread_id": None,
                              "source_sha256": source_hash, "reference_images": [str(path) for path in rgb_paths]}
    try:
        with Codex() as codex:
            thread = codex.thread_start(cwd=str(REPO_ROOT), sandbox=Sandbox.read_only,
                                        approval_mode=ApprovalMode.deny_all, model=MODEL, ephemeral=True)
            record["thread_id"] = thread.id
            try:
                inputs = [TextInput(text=prompt)] + [LocalImageInput(path=str(path)) for path in rgb_paths]
                for attempt in range(3):
                    turn = thread.run(inputs, model=MODEL, effort=EFFORT,
                                      output_schema=CLASSIFICATION_SCHEMA)
                    record["turn_id"] = turn.id
                    record["turn_status"] = str(turn.status)
                    try:
                        result = json.loads(turn.final_response or "{}")
                        mapping = validate_classification(result, objects)
                        break
                    except (ValueError, KeyError, TypeError) as exc:
                        record.setdefault("validation_errors", []).append(str(exc))
                        if attempt == 2:
                            raise
                        inputs = [TextInput(text=(
                            "Your previous classification failed validation: " + str(exc) +
                            ". Correct your full classification of ALL meshes. "
                            "For every operable part choose hinge/slide AND box collision; "
                            "static needs fixed and box/mesh, dynamic free and box, "
                            "visual_only fixed and none. Invalid previous rows: " +
                            json.dumps(result.get("objects", []), ensure_ascii=False)[-6000:]))]
                record["result"] = result
                if getattr(turn, "usage", None) is not None:
                    usage = turn.usage
                    record["usage"] = usage.model_dump(mode="json") if hasattr(usage, "model_dump") else str(usage)
                record["status"] = "classified"
                (output / "semantics.json").write_text(json.dumps({
                    "source_sha256": source_hash, "objects": result["objects"]
                }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return mapping
            finally:
                try:
                    codex.thread_archive(thread.id)
                    record["thread_cleanup"] = "archived"
                except Exception as exc:
                    record["thread_cleanup"] = "ephemeral_thread_not_persisted"
                    record["thread_archive_error"] = repr(exc)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "semantics_session.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
