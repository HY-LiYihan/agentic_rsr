"""Blender worker: replace only the Piper stand mesh orientation in an existing scene."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def main() -> None:
    args = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))
    arm = bpy.data.objects.get("Piper")
    stand = bpy.data.objects.get("_г_--__3_001-mesh")
    if arm is None or stand is None or stand.type != "MESH" or stand.parent != arm:
        raise ValueError("scene lacks the expected independent Piper camera stand mesh")
    if len(stand.data.vertices) != args["vertex_count"] or len(stand.data.polygons) != args["face_count"]:
        raise ValueError("Piper camera stand geometry differs from the pinned mesh")
    if stand.data.shape_keys or stand.data.users != 1:
        raise ValueError("cannot safely modify shared or shape-keyed stand geometry")

    relative = arm.matrix_world.inverted() @ stand.matrix_world
    inverse = relative.inverted()
    current = np.array([tuple(relative @ vertex.co) for vertex in stand.data.vertices])
    actual_bounds = np.array([current.min(axis=0), current.max(axis=0)])
    if not np.allclose(actual_bounds, args["before_bounds"], atol=1e-5):
        raise ValueError(f"stand is not in the expected old orientation: {actual_bounds.tolist()}")

    correction = Matrix(args["correction"])
    for vertex in stand.data.vertices:
        vertex.co = inverse @ correction @ relative @ vertex.co
    stand.data.update()
    repaired = np.array([tuple(relative @ vertex.co) for vertex in stand.data.vertices])
    after_bounds = np.array([repaired.min(axis=0), repaired.max(axis=0)])
    if not np.allclose(after_bounds, args["after_bounds"], atol=1e-5):
        raise ValueError(f"stand repair did not match robot_control geometry: {after_bounds.tolist()}")

    target = Path(args["output"])
    bpy.ops.wm.save_as_mainfile(filepath=str(target), check_existing=False)
    if not target.is_file():
        raise FileNotFoundError(target)
    print("REPAIRED_STAND", json.dumps({"output": str(target), "before": actual_bounds.tolist(),
                                         "after": after_bounds.tolist(),
                                         "dimensions": list(stand.dimensions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
