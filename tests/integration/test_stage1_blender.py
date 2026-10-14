import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from real2sim.blender import resolve_blender

SCRIPT = Path(__file__).resolve().parents[2] / "src/real2sim/stage1/blender_scene.py"
DELIVERY_SCRIPT = SCRIPT.with_name("franka_blender_scene.py")


@pytest.fixture(scope="module")
def blender():
    try:
        return str(resolve_blender())
    except FileNotFoundError as error:
        if os.environ.get("BLENDER_BIN"):
            pytest.fail(str(error))
        pytest.skip(str(error))


def test_blender_four_cameras_and_fixed_arm(tmp_path, blender):
    mesh = tmp_path / "triangle.obj"
    mesh.write_text("v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nf 1 2 3\n")
    spec = {"width": 512, "height": 288,
            "intrinsics": [[[400 + index, 0, 256], [0, 410, 144], [0, 0, 1]] for index in range(4)],
            "extrinsics": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]] * 4,
            "world_from_base": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -2], [0, 0, 0, 1]],
            "meshes": [{"name": "base_link_0", "obj": str(mesh),
                        "matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}]}
    path = tmp_path / "scene_spec.json"
    path.write_text(json.dumps(spec))
    output = tmp_path / "initial.blend"
    result = subprocess.run([blender, "-b", "-noaudio", "--python", str(SCRIPT),
                             "--", str(path), str(output)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.exists() and all(path.exists() for path in
                                   [tmp_path / "initial.smoke.png"] +
                                   [tmp_path / f"view_{index:02d}.smoke.png" for index in range(1, 4)])


def test_delivered_scene_uses_robot_base_as_world_origin(tmp_path, blender):
    mesh = tmp_path / "triangle.obj"
    mesh.write_text("v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nf 1 2 3\n")
    image = tmp_path / "reference.jpg"
    assert cv2.imwrite(str(image), np.zeros((288, 512, 3), dtype=np.uint8))
    extrinsic = [[1, 0, 0, 0.2], [0, 1, 0, -0.3], [0, 0, 1, 1.0]]
    spec = {"width": 512, "height": 288, "meshes": [
                {"obj": str(mesh), "rgba": [0.2, 0.3, 0.4, 1.0]}], "robot_object": "FR3",
            "world_from_base": np.eye(4).tolist(),
            "views": [{"frame": frame, "intrinsic": [[400, 0, 256], [0, 400, 144], [0, 0, 1]],
                       "extrinsic": extrinsic, "image": str(image)}
                      for frame in (0, 10, 21, 31)]}
    path = tmp_path / "scene_spec.json"
    path.write_text(json.dumps(spec))
    output = tmp_path / "initial.blend"
    result = subprocess.run([blender, "-b", "-noaudio", "--python", str(DELIVERY_SCRIPT),
                             "--", str(path), str(output)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.is_file()
    inspector = tmp_path / "inspect.py"
    inspector.write_text("import bpy\nfrom mathutils import Matrix\n"
                         "arm = bpy.data.objects['FR3']\n"
                         "assert arm.matrix_world == Matrix.Identity(4)\n"
                         "meshes = [obj for obj in bpy.data.objects if obj.parent == arm]\n"
                         "assert len(meshes) == 1\n"
                         "material = meshes[0].data.materials[0]\n"
                         "assert all(abs(a-b) < 1e-5 for a,b in zip(material.diffuse_color, (.2,.3,.4,1)))\n"
                         "assert material.use_nodes\n")
    checked = subprocess.run([blender, "-b", str(output), "--python", str(inspector)],
                             capture_output=True, text=True, timeout=120)
    assert checked.returncode == 0, checked.stdout + checked.stderr
