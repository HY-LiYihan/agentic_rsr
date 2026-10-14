"""Executed by Blender in background mode; arguments follow --."""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def create(spec_path: Path, blend_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.resolution_x = spec["width"]
    scene.render.resolution_y = spec["height"]
    scene.render.resolution_percentage = 100
    conversion = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    for index, (intrinsic, extrinsic) in enumerate(zip(spec["intrinsics"], spec["extrinsics"])):
        world_to_camera = Matrix(tuple(tuple(row) for row in extrinsic) + ((0, 0, 0, 1),))
        camera_data = bpy.data.cameras.new(f"Reference_{index:02d}")
        camera = bpy.data.objects.new(camera_data.name, camera_data)
        scene.collection.objects.link(camera)
        camera.matrix_world = world_to_camera.inverted() @ conversion
        camera_data.type = "PERSP"
        camera_data.lens = float(intrinsic[0][0]) * camera_data.sensor_width / spec["width"]
        camera_data.shift_x = (spec["width"] / 2 - intrinsic[0][2]) / spec["width"]
        camera_data.shift_y = (intrinsic[1][2] - spec["height"] / 2) / spec["width"]
        if index == 0:
            scene.camera = camera
            scene.name = "Reference_00"
            ratio = intrinsic[0][0] / intrinsic[1][1]
            scene.render.pixel_aspect_x = max(1.0, 1 / ratio)
            scene.render.pixel_aspect_y = max(1.0, ratio)
    root = bpy.data.objects.new("Piper", None)
    scene.collection.objects.link(root)
    root.matrix_world = Matrix(spec["world_from_base"])
    for mesh in spec["meshes"]:
        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=mesh["obj"])
        imported = list(set(bpy.data.objects) - before)
        if not imported:
            raise RuntimeError(f"Piper mesh was not imported: {mesh['obj']}")
        for part in imported:
            part.name = "Piper_" + mesh["name"]
            part.parent = root
            part.matrix_parent_inverse = Matrix.Identity(4)
            part.matrix_basis = Matrix(mesh["matrix"])
    for index in range(1, 4):
        view_scene = bpy.data.scenes.new(f"Reference_{index:02d}")
        view_scene.render.resolution_x = spec["width"]
        view_scene.render.resolution_y = spec["height"]
        view_scene.render.resolution_percentage = 100
        ratio = spec["intrinsics"][index][0][0] / spec["intrinsics"][index][1][1]
        view_scene.render.pixel_aspect_x = max(1.0, 1 / ratio)
        view_scene.render.pixel_aspect_y = max(1.0, ratio)
        for item in scene.objects:
            view_scene.collection.objects.link(item)
        view_scene.camera = bpy.data.objects[f"Reference_{index:02d}"]
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]
    arm = bpy.data.objects.get("Piper")
    if len(cameras) != 4 or arm is None or not any(obj.type == "MESH" and obj.parent == arm for obj in bpy.data.objects):
        raise RuntimeError("saved scene lacks four cameras or Piper geometry")
    if any(abs(arm.matrix_world[row][column] - spec["world_from_base"][row][column]) > 1e-5
           for row in range(4) for column in range(4)):
        raise RuntimeError("saved scene changed Piper registration")
    for index, extrinsic in enumerate(spec["extrinsics"]):
        view_scene = bpy.data.scenes[f"Reference_{index:02d}"]
        if (view_scene.camera != bpy.data.objects[f"Reference_{index:02d}"] or
                abs(view_scene.render.pixel_aspect_y / view_scene.render.pixel_aspect_x -
                    spec["intrinsics"][index][0][0] / spec["intrinsics"][index][1][1]) > 1e-5):
            raise RuntimeError("saved scene changed camera focal calibration")
        expected = Matrix(tuple(tuple(row) for row in extrinsic) + ((0, 0, 0, 1),)).inverted() @ conversion
        actual = bpy.data.objects[f"Reference_{index:02d}"].matrix_world
        if any(abs(expected[row][column] - actual[row][column]) > 1e-5
               for row in range(4) for column in range(4)):
            raise RuntimeError("saved scene changed camera calibration")
    for index in range(4):
        view_scene = bpy.data.scenes[f"Reference_{index:02d}"]
        view_scene.render.engine = "CYCLES"
        view_scene.cycles.samples = 1
        view_scene.render.resolution_percentage = 10
        target = blend_path.with_suffix(".smoke.png") if index == 0 else blend_path.with_name(f"view_{index:02d}.smoke.png")
        view_scene.render.filepath = str(target)
        bpy.ops.render.render(scene=view_scene.name, write_still=True)


if __name__ == "__main__":
    create(Path(sys.argv[sys.argv.index("--") + 1]), Path(sys.argv[sys.argv.index("--") + 2]))
