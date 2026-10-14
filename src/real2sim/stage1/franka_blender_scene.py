"""Blender-side builder for a portable four-view, fixed-pose robot scene."""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


CONVERSION = Matrix(((1, 0, 0, 0), (0, -1, 0, 0),
                     (0, 0, -1, 0), (0, 0, 0, 1)))


def build(spec_path: Path, output: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.name = "Reference_00"
    scene.render.resolution_x = spec["width"]
    scene.render.resolution_y = spec["height"]
    scene.render.resolution_percentage = 100

    robot_name = spec["robot_object"]
    arm = bpy.data.objects.new(robot_name, None)
    scene.collection.objects.link(arm)
    arm.matrix_world = Matrix(spec["world_from_base"])
    for index, part in enumerate(spec["meshes"]):
        before = set(scene.objects)
        bpy.ops.wm.obj_import(filepath=part["obj"], forward_axis="Y", up_axis="Z")
        material = bpy.data.materials.new(f"{robot_name}_color_{index:02d}")
        material.diffuse_color = part["rgba"]
        material.use_nodes = True
        nodes = material.node_tree.nodes
        shader = nodes.get("Principled BSDF") or nodes.new("ShaderNodeBsdfPrincipled")
        shader.inputs["Base Color"].default_value = part["rgba"]
        output_node = nodes.get("Material Output") or nodes.new("ShaderNodeOutputMaterial")
        material.node_tree.links.new(shader.outputs["BSDF"], output_node.inputs["Surface"])
        for mesh in set(scene.objects) - before:
            if mesh.type == "MESH":
                mesh.data.materials.clear()
                mesh.data.materials.append(material)
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"{robot_name} visual mesh import failed")
    if any(not mesh.data.materials for mesh in meshes):
        raise RuntimeError(f"{robot_name} visual mesh has no material")
    for mesh in meshes:
        if any(abs(mesh.matrix_world[row][col] - Matrix.Identity(4)[row][col]) > 1e-5
               for row in range(4) for col in range(4)):
            raise RuntimeError(f"{robot_name} mesh import changed URDF coordinates")
        mesh.parent = arm
        mesh.matrix_parent_inverse = Matrix.Identity(4)

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.color_type = "MATERIAL"

    cameras = []
    for index, view in enumerate(spec["views"]):
        intrinsic = view["intrinsic"]
        extrinsic = Matrix(tuple(tuple(row) for row in view["extrinsic"]) +
                           ((0, 0, 0, 1),))
        data = bpy.data.cameras.new(f"Reference_{index:02d}")
        camera = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(camera)
        camera.matrix_world = extrinsic.inverted() @ CONVERSION
        data.type = "PERSP"
        data.lens = intrinsic[0][0] * data.sensor_width / spec["width"]
        data.shift_x = (spec["width"] / 2 - intrinsic[0][2]) / spec["width"]
        data.shift_y = (intrinsic[1][2] - spec["height"] / 2) / spec["width"]
        ratio = intrinsic[0][0] / intrinsic[1][1]
        view_scene = scene if index == 0 else bpy.data.scenes.new(data.name)
        view_scene.render.resolution_x = spec["width"]
        view_scene.render.resolution_y = spec["height"]
        view_scene.render.resolution_percentage = 100
        view_scene.render.pixel_aspect_x = max(1.0, 1.0 / ratio)
        view_scene.render.pixel_aspect_y = max(1.0, ratio)
        if index != 0:
            for item in scene.objects:
                view_scene.collection.objects.link(item)
        view_scene.camera = camera
        image = bpy.data.images.load(view["image"], check_existing=False)
        image.pack()
        data.show_background_images = True
        background = data.background_images.new()
        background.image = image
        background.display_depth = "BACK"
        cameras.append(camera)

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    bpy.ops.wm.open_mainfile(filepath=str(output))
    arm = bpy.data.objects.get(robot_name)
    saved_meshes = [obj for obj in bpy.data.objects if obj.type == "MESH" and obj.parent == arm]
    if arm is None or len(saved_meshes) != len(spec["meshes"]):
        raise RuntimeError(f"saved scene is missing {robot_name} geometry")
    expected_colors = {tuple(part["rgba"]) for part in spec["meshes"]}
    actual_colors = {tuple(mesh.data.materials[0].diffuse_color) for mesh in saved_meshes
                     if len(mesh.data.materials) == 1}
    if len(actual_colors) != len(saved_meshes) or any(
            not any(all(abs(a - b) < 1e-5 for a, b in zip(actual, expected))
                    for actual in actual_colors) for expected in expected_colors):
        raise RuntimeError(f"saved {robot_name} materials differ from robot_control visuals")
    expected_arm = Matrix(spec["world_from_base"])
    if any(abs(arm.matrix_world[row][col] - expected_arm[row][col]) > 1e-5
           for row in range(4) for col in range(4)):
        raise RuntimeError(f"saved {robot_name} base pose differs from registration")
    if len([obj for obj in bpy.data.objects if obj.type == "CAMERA"]) != 4:
        raise RuntimeError("saved scene must contain exactly four reference cameras")
    for index, view in enumerate(spec["views"]):
        camera = bpy.data.objects[f"Reference_{index:02d}"]
        view_scene = bpy.data.scenes[camera.name]
        expected = Matrix(tuple(tuple(row) for row in view["extrinsic"]) +
                          ((0, 0, 0, 1),)).inverted() @ CONVERSION
        if any(abs(camera.matrix_world[row][col] - expected[row][col]) > 1e-5
               for row in range(4) for col in range(4)):
            raise RuntimeError(f"saved camera {index} differs from metric calibration")
        if view_scene.camera != camera or not camera.data.background_images[0].image.packed_file:
            raise RuntimeError(f"saved reference {index} is not portable")
        intrinsic = view["intrinsic"]
        ratio = intrinsic[0][0] / intrinsic[1][1]
        if abs(view_scene.render.pixel_aspect_y / view_scene.render.pixel_aspect_x - ratio) > 1e-5:
            raise RuntimeError(f"saved camera {index} has incorrect focal aspect")


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:]
    build(Path(arguments[0]), Path(arguments[1]))
