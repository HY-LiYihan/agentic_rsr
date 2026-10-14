"""Build an unregistered robot scene with deterministic generic cameras in Blender."""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def main():
    asset, output, robot = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.open_mainfile(filepath=asset)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    transforms = {obj.name: obj.matrix_world.copy() for obj in meshes}
    for obj in list(bpy.data.objects):
        if obj not in meshes:
            bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    for other in list(bpy.data.scenes):
        if other != scene:
            bpy.data.scenes.remove(other)
    root = bpy.data.objects.new("Piper" if robot == "piper" else "FR3", None)
    scene.collection.objects.link(root)
    for obj in meshes:
        obj.parent = root
        obj.matrix_world = transforms[obj.name]
    scene.name = "Reference_00"
    scene.unit_settings.system = "METRIC"
    for index, position in enumerate(((1.2, -1.2, 1.1), (-1.2, -1.2, 1.1), (-1.2, 1.2, 1.1), (1.2, 1.2, 1.1))):
        data = bpy.data.cameras.new(f"Reference_{index:02d}")
        camera = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(camera)
        camera.location = position
        camera.rotation_euler = (Vector((0, 0, 0.3)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        data.lens = 35
        data.clip_start = 0.01
        data.clip_end = 100
    for index in range(4):
        view = scene if index == 0 else bpy.data.scenes.new(f"Reference_{index:02d}")
        if index:
            for obj in scene.objects:
                view.collection.objects.link(obj)
        view.camera = bpy.data.objects[f"Reference_{index:02d}"]
        view.render.resolution_x = 512
        view.render.resolution_y = 288
        view.render.resolution_percentage = 100
    for image in list(bpy.data.images):
        if image.source == "FILE" and image.users == 0:
            bpy.data.images.remove(image)
    bpy.ops.file.pack_all()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    bpy.ops.wm.open_mainfile(filepath=str(destination))
    assert len([obj for obj in bpy.data.objects if obj.type == "CAMERA"]) == 4
    assert len([obj for obj in bpy.data.objects if obj.type == "MESH"]) == len(meshes)
    destination.with_suffix(".json").write_text(json.dumps({"asset": asset, "robot": robot, "calibrated": False, "camera_initialization": "generic orbit, no Stage1 transforms", "mesh_count": len(meshes)}, indent=2))


if __name__ == "__main__":
    main()
