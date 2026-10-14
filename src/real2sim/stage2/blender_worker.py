"""Blender-side worker for Stage 2 inspect, render, freeze and apply commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
import OpenImageIO as oiio


DEFAULT_LIGHTING_COLLECTION = "Stage2_Lighting"
DEFAULT_LIGHT_PREFIX = "Stage2_DefaultLight_"


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matrix(matrix: Any) -> list[list[float]]:
    return [[round(float(value), 10) for value in row] for row in matrix]


def _object_record(obj: bpy.types.Object) -> dict[str, Any]:
    return {
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "location": [float(value) for value in obj.location],
        "rotation_mode": obj.rotation_mode,
        "rotation": [float(value) for value in obj.rotation_euler] if obj.rotation_mode == "XYZ" else [float(value) for value in obj.rotation_quaternion],
        "scale": [float(value) for value in obj.scale],
        "dimensions": [float(value) for value in obj.dimensions],
        "data": obj.data.name if obj.data else None,
    }


def _ensure_world(scene: bpy.types.Scene) -> None:
    world = scene.world or bpy.data.worlds.new(f"Stage2World_{scene.name}")
    scene.world = world
    world.color = (0.055, 0.055, 0.055)
    if world.use_nodes:
        background = world.node_tree.nodes.get("Background")
        if background is not None:
            background.inputs["Color"].default_value = (0.055, 0.055, 0.055, 1.0)
            background.inputs["Strength"].default_value = 0.35


def _restore_robot_material_colors() -> None:
    for obj in bpy.data.objects:
        if not _is_robot_object(obj) or obj.type != "MESH":
            continue
        for material in obj.data.materials:
            if material is None or not material.use_nodes:
                continue
            for output in material.node_tree.nodes:
                if output.type != "OUTPUT_MATERIAL" or not output.is_active_output:
                    continue
                surface = output.inputs.get("Surface")
                if surface is None or not surface.is_linked:
                    continue
                shader = surface.links[0].from_node
                if shader.type != "BSDF_PRINCIPLED":
                    continue
                base_color = shader.inputs.get("Base Color")
                if base_color is not None and not base_color.is_linked:
                    base_color.default_value = material.diffuse_color


def _lighting_collection() -> bpy.types.Collection:
    collection = bpy.data.collections.get(DEFAULT_LIGHTING_COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(DEFAULT_LIGHTING_COLLECTION)
    for scene in bpy.data.scenes:
        if collection.name not in {item.name for item in scene.collection.children}:
            scene.collection.children.link(collection)
    return collection


def _ensure_default_lighting() -> dict[str, Any]:
    _restore_robot_material_colors()
    collection = _lighting_collection()
    for scene in bpy.data.scenes:
        _ensure_world(scene)
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "Medium High Contrast"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    if not any(obj.name.startswith(DEFAULT_LIGHT_PREFIX) for obj in collection.objects):
        for suffix, rotation, energy in (
            ("Key", (math.radians(25), math.radians(-35), math.radians(-25)), 2.2),
            ("Fill", (math.radians(-35), math.radians(25), math.radians(155)), 0.9),
        ):
            light_data = bpy.data.lights.new(f"{DEFAULT_LIGHT_PREFIX}Data_{suffix}", type="SUN")
            light_data.energy = energy
            light_data.color = (1.0, 1.0, 1.0)
            light_object = bpy.data.objects.new(f"{DEFAULT_LIGHT_PREFIX}{suffix}", light_data)
            light_object.rotation_euler = rotation
            light_object["stage2_role"] = suffix.lower()
            light_object["stage2_replaceable"] = True
            collection.objects.link(light_object)
    return {
        "collection": collection.name,
        "lights": sorted(obj.name for obj in collection.objects if obj.type == "LIGHT"),
        "scenes": sorted(scene.name for scene in bpy.data.scenes),
    }


def add_default_lighting(output: Path) -> dict[str, Any]:
    lighting = _ensure_default_lighting()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    return {"output": str(output), "lighting": lighting, "passed": True}


def _add_neutral_lighting(scene: bpy.types.Scene) -> None:
    """Add transient neutral lights so unlit Stage 1 meshes render as gray."""
    _restore_robot_material_colors()
    if any(obj.name == DEFAULT_LIGHTING_COLLECTION for obj in scene.collection.children):
        _ensure_world(scene)
        return
    if any(obj.name.startswith("Stage2GrayLight_") for obj in scene.objects):
        return
    _ensure_world(scene)
    for suffix, rotation, energy in (
        ("Key", (math.radians(25), math.radians(-35), math.radians(-25)), 2.2),
        ("Fill", (math.radians(-35), math.radians(25), math.radians(155)), 0.9),
    ):
        light_data = bpy.data.lights.new(f"Stage2GrayLightData_{scene.name}_{suffix}", type="SUN")
        light_data.energy = energy
        light_object = bpy.data.objects.new(f"Stage2GrayLight_{scene.name}_{suffix}", light_data)
        light_object.rotation_euler = rotation
        scene.collection.objects.link(light_object)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def _is_robot_object(obj: bpy.types.Object) -> bool:
    current = obj
    while current is not None:
        name = current.name.lower()
        if current.name in {"Piper", "FR3"} or any(token in name for token in ("piper", "franka", "robot", "gripper")):
            return True
        current = current.parent
    return False


def frozen_signature() -> dict[str, Any]:
    cameras = []
    for obj in sorted((item for item in bpy.data.objects if item.type == "CAMERA"), key=lambda item: item.name):
        data = obj.data
        cameras.append({
            "name": obj.name,
            "matrix_world": _matrix(obj.matrix_world),
            "lens": float(data.lens),
            "sensor_width": float(data.sensor_width),
            "sensor_height": float(data.sensor_height),
            "shift_x": float(data.shift_x),
            "shift_y": float(data.shift_y),
            "clip_start": float(data.clip_start),
            "clip_end": float(data.clip_end),
        })
    robot = [_object_record(obj) for obj in sorted(bpy.data.objects, key=lambda item: item.name)
             if _is_robot_object(obj)]
    if os.environ.get("REAL2SIM_UNCALIBRATED") == "1":
        for camera in cameras:
            for key in ("matrix_world", "lens", "shift_x", "shift_y"):
                camera.pop(key)
    return {
        "cameras": cameras,
        "robot": robot,
        "camera_hash": _json_hash(cameras),
        "robot_hash": _json_hash(robot),
    }


def inspect_scene() -> dict[str, Any]:
    scenes = []
    for scene in sorted(bpy.data.scenes, key=lambda item: item.name):
        scenes.append({
            "name": scene.name,
            "camera": scene.camera.name if scene.camera else None,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "engine": scene.render.engine,
        })
    objects = [_object_record(obj) for obj in sorted(bpy.data.objects, key=lambda item: item.name)]
    return {"scenes": scenes, "objects": objects, "frozen_signature": frozen_signature()}


def _save_rgb_and_depth(scene: bpy.types.Scene, output: Path, ordinal: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob(f"stage2_depth_{ordinal}*.exr"):
        stale.unlink(missing_ok=True)
    available_engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    scene.render.engine = ("BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines
                           else "BLENDER_EEVEE")
    _add_neutral_lighting(scene)
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.use_nodes = True
    scene.view_layers[0].use_pass_z = True
    if scene.compositing_node_group is None:
        scene.compositing_node_group = bpy.data.node_groups.new(
            "Stage2Compositor", "CompositorNodeTree")
    nodes = scene.compositing_node_group.nodes
    links = scene.compositing_node_group.links
    nodes.clear()
    render_layers = nodes.new("CompositorNodeRLayers")
    depth_output = nodes.new("CompositorNodeOutputFile")
    depth_output.format.file_format = "OPEN_EXR_MULTILAYER"
    depth_output.format.color_depth = "32"
    if hasattr(depth_output, "file_output_items"):
        depth_output.directory = str(output)
        depth_output.file_output_items.new("FLOAT", "Depth")
        depth_output.file_name = f"stage2_depth_{ordinal}"
        links.new(render_layers.outputs["Depth"], depth_output.inputs["Depth"])
    else:
        depth_output.base_path = str(output)
        depth_output.file_slots[0].path = f"camera_{ordinal}_depth"
        links.new(render_layers.outputs["Depth"], depth_output.inputs[0])
    scene.render.filepath = str(output / f"camera_{ordinal}_rgb.png")
    bpy.ops.render.render(write_still=True)
    exr_candidates = sorted(output.glob(f"stage2_depth_{ordinal}*.exr"),
                            key=lambda path: path.stat().st_mtime)
    if not exr_candidates:
        raise RuntimeError("Blender compositor did not produce an OpenEXR Depth pass")
    image_input = oiio.ImageInput.open(str(exr_candidates[-1]))
    if image_input is None:
        raise RuntimeError(f"could not open depth EXR: {exr_candidates[-1]}")
    try:
        specification = image_input.spec()
        pixels = np.asarray(image_input.read_image(format=oiio.FLOAT), dtype=np.float32)
        if pixels.size != specification.width * specification.height * specification.nchannels:
            raise RuntimeError("depth EXR has an unexpected pixel count")
        depth = pixels[..., 0].reshape((specification.height, specification.width))
    finally:
        image_input.close()
    os.unlink(exr_candidates[-1])
    np.save(output / f"camera_{ordinal}_depth.npy", depth)
    valid = np.isfinite(depth) & (depth > 0) & (depth < 1.0e6)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], [1.0, 99.0])
        if high <= low:
            high = low + 1.0e-6
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
        preview[valid] = ((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    preview_path = output / f"camera_{ordinal}_depth.png"
    image_output = oiio.ImageOutput.create(str(preview_path))
    if image_output is None:
        raise RuntimeError(f"could not create depth preview: {preview_path}")
    specification = oiio.ImageSpec(preview.shape[1], preview.shape[0], 1, oiio.UINT8)
    image_output.open(str(preview_path), specification)
    image_output.write_image(preview[..., None])
    image_output.close()


def render_scene(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    scenes = sorted((scene for scene in bpy.data.scenes if scene.camera), key=lambda item: item.name)
    if len(scenes) != 4:
        raise ValueError(f"expected four camera scenes, found {len(scenes)}")
    records = []
    for ordinal, scene in enumerate(scenes):
        if bpy.context.window is not None:
            bpy.context.window.scene = scene
        scene.render.resolution_percentage = 100
        _save_rgb_and_depth(scene, output, ordinal)
        records.append({"camera": scene.camera.name, "scene": scene.name,
                        "resolution": [scene.render.resolution_x, scene.render.resolution_y]})
    manifest = {"schema": "stage2-render-v1", "views": records}
    (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def render_rgb_scene(output: Path) -> dict[str, Any]:
    """Render only the four RGB views for Agent-side visual comparison."""
    output.mkdir(parents=True, exist_ok=True)
    scenes = sorted((scene for scene in bpy.data.scenes if scene.camera), key=lambda item: item.name)
    if len(scenes) != 4:
        raise ValueError(f"expected four camera scenes, found {len(scenes)}")
    records = []
    for ordinal, scene in enumerate(scenes):
        if bpy.context.window is not None:
            bpy.context.window.scene = scene
        available_engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
        scene.render.engine = ("BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines
                               else "BLENDER_EEVEE")
        _add_neutral_lighting(scene)
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.film_transparent = False
        scene.render.filepath = str(output / f"camera_{ordinal}_rgb.png")
        bpy.ops.render.render(write_still=True)
        records.append({"camera": scene.camera.name, "scene": scene.name,
                        "resolution": [scene.render.resolution_x, scene.render.resolution_y]})
    manifest = {"schema": "stage2-rgb-render-v1", "views": records}
    (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def apply_script(base: Path, script: Path, output: Path, frozen: Path) -> dict[str, Any]:
    if Path(bpy.data.filepath).resolve() != base.resolve():
        raise RuntimeError(f"Blender loaded {bpy.data.filepath}, expected {base}")
    before = frozen_signature()
    expected = json.loads(frozen.read_text(encoding="utf-8"))
    if before != expected:
        raise RuntimeError("base scene frozen signature differs from run baseline")
    _restore_robot_material_colors()
    source = script.read_text(encoding="utf-8")
    namespace = {"__name__": "__stage2_edit__", "__file__": str(script)}
    exec(compile(source, str(script), "exec"), namespace, namespace)
    after = frozen_signature()
    if after != expected:
        raise RuntimeError("edit script changed a frozen camera or robot asset")
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    return {"output": str(output), "frozen_signature": after, "passed": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inspect", "render", "render-rgb",
                                             "add-default-lighting", "apply", "freeze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--script", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        result = inspect_scene()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif args.command == "render":
        result = render_scene(args.output)
    elif args.command == "render-rgb":
        result = render_rgb_scene(args.output)
    elif args.command == "add-default-lighting":
        result = add_default_lighting(args.output)
    elif args.command == "freeze":
        current = frozen_signature()
        expected = json.loads(args.frozen.read_text(encoding="utf-8"))
        result = {"passed": current == expected, "current": current, "expected": expected}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        if not args.base or not args.script or not args.frozen:
            raise ValueError("apply requires --base, --script and --frozen")
        result = apply_script(args.base.resolve(), args.script.resolve(), args.output.resolve(), args.frozen.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None))
