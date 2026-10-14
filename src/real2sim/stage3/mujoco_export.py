"""Compile a Stage 2 Blender scene into a MuJoCo MJCF scene.

The Blender robot is used only to recover the registered Piper base pose. Piper
geometry, joints, inertias, gripper and actuators come from the pinned external
URDF/Xacro adapter. All non-camera, non-Piper mesh objects are exported as
MuJoCo visual geoms, with explicit task-object semantics recorded in a manifest.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from real2sim.blender import resolve_blender

CAMERA_NAMES = tuple(f"stage3_camera_{index}" for index in range(4))
EXPORTER_VERSION = 8
PIPER_NAMES = {
    "piper", "FR3", "fr3", "arm_base", "base_link", "camera_link", "camera_stand_link",
    *(f"link{index}" for index in range(1, 9)),
}
STATIC_TOKENS = (
    "table", "desk", "floor", "ground", "wall", "rack", "stand", "support",
    "shelf", "platform", "fixture", "mount", "background", "environment",
)
JOINT_TOKENS = {
    "drawer": "slide",
    "lid": "hinge",
    "cap": "hinge",
    "button": "slide",
    "switch": "slide",
}
DYNAMIC_TOKENS = (
    "cube", "block", "bottle", "cup", "tissue", "cone", "piece", "object",
    "handle", "plug", "container", "tool", "box",
)


@dataclass(frozen=True)
class ExportConfig:
    blend: Path | None
    output: Path
    piper_xacro: Path | None = None
    blender: str | None = None
    width: int | None = None
    height: int | None = None
    stage2_delivery: Path | None = None
    semantics_path: Path | None = None


def _safe_name(value: str) -> str:
    value = "".join(character if character.isalnum() or character == "_" else "_"
                    for character in value)
    value = value.strip("_") or "object"
    if value[0].isdigit():
        value = "object_" + value
    return value[:80]


def _numbers(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _quat_normalized(value: list[float] | tuple[float, ...]) -> list[float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (4,) or not np.isfinite(array).all():
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(array))
    if norm < 1e-9:
        raise ValueError("quaternion must be nonzero")
    return (array / norm).tolist()


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return str(value)


def infer_semantics(name: str, custom_properties: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic role/joint/collision semantics for one Blender object."""
    lower = name.lower()
    explicit_role = custom_properties.get("real2sim_role")
    explicit_joint = custom_properties.get("real2sim_joint_type")
    role = str(explicit_role) if explicit_role else None
    joint_type = str(explicit_joint) if explicit_joint else None
    inferred = []
    if role is None:
        if "storage" in lower and "box" in lower:
            role = "static"
        elif any(token in lower for token in JOINT_TOKENS) or any(token in lower for token in DYNAMIC_TOKENS):
            role = "dynamic"
        elif any(token in lower for token in STATIC_TOKENS):
            role = "static"
        elif any(token in lower for token in DYNAMIC_TOKENS):
            role = "dynamic"
        else:
            role = "static"
        inferred.append("role")
    if joint_type is None:
        for token, candidate in JOINT_TOKENS.items():
            if token in lower:
                joint_type = candidate
                break
        if joint_type is None:
            joint_type = "free" if role in {"dynamic", "operable"} else "fixed"
        inferred.append("joint_type")
    collision = str(custom_properties.get("real2sim_collision", ""))
    if not collision:
        collision = "box" if joint_type in {"free", "hinge", "slide"} else "mesh"
        inferred.append("collision")
    return {
        "role": role,
        "joint_type": joint_type,
        "collision": collision,
        "mass_kg": float(custom_properties.get("real2sim_mass_kg", 0.25)),
        "friction": float(custom_properties.get("real2sim_friction", 0.6)),
        "damping": float(custom_properties.get("real2sim_damping", 0.05)),
        "restitution": float(custom_properties.get("real2sim_restitution", 0.05)),
        "inferred_fields": inferred,
    }


def _external_piper_modules() -> tuple[Any, Any]:
    repo = Path(__file__).resolve().parents[3]
    source = repo / "external" / "robot_control" / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"external robot_control source is missing: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    piper_model = importlib.import_module("robot_control.backends.piper_model")
    return piper_model, importlib.import_module("robot_control.backends.scene_builder")


def _default_piper_xacro() -> Path:
    piper_model, _ = _external_piper_modules()
    return Path(piper_model.DEFAULT_MODEL).resolve()


def _stage2_delivery(path: Path) -> dict[str, Any]:
    directory = path.expanduser().resolve(strict=True)
    manifest_path = (directory / "geometry" / "stage2_manifest.json"
                     if directory.is_dir() and directory.name != "geometry" else
                     directory / "stage2_manifest.json" if directory.is_dir() else directory)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("accepted") is not True or not manifest.get("evaluation", {}).get("passed"):
        raise ValueError(f"Stage 2 delivery was not accepted: {manifest_path}")
    scene = Path(manifest["scene"]).expanduser().resolve(strict=True)
    if scene != (manifest_path.parent / "final.blend").resolve():
        raise ValueError("Stage 2 delivery must reference its own geometry/final.blend")
    if hashlib.sha256(scene.read_bytes()).hexdigest() != manifest["scene_sha256"]:
        raise ValueError("Stage 2 delivery Blend does not match scene_sha256")
    task_dir = manifest_path.parent.parent
    robot = task_dir.parent.parent.name
    if robot not in {"piper", "franka"}:
        raise ValueError(f"unsupported Stage 2 robot: {robot}")
    plan = task_dir / "plan" / "plan.md"
    if not plan.is_file():
        raise FileNotFoundError(f"missing Stage 2 plan: {plan}")
    rgb_paths = [Path(view["render_rgb"]) for view in manifest["evaluation"]["views"]]
    if len(rgb_paths) != 4 or any(not image.is_file() for image in rgb_paths):
        raise FileNotFoundError("Stage 2 delivery needs four existing evaluated RGB renders")
    return {"manifest": manifest_path, "scene": scene, "robot": robot,
            "task": task_dir.name.replace("_", " "), "plan": plan, "rgb_paths": rgb_paths}


def _run_blender_payload(config: ExportConfig) -> dict[str, Any]:
    if config.blend is None:
        raise ValueError("Stage 3 requires a source Blend scene")
    blender = resolve_blender(config.blender)
    with tempfile.TemporaryDirectory(prefix="real2sim_stage3_blender_") as directory:
        output = Path(directory) / "payload.json"
        script = _blender_script(output, config.width, config.height)
        script_path = Path(directory) / "extract.py"
        script_path.write_text(script, encoding="utf-8")
        process = __import__("subprocess").run(
            [str(blender), "--background", str(config.blend.resolve()), "--python-exit-code", "1",
             "--python", str(script_path)], capture_output=True, text=True, timeout=900)
        if process.returncode != 0 or not output.is_file():
            raise RuntimeError(
                "Blender extraction failed: " + (process.stdout + process.stderr)[-5000:])
        payload = json.loads(output.read_text(encoding="utf-8"))
        for item in payload["objects"]:
            for material in item.get("materials", []):
                source = material.get("image")
                if source and Path(source).is_relative_to(Path(directory) / "textures"):
                    destination = config.output / "assets" / "textures" / Path(source).name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    material["image"] = str(destination.resolve())
        return payload


def _blender_script(output: Path, width: int | None, height: int | None) -> str:
    return r'''import bpy, json, pathlib
from mathutils import Matrix

OUTPUT = OUTPUT_PATH
TEXTURE_OUTPUT = TEXTURE_DIR
CAMERAS = CAMERA_NAMES
PIPER_NAMES = PIPER_NAME_SET

def json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "to_list"):
        return [json_value(item) for item in value.to_list()]
    return str(value)

def piper_object(obj):
    current = obj
    while current is not None:
        lower = current.name.lower()
        if (current.name in PIPER_NAMES or lower.startswith("piper") or
                current.get("real2sim_robot") or current.get("real2sim_role") == "robot"):
            return True
        current = current.parent
    return False

def matrix_rows(value):
    return [[float(value[row][column]) for column in range(4)] for row in range(4)]

def material_data(obj):
    values = []
    for slot in obj.material_slots:
        material = slot.material
        if material is None:
            values.append({"name": "empty_slot", "rgba": [0.7, 0.7, 0.7, 1.0], "image": None})
            continue
        rgba = [float(value) for value in material.diffuse_color]
        image = None
        if material.use_nodes and material.node_tree:
            principled = next((node for node in material.node_tree.nodes
                               if node.type == "BSDF_PRINCIPLED"), None)
            if principled and principled.inputs.get("Base Color"):
                rgba = [float(item) for item in principled.inputs["Base Color"].default_value]
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    source = pathlib.Path(bpy.path.abspath(node.image.filepath)) if node.image.filepath else None
                    if source is not None and source.is_file() and not node.image.packed_file:
                        image = str(source.resolve())
                    elif node.image.packed_file:
                        import hashlib
                        data = bytes(node.image.packed_file.data)
                        suffix = pathlib.Path(node.image.filepath).suffix.lower() or '.png'
                        packed = pathlib.Path(TEXTURE_OUTPUT) / (hashlib.sha256(data).hexdigest()[:16] + suffix)
                        packed.parent.mkdir(parents=True, exist_ok=True)
                        packed.write_bytes(data)
                        image = str(packed)
                    break
        values.append({"name": material.name, "rgba": rgba, "image": image})
    return values or [{"name": "default", "rgba": [0.7, 0.7, 0.7, 1.0], "image": None}]

def object_properties(obj):
    return {str(key): json_value(value) for key, value in obj.items() if not str(key).startswith("_")}

def mesh_payload(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        matrix = obj.matrix_world
        translation = [float(item) for item in matrix.to_translation()]
        quaternion = [float(item) for item in matrix.to_quaternion()]
        scale = [float(item) for item in matrix.to_scale()]
        vertices = []
        for vertex in mesh.vertices:
            vertices.append([float(vertex.co.x * scale[0]), float(vertex.co.y * scale[1]), float(vertex.co.z * scale[2])])
        faces = [[int(vertex) for vertex in polygon.vertices] for polygon in mesh.polygons if len(polygon.vertices) >= 3]
        face_materials = [int(polygon.material_index) for polygon in mesh.polygons if len(polygon.vertices) >= 3]
        uv_layer = mesh.uv_layers.active
        face_uvs = ([[[float(uv_layer.data[index].uv.x), float(uv_layer.data[index].uv.y)]
                     for index in polygon.loop_indices] for polygon in mesh.polygons
                    if len(polygon.vertices) >= 3] if uv_layer else None)
        return {
            "name": obj.name,
            "parent": obj.parent.name if obj.parent is not None else None,
            "matrix_world": matrix_rows(matrix),
            "location": translation,
            "quaternion": quaternion,
            "scale": scale,
            "vertices": vertices,
            "faces": faces,
            "face_materials": face_materials,
            "face_uvs": face_uvs,
            "materials": material_data(obj),
            "custom_properties": object_properties(obj),
        }
    finally:
        evaluated.to_mesh_clear()

scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()
render = scene.render
render_width = int(render.resolution_x if WIDTH is None else WIDTH)
render_height = int(render.resolution_y if HEIGHT is None else HEIGHT)
root = bpy.data.objects.get("Piper") or bpy.data.objects.get("piper")
robot_type = "piper"
if root is None:
    root = bpy.data.objects.get("FR3") or bpy.data.objects.get("fr3")
    robot_type = "franka" if root is not None else "piper"
if root is None:
    root = bpy.data.objects.get("arm_base") or bpy.data.objects.get("base_link")
if root is None:
    raise RuntimeError("Stage 3 requires a Piper or FR3 Blender robot root")
camera_objects = [bpy.data.objects.get(name) or bpy.data.objects.get(f"Reference_{index:02d}")
                  for index, name in enumerate(CAMERAS)]
if any(camera is None or camera.type != "CAMERA" for camera in camera_objects):
    raise RuntimeError("Stage 3 requires four stage3_camera or Reference cameras")
payload = {
    "resolution": [render_width, render_height],
    "robot": {"type": robot_type, "mount": {
        "pos": [float(item) for item in root.matrix_world.to_translation()],
        "quat": [float(item) for item in root.matrix_world.to_quaternion()],
        "source": root.name}},
    "cameras": [], "lights": [], "objects": [],
}
for name, camera in zip(CAMERAS, camera_objects):
    data = camera.data
    payload["cameras"].append({
        "name": name,
        "source_name": camera.name,
        "pos": [float(item) for item in camera.matrix_world.to_translation()],
        "quat": [float(item) for item in camera.matrix_world.to_quaternion()],
        "lens": float(data.lens), "sensor_width": float(data.sensor_width),
        "sensor_height": float(data.sensor_height), "sensor_fit": str(data.sensor_fit),
        "shift_x": float(data.shift_x), "shift_y": float(data.shift_y),
        "clip_start": float(data.clip_start), "clip_end": float(data.clip_end),
    })
for obj in sorted(bpy.data.objects, key=lambda item: item.name):
    if obj.type == "LIGHT" and not obj.hide_render:
        payload["lights"].append({"name": obj.name, "type": obj.data.type,
                                  "pos": [float(item) for item in obj.matrix_world.to_translation()],
                                  "quat": [float(item) for item in obj.matrix_world.to_quaternion()],
                                  "energy": float(obj.data.energy)})
    if obj.type != "MESH" or obj.hide_render or piper_object(obj):
        continue
    item = mesh_payload(obj, depsgraph)
    if item["vertices"] and item["faces"]:
        payload["objects"].append(item)
pathlib.Path(OUTPUT).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
print("REAL2SIM_STAGE3_PAYLOAD_DONE")
'''.replace("OUTPUT_PATH", repr(str(output))).replace("TEXTURE_DIR", repr(str(output.parent / "textures"))).replace("CAMERA_NAMES", repr(list(CAMERA_NAMES))).replace("PIPER_NAME_SET", repr(sorted(PIPER_NAMES))).replace("WIDTH", repr(width)).replace("HEIGHT", repr(height))


def _write_obj(item: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(item["vertices"], dtype=float)
    faces = item["faces"]
    source_face_count = len(faces)
    if len(vertices) >= 3 and np.linalg.matrix_rank(vertices - vertices[0], tol=1e-7) == 2:
        _, _, basis = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)
        offset = basis[-1] * 0.0005
        vertices = np.concatenate((vertices + offset, vertices - offset))
        offset_index = len(item["vertices"])
        faces = (list(faces) + [[index + offset_index for index in reversed(face)] for face in faces] +
                 [[face[index], face[(index + 1) % len(face)],
                   face[(index + 1) % len(face)] + offset_index, face[index] + offset_index]
                  for face in faces for index in range(len(face))])
    elif len(vertices) < 3 or np.linalg.matrix_rank(vertices - vertices[0], tol=1e-7) < 2:
        center = vertices.mean(axis=0)
        vertices = center + np.asarray([[x, y, z] for x in (-0.0005, 0.0005)
                                        for y in (-0.0005, 0.0005) for z in (-0.0005, 0.0005)])
        faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                 [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    lines = [f"# source object: {item['name']}"]
    lines.extend("v %.9g %.9g %.9g" % tuple(vertex) for vertex in vertices)
    face_uvs = item.get("face_uvs")
    if face_uvs and len(face_uvs) != source_face_count:
        raise ValueError(f"face UV count differs from face count: {item['name']}")
    uv_index = 1
    for index, face in enumerate(faces):
        if face_uvs and index < len(face_uvs) and len(face) == len(face_uvs[index]):
            lines.extend("vt %.9g %.9g" % tuple(uv) for uv in face_uvs[index])
            lines.append("f " + " ".join(f"{vertex + 1}/{uv_index + offset}" for offset, vertex in enumerate(face)))
            uv_index += len(face)
        else:
            lines.append("f " + " ".join(str(vertex + 1) for vertex in face))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_textures(objects: list[dict[str, Any]], output: Path) -> dict[str, str]:
    target = output / "assets" / "textures"
    target.mkdir(parents=True, exist_ok=True)
    copied = {}
    for item in objects:
        for material in item.get("materials", []):
            source = material.get("image")
            if not source:
                continue
            source_path = Path(source).expanduser()
            if not source_path.is_file():
                continue
            destination_name = _safe_name(source_path.stem) + source_path.suffix.lower()
            destination = target / destination_name
            if source_path.resolve() != destination.resolve() and not destination.exists():
                shutil.copy2(source_path, destination)
            copied[str(source_path)] = str(destination.relative_to(output))
    return copied


def _add_task_material(asset: ET.Element, item: dict[str, Any], texture_paths: dict[str, str],
                       material_index: int = 0) -> str:
    materials = item.get("materials") or [{"name": "default", "rgba": [0.7, 0.7, 0.7, 1.0]}]
    material = materials[material_index]
    name = _safe_name(f"mat_{item['name']}" + (f"_{material_index}" if material_index else ""))
    rgba = [float(value) for value in material.get("rgba", [0.7, 0.7, 0.7, 1.0])[:4]]
    attributes = {"name": name, "rgba": _numbers(rgba)}
    image = material.get("image")
    if image and str(Path(image).expanduser()) in texture_paths:
        texture_name = _safe_name(f"tex_{Path(image).stem}")
        if not any(element.attrib.get("name") == texture_name for element in asset.findall("texture")):
            ET.SubElement(asset, "texture", {
                "name": texture_name, "type": "2d",
                "file": Path(texture_paths[str(Path(image).expanduser())]).name})
        attributes.pop("rgba", None)
        attributes["texture"] = texture_name
    ET.SubElement(asset, "material", attributes)
    return name


def _object_quat(item: dict[str, Any]) -> list[float]:
    return _quat_normalized(item.get("quaternion", [1, 0, 0, 0]))


def _quat_product(first: list[float], second: list[float]) -> list[float]:
    first_w, first_x, first_y, first_z = first
    second_w, second_x, second_y, second_z = second
    return [first_w * second_w - first_x * second_x - first_y * second_y - first_z * second_z,
            first_w * second_x + first_x * second_w + first_y * second_z - first_z * second_y,
            first_w * second_y - first_x * second_z + first_y * second_w + first_z * second_x,
            first_w * second_z + first_x * second_y - first_y * second_x + first_z * second_w]


def _relative_pose(parent: dict[str, Any], child: dict[str, Any]) -> tuple[list[float], list[float]]:
    quaternion = _object_quat(parent)
    inverse = [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
    displacement = [float(child["location"][index] - parent["location"][index]) for index in range(3)]
    rotated = _quat_product(_quat_product(inverse, [0.0, *displacement]), quaternion)
    return rotated[1:], _quat_normalized(_quat_product(inverse, _object_quat(child)))


def _bounds(item: dict[str, Any]) -> tuple[list[float], list[float]]:
    vertices = np.asarray(item["vertices"], dtype=float)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    return minimum.tolist(), maximum.tolist()


def _support_slab(item: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    """Use a horizontal support face instead of a backdrop's concave mesh hull."""
    vertices = np.asarray(item["vertices"], dtype=float)
    if len(vertices) < 30 or np.ptp(vertices[:, 2]) < 0.05:
        return None
    footprint = float(np.ptp(vertices[:, 0]) * np.ptp(vertices[:, 1]))
    if footprint <= 0:
        return None
    planes = []
    for face in item["faces"]:
        polygon = vertices[face]
        if len(face) < 3 or np.ptp(polygon[:, 2]) > 1e-4:
            continue
        area = abs(float(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1)) -
                         np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1)))) / 2
        if area >= footprint * 0.15:
            planes.append((area, float(polygon[0, 2]), polygon))
    if not planes:
        return None
    _, surface_z, polygon = max(planes, key=lambda plane: (plane[0], plane[1]))
    lower = polygon.min(axis=0)
    upper = polygon.max(axis=0)
    half_thickness = 0.003
    return ([float((lower[0] + upper[0]) / 2), float((lower[1] + upper[1]) / 2),
             surface_z - half_thickness],
            [float((upper[0] - lower[0]) / 2), float((upper[1] - lower[1]) / 2),
             half_thickness])


def _append_visual_geoms(parent: ET.Element, asset: ET.Element, item: dict[str, Any],
                         texture_paths: dict[str, str], position: list[float] | None = None,
                         quaternion: list[float] | None = None) -> None:
    name = _safe_name(item["name"])
    parts = item.get("visual_parts") or [(name, 0)]
    for mesh_name, material_index in parts:
        attributes = {"name": f"{mesh_name}_visual", "type": "mesh", "mesh": mesh_name,
                      "material": _add_task_material(asset, item, texture_paths, material_index),
                      "contype": "0", "conaffinity": "0", "mass": "0", "group": "0"}
        if position is not None and quaternion is not None:
            attributes.update(pos=_numbers(position), quat=_numbers(quaternion))
        ET.SubElement(parent, "geom", attributes)


def _append_task_body(worldbody: ET.Element, asset: ET.Element, item: dict[str, Any],
                      texture_paths: dict[str, str], manifest_rows: list[dict[str, Any]]) -> ET.Element | None:
    name = _safe_name(item["name"])
    semantics = infer_semantics(item["name"], item.get("custom_properties", {}))
    mesh_file = f"task/{name}.obj"
    body = (None if semantics["role"] == "visual_only" else ET.SubElement(worldbody, "body", {
        "name": name, "pos": _numbers(item["location"]), "quat": _numbers(_object_quat(item))}))
    joint_type = semantics["joint_type"]
    if semantics["role"] not in {"static", "visual_only", "collision_only"}:
        if joint_type == "free":
            ET.SubElement(body, "freejoint", {"name": f"{name}_freejoint"})
        elif joint_type in {"hinge", "slide"}:
            axis = item.get("custom_properties", {}).get("real2sim_joint_axis")
            if axis is None and joint_type == "slide" and "drawer" in item["name"].lower():
                minimum, maximum = _bounds(item)
                thin_dimension = min(range(3), key=lambda index: maximum[index] - minimum[index])
                axis = [int(index == thin_dimension) for index in range(3)]
                item["joint_axis_source"] = "drawer_panel_normal_estimate"
            if axis is None:
                axis = [0, 0, 1]
            lower = float(item.get("custom_properties", {}).get("real2sim_joint_lower", -math.pi if joint_type == "hinge" else -0.2))
            upper = float(item.get("custom_properties", {}).get("real2sim_joint_upper", math.pi if joint_type == "hinge" else 0.2))
            ET.SubElement(body, "joint", {"name": f"{name}_joint", "type": joint_type,
                                            "axis": _numbers(axis), "range": _numbers((lower, upper)),
                                            "limited": "true", "damping": str(semantics["damping"])})
    _append_visual_geoms(body if body is not None else worldbody, asset, item, texture_paths,
                         item["location"] if body is None else None,
                         _object_quat(item) if body is None else None)
    if semantics["role"] != "visual_only":
        minimum, maximum = _bounds(item)
        collision = semantics["collision"]
        if collision == "box":
            center = [(minimum[index] + maximum[index]) / 2 for index in range(3)]
            size = [max((maximum[index] - minimum[index]) / 2, 1e-4) for index in range(3)]
            ET.SubElement(body, "geom", {"name": f"{name}_collision", "type": "box",
                                           "pos": _numbers(center), "size": _numbers(size),
                                           "mass": str(semantics["mass_kg"]),
                                           "friction": _numbers((semantics["friction"],) * 3),
                                           "group": "3"})
        elif collision == "open_container":
            extent = [maximum[index] - minimum[index] for index in range(3)]
            if min(extent) <= 0.01:
                raise ValueError(f"open container needs nondegenerate three-dimensional bounds: {name}")
            center = [(minimum[index] + maximum[index]) / 2 for index in range(3)]
            thickness = min(0.01, max(0.002, min(extent[0], extent[1]) * 0.1))
            wall_height = max(extent[2] - thickness, 1e-4)
            pieces = [
                ("floor", [center[0], center[1], minimum[2] + thickness / 2],
                 [extent[0] / 2, extent[1] / 2, thickness / 2]),
                ("left", [minimum[0] + thickness / 2, center[1], minimum[2] + thickness + wall_height / 2],
                 [thickness / 2, extent[1] / 2, wall_height / 2]),
                ("right", [maximum[0] - thickness / 2, center[1], minimum[2] + thickness + wall_height / 2],
                 [thickness / 2, extent[1] / 2, wall_height / 2]),
                ("front", [center[0], minimum[1] + thickness / 2, minimum[2] + thickness + wall_height / 2],
                 [max(extent[0] / 2 - thickness, 1e-4), thickness / 2, wall_height / 2]),
                ("back", [center[0], maximum[1] - thickness / 2, minimum[2] + thickness + wall_height / 2],
                 [max(extent[0] / 2 - thickness, 1e-4), thickness / 2, wall_height / 2]),
            ]
            volumes = [8 * math.prod(size) for _, _, size in pieces]
            for (part, position, size), volume in zip(pieces, volumes):
                ET.SubElement(body, "geom", {
                    "name": f"{name}_{part}_collision", "type": "box",
                    "pos": _numbers(position), "size": _numbers(size),
                    "mass": str(semantics["mass_kg"] * volume / sum(volumes)),
                    "friction": _numbers((semantics["friction"],) * 3),
                    "group": "3"})
            item["collision_proxy"] = "open_container_five_boxes"
        elif (slab := _support_slab(item)) is not None:
            center, size = slab
            ET.SubElement(body, "geom", {"name": f"{name}_collision", "type": "box",
                                           "pos": _numbers(center), "size": _numbers(size),
                                           "mass": str(semantics["mass_kg"]),
                                           "friction": _numbers((semantics["friction"],) * 3),
                                           "group": "3"})
            item["collision_proxy"] = "horizontal_support_slab"
        else:
            ET.SubElement(body, "geom", {"name": f"{name}_collision", "type": "mesh", "mesh": name,
                                           "mass": str(semantics["mass_kg"]),
                                           "friction": _numbers((semantics["friction"],) * 3),
                                           "group": "3"})
    manifest_rows.append({"name": name, "source_name": item["name"], "semantics": semantics,
                          "semantic_source": item.get("semantic_source", "scene_property_or_inference"),
                          "collision_proxy": item.get("collision_proxy"),
                          "joint_axis_source": item.get("joint_axis_source"),
                          "mesh": mesh_file, "transform": {"pos": item["location"], "quat": _object_quat(item)}})
    return body


def _append_visual_attachment(parent_body: ET.Element, parent_item: dict[str, Any],
                              asset: ET.Element, item: dict[str, Any],
                              texture_paths: dict[str, str], manifest_rows: list[dict[str, Any]]) -> None:
    name = _safe_name(item["name"])
    position, quaternion = _relative_pose(parent_item, item)
    _append_visual_geoms(parent_body, asset, item, texture_paths, position, quaternion)
    semantics = infer_semantics(item["name"], item.get("custom_properties", {}))
    if semantics["role"] == "compound":
        minimum, maximum = _bounds(item)
        center = [(minimum[index] + maximum[index]) / 2 for index in range(3)]
        size = [max((maximum[index] - minimum[index]) / 2, 1e-4) for index in range(3)]
        fixed_part = ET.SubElement(parent_body, "body", {
            "name": f"{name}_compound", "pos": _numbers(position), "quat": _numbers(quaternion)})
        ET.SubElement(fixed_part, "geom", {
            "name": f"{name}_collision", "type": "box", "pos": _numbers(center),
            "size": _numbers(size), "mass": "0",
            "friction": _numbers((semantics["friction"],) * 3), "group": "3"})
    manifest_rows.append({
        "name": name, "source_name": item["name"], "attached_to": parent_item["name"],
        "semantic_source": item.get("semantic_source", "scene_property_or_inference"),
        "semantics": semantics,
        "mesh": f"task/{name}.obj",
        "transform": {"pos": item["location"], "quat": _object_quat(item)},
    })


def _merge_piper(root: ET.Element, worldbody: ET.Element, asset: ET.Element,
                 output: Path, mount: dict[str, Any], piper_xacro: Path) -> dict[str, Any]:
    piper_model, _ = _external_piper_modules()
    component = piper_model.build_piper_scene(piper_xacro).getroot()
    piper_mesh_dir = output / "assets" / "meshes" / "piper"
    piper_mesh_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []
    component_asset = component.find("asset")
    if component_asset is not None:
        for child in list(component_asset):
            child = copy.deepcopy(child)
            if child.tag == "mesh" and child.get("file"):
                source = Path(child.attrib["file"]).resolve()
                if not source.is_file():
                    raise FileNotFoundError(f"Piper mesh is missing: {source}")
                destination = piper_mesh_dir / source.name
                if not destination.exists():
                    shutil.copy2(source, destination)
                child.set("file", f"piper/{source.name}")
                copied_files.append(str(destination.relative_to(output)))
            asset.append(child)
    component_worldbody = component.find("worldbody")
    if component_worldbody is None or len(component_worldbody) != 1:
        raise ValueError("external Piper model must contain one root body")
    mount_body = ET.SubElement(worldbody, "body", {
        "name": "piper_mount", "pos": _numbers(mount["pos"]),
        "quat": _numbers(_quat_normalized(mount["quat"]))})
    mount_body.append(copy.deepcopy(component_worldbody[0]))
    for tag in ("contact", "equality", "actuator", "sensor"):
        source = component.find(tag)
        if source is not None:
            target = root.find(tag)
            if target is None:
                target = ET.SubElement(root, tag)
            for child in list(source):
                target.append(copy.deepcopy(child))
    return {"source": str(piper_xacro), "mount": mount, "mesh_files": copied_files,
            "joint_count": len(component.findall(".//joint")),
            "actuator_count": len(component.findall("./actuator/*"))}


def _merge_franka(root: ET.Element, worldbody: ET.Element, asset: ET.Element,
                  output: Path, mount: dict[str, Any]) -> dict[str, Any]:
    source = Path(__file__).resolve().parents[3] / "external/robot_control/vendor/fr3_d435i/mujoco_fr3/fr3.xml"
    component = ET.parse(source).getroot()
    defaults = component.find("default")
    if defaults is not None:
        root.insert(list(root).index(asset), copy.deepcopy(defaults))
    mesh_dir = output / "assets/meshes/fr3"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []
    for element in component.findall("./asset/*"):
        child = copy.deepcopy(element)
        if child.tag == "mesh" and child.get("file"):
            mesh_source = source.parent / "assets" / child.attrib["file"]
            if not mesh_source.is_file():
                raise FileNotFoundError(f"FR3 mesh is missing: {mesh_source}")
            target = mesh_dir / mesh_source.name
            shutil.copy2(mesh_source, target)
            child.set("file", f"fr3/{mesh_source.name}")
            copied_files.append(str(target.relative_to(output)))
        asset.append(child)
    component_worldbody = component.find("worldbody")
    if component_worldbody is None or len(component_worldbody) != 1:
        raise ValueError("external FR3 model must contain one root body")
    robot_mount = ET.SubElement(worldbody, "body", {
        "name": "fr3_mount", "pos": _numbers(mount["pos"]),
        "quat": _numbers(_quat_normalized(mount["quat"]))})
    robot_body = copy.deepcopy(component_worldbody[0])
    for body in robot_body.iter():
        for camera in list(body.findall("camera")):
            body.remove(camera)
    robot_mount.append(robot_body)
    for tag in ("contact", "equality", "actuator", "sensor", "tendon"):
        source_tag = component.find(tag)
        if source_tag is not None:
            target = root.find(tag)
            if target is None:
                target = ET.SubElement(root, tag)
            for child in source_tag:
                target.append(copy.deepcopy(child))
    return {"source": str(source), "mount": mount, "mesh_files": copied_files,
            "joint_count": len(component.findall(".//joint")),
            "actuator_count": len(component.findall("./actuator/*"))}


def _camera_element(camera: dict[str, Any], width: int, height: int) -> ET.Element:
    fx = float(camera["lens"]) / float(camera["sensor_width"]) * width
    fy = float(camera["lens"]) / float(camera["sensor_height"]) * height
    cx = width * (0.5 - float(camera["shift_x"]))
    cy = height / 2 + float(camera["shift_y"]) * width
    return ET.Element("camera", {
        "name": camera["name"], "pos": _numbers(camera["pos"]), "quat": _numbers(_quat_normalized(camera["quat"])),
        "focalpixel": _numbers((fx, fy)), "principalpixel": _numbers((cx - width / 2, cy - height / 2)),
        "sensorsize": _numbers((float(camera["sensor_width"]), float(camera["sensor_height"]))),
        "resolution": _numbers((width, height)),
    })


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def _add_visual_background(asset: ET.Element, worldbody: ET.Element, mount: dict[str, Any]) -> None:
    ET.SubElement(asset, "texture", {
        "name": "stage3_sky", "type": "skybox", "builtin": "gradient",
        "rgb1": "0.27 0.41 0.56", "rgb2": "0.07 0.12 0.19",
        "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {
        "name": "stage3_grid", "type": "2d", "builtin": "checker",
        "rgb1": "0.10 0.17 0.26", "rgb2": "0.05 0.09 0.15",
        "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {
        "name": "stage3_grid_material", "texture": "stage3_grid",
        "texrepeat": "5 5", "texuniform": "true", "reflectance": "0",
        "specular": "0", "rgba": "0.6 0.7 0.8 1"})
    ET.SubElement(worldbody, "geom", {
        "name": "stage3_background_grid", "type": "plane",
        "pos": _numbers((0, 0, float(mount["pos"][2]) - 0.1)),
        "size": "0 0 0.1", "material": "stage3_grid_material",
        "contype": "0", "conaffinity": "0"})


def _set_stage2_robot_pose(root: ET.Element, robot_type: str, scene: Path) -> dict[str, float]:
    from real2sim.stage1.robot_surface import POSES

    pose = POSES[robot_type]
    if robot_type == "piper":
        _external_piper_modules()
        from robot_control.config import PIPER_INITIAL_JOINTS_RAD

        arm = [f"joint{index}" for index in range(1, 7)]
        if not np.allclose([pose[name] for name in arm], PIPER_INITIAL_JOINTS_RAD, atol=1e-8):
            raise ValueError("Piper CTRL initial pose differs from Stage 2 registration pose")
        positions = {name: float(pose[name]) for name in arm}
        positions.update({"gripper_joint1": float(pose["gripper"] / 2),
                          "gripper_joint2": float(-pose["gripper"] / 2)})
    else:
        positions = {name: float(value) for name, value in pose.items() if name.startswith("fr3_joint")}
        positions.update({"fr3_finger_joint1": float(pose["fr3_finger_joint1"]),
                          "fr3_finger_joint2": float(pose["fr3_finger_joint1"])})
    joints = {joint.get("name"): joint for joint in root.findall(".//joint")}
    actuators = {actuator.get("joint"): actuator for actuator in root.findall("./actuator/*")}
    if any(name not in joints or name not in actuators for name in positions):
        raise ValueError("Stage 2 robot pose does not match the exported robot joints and actuators")
    _write_xml(root, scene)

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    for name, value in positions.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        data.ctrl[model.actuator(name).id] = value
    mujoco.mj_forward(model, data)
    if robot_type == "piper":
        from real2sim.stage1.robot_surface import link_poses, piper_description

        frames, _ = link_poses(piper_description(), "base_link", pose)
        mount = model.body("piper_mount").id
        mount_pos = data.xpos[mount]
        mount_rotation = data.xmat[mount].reshape(3, 3)
        for name in ("link3", "link4", "link6", "gripper_link1", "gripper_link2"):
            expected = mount_pos + mount_rotation @ frames[name][:3, 3]
            if not np.allclose(data.xpos[model.body(name).id], expected, atol=1e-5, rtol=0):
                raise ValueError(f"MuJoCo {name} initial pose differs from Stage 2 robot geometry")
    keyframes = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframes, "key", {"name": "stage2_initial", "qpos": _numbers(data.qpos.tolist()),
                                      "ctrl": _numbers(data.ctrl.tolist())})
    _write_xml(root, scene)
    return positions


def export_blend(config: ExportConfig) -> dict[str, Any]:
    """Export a Blender scene and validate the resulting MuJoCo XML."""
    delivery = _stage2_delivery(config.stage2_delivery) if config.stage2_delivery else None
    if delivery:
        config = replace(config, blend=delivery["scene"])
        if config.output.exists() and any(config.output.iterdir()):
            raise FileExistsError(f"Stage 3 output must be empty: {config.output}")
    if config.blend is None:
        raise ValueError("either --blend or --stage2 must provide a Blend scene")
    config.output.mkdir(parents=True, exist_ok=True)
    payload = _run_blender_payload(config)
    if delivery and payload["robot"]["type"] != delivery["robot"]:
        raise ValueError("Stage 2 delivery robot does not match the Blender scene root")
    width, height = config.width or payload["resolution"][0], config.height or payload["resolution"][1]
    if width <= 0 or height <= 0:
        raise ValueError("render resolution must be positive")
    if len(payload["cameras"]) != 4:
        raise ValueError("Stage 3 requires exactly four Blender cameras")
    output = config.output.resolve()
    if config.semantics_path or delivery:
        from .semantics_agent import classify_objects, validate_classification

        source_hash = hashlib.sha256(config.blend.read_bytes()).hexdigest()
        if config.semantics_path:
            classification = json.loads(config.semantics_path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
            if classification.get("source_sha256") != source_hash:
                raise ValueError("semantics JSON must match the source Blend SHA256")
            semantics = validate_classification(classification, payload["objects"])
            (output / "semantics.json").write_text(
                json.dumps(classification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            semantics = classify_objects(payload["objects"], delivery["task"],
                                         delivery["plan"].read_text(encoding="utf-8"),
                                         delivery["rgb_paths"], output, source_hash)
        for item in payload["objects"]:
            chosen = semantics[item["name"]]
            item["semantic_source"] = ("scene_property" if any(
                key in item.get("custom_properties", {}) for key in
                ("real2sim_role", "real2sim_joint_type", "real2sim_collision")) else
                "model" if not config.semantics_path else "reviewed_json")
            item["attached_to"] = chosen.get("attached_to")
            item["custom_properties"] = {
                "real2sim_role": chosen["role"],
                "real2sim_joint_type": chosen["joint_type"],
                "real2sim_collision": chosen["collision"],
                **item.get("custom_properties", {}),
            }
    task_dir = output / "assets" / "meshes" / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    for item in payload["objects"]:
        name = _safe_name(item["name"])
        _write_obj(item, task_dir / f"{name}.obj")
        face_materials = item.get("face_materials") or []
        if len(face_materials) != len(item["faces"]):
            continue
        distinct_materials = sorted(set(face_materials))
        if distinct_materials == [0] or not distinct_materials:
            continue
        item["visual_parts"] = []
        for material_index in distinct_materials:
            if material_index >= len(item["materials"]):
                raise ValueError(f"material index out of range: {item['name']}")
            indices = [index for index, value in enumerate(face_materials) if value == material_index]
            part = f"{name}_material_{material_index}"
            visual_item = {**item, "faces": [item["faces"][index] for index in indices],
                           "face_uvs": ([item["face_uvs"][index] for index in indices]
                                        if item.get("face_uvs") else None)}
            _write_obj(visual_item, task_dir / f"{part}.obj")
            item["visual_parts"].append((part, material_index))
    texture_paths = _copy_textures(payload["objects"], output)
    root = ET.Element("mujoco", {"model": "real2sim_stage3"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true", "fusestatic": "false",
                                       "meshdir": "assets/meshes", "texturedir": "assets/textures"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81", "integrator": "implicitfast"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": str(width), "offheight": str(height)})
    ET.SubElement(visual, "headlight", {"ambient": "0.35 0.35 0.35", "diffuse": "0.8 0.8 0.8"})
    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    _add_visual_background(asset, worldbody, payload["robot"]["mount"])
    for light in payload.get("lights", []):
        ET.SubElement(worldbody, "light", {"name": _safe_name(light["name"]), "pos": _numbers(light["pos"]),
                                             "directional": "true" if light["type"] == "SUN" else "false",
                                             "diffuse": "1 1 1", "specular": "0.2 0.2 0.2",
                                             "dir": "0 0 -1", "castshadow": "true"})
    rows = []
    for item in payload["objects"]:
        asset_mesh = ET.SubElement(asset, "mesh", {"name": _safe_name(item["name"]),
                                                     "file": f"task/{_safe_name(item['name'])}.obj"})
        del asset_mesh
        for part, _ in item.get("visual_parts", []):
            ET.SubElement(asset, "mesh", {"name": part, "file": f"task/{part}.obj"})
    physical_bodies = {}
    physical_items = {}
    for item in payload["objects"]:
        if item.get("attached_to"):
            continue
        body = _append_task_body(worldbody, asset, item, texture_paths, rows)
        if body is not None:
            physical_bodies[item["name"]] = body
            physical_items[item["name"]] = item
    for item in payload["objects"]:
        owner = item.get("attached_to")
        if owner:
            _append_visual_attachment(physical_bodies[owner], physical_items[owner],
                                      asset, item, texture_paths, rows)
    camera_body = ET.SubElement(worldbody, "body", {"name": "stage3_cameras", "pos": "0 0 0", "quat": "1 0 0 0"})
    for camera in payload["cameras"]:
        camera_body.append(_camera_element(camera, width, height))
    robot_type = payload["robot"]["type"]
    robot = (_merge_piper(root, worldbody, asset, output, payload["robot"]["mount"],
                          config.piper_xacro or _default_piper_xacro()) if robot_type == "piper" else
             _merge_franka(root, worldbody, asset, output, payload["robot"]["mount"]))
    scene = output / "scene.xml"
    initial_pose = _set_stage2_robot_pose(root, robot_type, scene)
    manifest = {
        "status": "exported", "exporter_version": EXPORTER_VERSION,
        "source_blend": str(config.blend.resolve()), "scene": str(scene),
        "stage2_robot_initial_pose": initial_pose,
        "resolution": [width, height], "cameras": [camera["name"] for camera in payload["cameras"]],
        "robot": {"type": robot_type, **robot}, "objects": rows,
        "textures": sorted(texture_paths.values()),
        "policy": {"robot": "external_piper_urdf_xacro" if robot_type == "piper" else "external_fr3_mjcf",
                   "cameras": "from_blend", "task_objects": "all_non_robot_meshes",
                   "collision": "reviewed_classification" if config.semantics_path else
                                "model_classified" if delivery else "explicit_or_inferred"},
    }
    if robot_type == "piper":
        manifest["piper"] = robot
    if delivery:
        manifest["stage2_delivery"] = str(delivery["manifest"])
        manifest["stage2_scene_sha256"] = hashlib.sha256(config.blend.read_bytes()).hexdigest()
    (output / "object_manifest.json").write_text(json.dumps({"objects": rows}, indent=2), encoding="utf-8")
    (output / "camera_manifest.json").write_text(json.dumps({"cameras": payload["cameras"], "resolution": [width, height]}, indent=2), encoding="utf-8")
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate_xml(scene, robot_type)
    manifest["status"] = "passed"
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_xml(scene: Path, robot: str = "piper") -> dict[str, Any]:
    """Load the generated MJCF and enforce Stage 3 resource contracts."""
    import mujoco
    root = ET.parse(scene).getroot()
    if root.tag != "mujoco":
        raise ValueError("generated scene root must be mujoco")
    cameras = {item.get("name") for item in root.findall(".//camera")}
    missing = sorted(set(CAMERA_NAMES) - cameras)
    if missing:
        raise ValueError("generated scene is missing cameras: " + ", ".join(missing))
    mount_name = "piper_mount" if robot == "piper" else "fr3_mount"
    mounts = root.findall(f"./worldbody/body[@name='{mount_name}']")
    if len(mounts) != 1:
        raise ValueError(f"generated scene must contain one worldbody {mount_name}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    if model.nbody < 2 or model.njnt < (8 if robot == "piper" else 7):
        raise ValueError(f"generated scene does not contain the external {robot} model")
    return {"xml": str(scene), "bodies": int(model.nbody), "joints": int(model.njnt),
            "geoms": int(model.ngeom), "cameras": sorted(cameras), "passed": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile a Stage 2 Blender scene into MuJoCo MJCF")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--blend", type=Path)
    source.add_argument("--stage2", type=Path, help="accepted Stage 2 task directory or delivery manifest")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--piper-xacro", type=Path, default=None)
    parser.add_argument("--blender", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--semantics", type=Path, help="reviewed semantics.json for a matching Blend SHA256")
    args = parser.parse_args(argv)
    manifest = export_blend(ExportConfig(args.blend, args.output, args.piper_xacro,
                                         args.blender, args.width, args.height,
                                         args.stage2, args.semantics))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
