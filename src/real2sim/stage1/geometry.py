"""Read Piper visual geometry and joint transforms from the pinned description."""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


ASSET_ROOT = Path(__file__).resolve().parents[3] / "external/robot_control/vendor/agx_arm_urdf"
MODEL = ASSET_ROOT / "piper/urdf/piper_with_gripper_description.xacro"
XACRO = "{http://www.ros.org/wiki/xacro}include"


def description(path: Path = MODEL) -> ET.Element:
    if not path.is_file():
        raise FileNotFoundError(f"Piper submodule assets missing: {path}; initialize nested submodules")
    root = ET.parse(path).getroot()
    expanded = ET.Element("robot", name=root.get("name", "piper"))
    for child in root:
        if child.tag == XACRO:
            uri = child.attrib["filename"]
            prefixes = ("package://agx_arm_description/agx_arm_urdf/",
                        "$(find agx_arm_description)/agx_arm_urdf/")
            prefix = next((candidate for candidate in prefixes if uri.startswith(candidate)), None)
            if prefix is None:
                raise ValueError(f"unsupported Xacro include: {uri}")
            expanded.extend(description(ASSET_ROOT / uri.removeprefix(prefix)))
        elif child.tag.startswith("{http://www.ros.org/wiki/xacro}"):
            raise ValueError("unsupported Xacro directive")
        else:
            expanded.append(copy.deepcopy(child))
    return expanded


def transform(origin: ET.Element | None) -> np.ndarray:
    result = np.eye(4)
    if origin is None:
        return result
    roll, pitch, yaw = map(float, origin.get("rpy", "0 0 0").split())
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    result[:3, :3] = np.array([
        [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
        [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
        [-sin_p, cos_p * sin_r, cos_p * cos_r],
    ])
    result[:3, 3] = list(map(float, origin.get("xyz", "0 0 0").split()))
    return result


def link_frames(root: ET.Element) -> dict[str, np.ndarray]:
    frames = {"world": np.eye(4)}
    pending = list(root.findall("joint"))
    while pending:
        remaining = []
        for joint in pending:
            parent = joint.find("parent").attrib["link"]
            if parent not in frames:
                remaining.append(joint)
                continue
            frames[joint.find("child").attrib["link"]] = frames[parent] @ transform(joint.find("origin"))
        if len(remaining) == len(pending):
            raise ValueError("URDF joints do not form a connected tree")
        pending = remaining
    if "base_link" not in frames:
        raise ValueError("Piper base_link is missing")
    base_inverse = np.linalg.inv(frames["base_link"])
    return {name: base_inverse @ matrix for name, matrix in frames.items()}


def visual_meshes(output: Path) -> tuple[list[dict], np.ndarray]:
    root = description()
    frames = link_frames(root)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    samples = []
    for link in root.findall("link"):
        name = link.attrib["name"]
        if name not in frames:
            continue
        for index, visual in enumerate(link.findall("visual")):
            entry = visual.find("geometry/mesh")
            if entry is None:
                continue
            uri = entry.attrib["filename"]
            prefix = "package://agx_arm_description/agx_arm_urdf/"
            if not uri.startswith(prefix):
                raise ValueError(f"unsupported Piper mesh: {uri}")
            source = ASSET_ROOT / uri.removeprefix(prefix)
            if not source.is_file():
                raise FileNotFoundError(source)
            mesh = trimesh.load(source, force="scene").to_geometry()
            if entry.get("scale"):
                mesh.apply_scale(list(map(float, entry.get("scale").split())))
            local = frames[name] @ transform(visual.find("origin"))
            destination = output / f"{name}_{index}.obj"
            mesh.export(destination)
            rows.append({"name": f"{name}_{index}", "obj": str(destination), "matrix": local.tolist()})
            points, _ = trimesh.sample.sample_surface(mesh, 350, seed=2026 + len(rows))
            samples.append(points @ local[:3, :3].T + local[:3, 3])
    if not rows or not any(row["name"].startswith("base_link") for row in rows):
        raise ValueError("Piper model has no base visual mesh")
    return rows, np.concatenate(samples).astype(np.float32)
