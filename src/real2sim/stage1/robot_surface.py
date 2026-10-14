"""Sample fixed-pose Piper and FR3 visual surfaces for rigid registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from .geometry import ASSET_ROOT, MODEL, transform
from .piper_wrist import ASSET_ROOT as WRIST_ASSET_ROOT, assembly_sha256, assembly_sources, piper_description, stand_mesh_parts


ROOT = Path(__file__).resolve().parents[3]
FRANKA_URDF = ROOT / "external/robot_control/vendor/fr3_d435i/urdf/fr3_d435i_official.urdf"
POSES = {
    "piper": {**dict(zip((f"joint{index}" for index in range(1, 7)),
                           np.deg2rad([0, 30, -45, 0, 60, 0]))),
              "gripper": 0.07},
    "franka": {**dict(zip((f"fr3_joint{index}" for index in range(1, 8)),
                            (0.0, -math.pi / 4, 0.0, -3 * math.pi / 4,
                             0.0, math.pi / 2, math.pi / 4))),
               "fr3_finger_joint1": 0.04},
}
BASE_LINKS = {"piper": "base_link", "franka": "fr3_link0"}


def joint_motion(joint: ET.Element, value: float) -> np.ndarray:
    motion = np.eye(4)
    kind = joint.get("type")
    if kind in {"revolute", "continuous", "prismatic"}:
        axis_element = joint.find("axis")
        axis = np.asarray(list(map(float, (axis_element.get("xyz", "1 0 0")
                                         if axis_element is not None else "1 0 0").split())))
        if kind == "prismatic":
            motion[:3, 3] = axis * value
        else:
            axis /= np.linalg.norm(axis)
            skew = np.array([[0, -axis[2], axis[1]],
                             [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            motion[:3, :3] += math.sin(value) * skew + (1 - math.cos(value)) * skew @ skew
    elif kind != "fixed":
        raise ValueError(f"unsupported joint type: {kind}")
    return motion


def link_poses(root: ET.Element, base_link: str, requested: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    joints = root.findall("joint")
    by_name = {joint.get("name"): joint for joint in joints}
    if set(requested) - set(by_name):
        raise ValueError(f"unknown joints: {set(requested) - set(by_name)}")
    values = dict(requested)

    def value_for(name: str) -> float:
        if name in values:
            return values[name]
        joint = by_name[name]
        mimic = joint.find("mimic")
        values[name] = (value_for(mimic.get("joint")) * float(mimic.get("multiplier", "1")) +
                        float(mimic.get("offset", "0"))) if mimic is not None else 0.0
        return values[name]

    for joint in joints:
        value = value_for(joint.get("name"))
        limit = joint.find("limit")
        if limit is not None and (value < float(limit.get("lower", "-inf")) - 1e-8 or
                                  value > float(limit.get("upper", "inf")) + 1e-8):
            raise ValueError(f"{joint.get('name')}={value} exceeds URDF limits")
    children = {joint.find("child").get("link") for joint in joints}
    frames = {link.get("name"): np.eye(4) for link in root.findall("link")
              if link.get("name") not in children}
    pending = list(joints)
    while pending:
        remaining = []
        for joint in pending:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent in frames:
                frames[child] = frames[parent] @ transform(joint.find("origin")) @ joint_motion(
                    joint, values[joint.get("name")])
            else:
                remaining.append(joint)
        if len(remaining) == len(pending):
            raise ValueError("URDF joints do not form a connected tree")
        pending = remaining
    if base_link not in frames:
        raise ValueError(f"missing base link: {base_link}")
    inverse_base = np.linalg.inv(frames[base_link])
    return {name: inverse_base @ pose for name, pose in frames.items()}, values


def visual_mesh_path(uri: str, urdf: Path, robot: str) -> Path:
    prefix = "package://agx_arm_description/agx_arm_urdf/"
    if robot == "piper":
        if uri.startswith(prefix):
            path = ASSET_ROOT / uri.removeprefix(prefix)
        elif uri.startswith("package://piper_description/"):
            path = WRIST_ASSET_ROOT / "piper_description" / uri.removeprefix("package://piper_description/")
        elif uri.startswith("package://realsense2_description/"):
            path = WRIST_ASSET_ROOT / "realsense2_description" / uri.removeprefix("package://realsense2_description/")
        else:
            raise ValueError(f"unsupported Piper mesh URI: {uri}")
    else:
        path = urdf.parent / uri
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def visual_meshes(root: ET.Element, urdf: Path, robot: str, frames: dict[str, np.ndarray]) -> list[trimesh.Trimesh]:
    meshes = []
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            element = geometry.find("mesh")
            if element is None:
                raise ValueError(f"unsupported {robot} visual geometry in {link.get('name')}")
            path = visual_mesh_path(element.get("filename"), urdf, robot)
            mesh = (stand_mesh_parts()[0] if robot == "piper" and link.get("name") == "camera_stand_link"
                    else trimesh.load(path, force="scene").to_geometry())
            if element.get("scale"):
                mesh.apply_scale(list(map(float, element.get("scale").split())))
            mesh.apply_transform(frames[link.get("name")] @ transform(visual.find("origin")))
            if mesh.area > 0 and len(mesh.faces):
                meshes.append(mesh)
    if not meshes:
        raise ValueError(f"{robot} has no visual mesh surfaces")
    return meshes


def colored_visual_meshes(root: ET.Element, urdf: Path, robot: str,
                          frames: dict[str, np.ndarray]) -> list[tuple[trimesh.Trimesh, tuple[float, ...]]]:
    materials = {material.get("name"): material.find("color")
                 for material in root.findall("material")}
    parts = []
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            element = visual.find("geometry/mesh")
            if element is None:
                continue
            path = visual_mesh_path(element.get("filename"), urdf, robot)
            material = visual.find("material")
            color = material.find("color") if material is not None else None
            if color is None and material is not None:
                color = materials.get(material.get("name"))
            rgba = tuple(map(float, color.get("rgba").split())) if color is not None else None
            meshes = (stand_mesh_parts() if robot == "piper" and link.get("name") == "camera_stand_link"
                      else trimesh.load(path, force="scene").dump(concatenate=False))
            for mesh in meshes:
                if element.get("scale"):
                    mesh.apply_scale(list(map(float, element.get("scale").split())))
                mesh.apply_transform(frames[link.get("name")] @ transform(visual.find("origin")))
                if mesh.area <= 0 or not len(mesh.faces):
                    continue
                embedded = getattr(getattr(mesh.visual, "material", None), "main_color", None)
                mesh_rgba = rgba or (tuple(float(value) / 255 for value in embedded)
                                     if embedded is not None else (0.65, 0.65, 0.65, 1.0))
                parts.append((mesh, mesh_rgba))
    if not parts:
        raise ValueError(f"{robot} has no visual mesh surfaces")
    return parts


def sample_surface(meshes: list[trimesh.Trimesh], count: int, voxel: float,
                   seed: int = 2026) -> np.ndarray:
    if count < 1 or not math.isfinite(voxel) or voxel <= 0:
        raise ValueError("count and voxel size must be positive")
    combined = trimesh.util.concatenate(meshes)
    rng = np.random.default_rng(seed)
    selected = np.empty((0, 3), dtype=float)
    for factor in (4, 8, 16):
        candidates, _ = trimesh.sample.sample_surface(combined, count * factor, seed=seed + factor)
        candidates = candidates.astype(np.float32)
        cells = np.floor(candidates / voxel).astype(np.int64)
        _, unique = np.unique(cells, axis=0, return_index=True)
        selected = candidates[np.sort(unique)]
        if len(selected) >= count:
            break
    if len(selected) > count:
        selected = selected[rng.choice(len(selected), size=count, replace=False)]
    if not np.isfinite(selected).all():
        raise ValueError("non-finite mesh samples")
    return selected.astype(np.float32)


def generate(robot: str, output: Path, count: int = 40000, voxel: float = 0.003) -> dict:
    if robot not in POSES:
        raise ValueError(f"unknown robot: {robot}")
    urdf = MODEL if robot == "piper" else FRANKA_URDF
    root = piper_description() if robot == "piper" else ET.parse(urdf).getroot()
    frames, values = link_poses(root, BASE_LINKS[robot], POSES[robot])
    meshes = visual_meshes(root, urdf, robot, frames)
    points = sample_surface(meshes, count, voxel)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "surface.npz", points=points)
    trimesh.points.PointCloud(points).export(output / "surface.ply")
    report = {
        "robot": robot, "urdf": urdf.relative_to(ROOT).as_posix(),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "base_frame": BASE_LINKS[robot], "unit": "m", "joint_positions": values,
        "requested_points": count, "sampled_points": len(points),
        "voxel_size_m": voxel, "random_seed": 2026, "visual_mesh_count": len(meshes),
        "surface_area_m2": float(sum(mesh.area for mesh in meshes)),
        "bounds_min_m": points.min(axis=0).tolist(), "bounds_max_m": points.max(axis=0).tolist(),
    }
    if robot == "piper":
        report["assembly_sha256"] = assembly_sha256()
        report["camera_and_stand_sources"] = assembly_sources()
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def generate_pyramid(robot: str, output: Path, sizes: tuple[int, ...] = (40000, 10000, 5000),
                     voxel: float = 0.003) -> dict:
    if not sizes or len(set(sizes)) != len(sizes) or min(sizes) < 1:
        raise ValueError("sizes must be distinct positive integers")
    report = generate(robot, output, max(sizes), voxel)
    with np.load(output / "surface.npz") as data:
        points = data["points"]
    if len(points) < max(sizes):
        raise ValueError(f"only {len(points)} points survived voxelization, need {max(sizes)}")
    permutation = np.random.default_rng(2027).permutation(len(points))
    for size in sorted(sizes, reverse=True):
        subset = points if size == len(points) else points[permutation[:size]]
        if size == len(points):
            (output / "surface.npz").replace(output / f"surface_{size}.npz")
            (output / "surface.ply").replace(output / f"surface_{size}.ply")
        else:
            np.savez_compressed(output / f"surface_{size}.npz", points=subset)
            trimesh.points.PointCloud(subset).export(output / f"surface_{size}.ply")
    report["resolutions"] = sorted(sizes, reverse=True)
    report["subsample_seed"] = 2027
    report["small_clouds_are_subsets_of_large"] = True
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample fixed-pose robot visual surfaces")
    parser.add_argument("--robot", choices=("piper", "franka", "all"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "asset/robot_surfaces")
    parser.add_argument("--sizes", type=int, nargs="+", default=(40000, 10000, 5000))
    parser.add_argument("--voxel", type=float, default=0.003, help="Deduplication grid in metres")
    args = parser.parse_args()
    for robot in POSES if args.robot == "all" else (args.robot,):
        print(json.dumps(generate_pyramid(robot, args.output / robot, tuple(args.sizes), args.voxel), indent=2))


if __name__ == "__main__":
    main()
