"""Attach the pinned Piper D435 and printed stand visual assets to the arm URDF."""

from __future__ import annotations

import copy
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from .geometry import MODEL, description, transform


ASSET_ROOT = Path(__file__).resolve().parents[3] / "external/robot_control/vendor/piper_isaac_sim"
CAMERA_URDF = ASSET_ROOT / "piper_description/urdf/piper_description_v100_realsense_camera_v2.urdf"
CAMERA_MESH = ASSET_ROOT / "realsense2_description/meshes/d435.dae"
STAND_MESH = ASSET_ROOT / "piper_description/meshes/dae/realsense_mid_stand.dae"
CAMERA_ASSETS = (MODEL, CAMERA_URDF, CAMERA_MESH, STAND_MESH)
STAND_MESH_SCALE = 0.001


def stand_mesh_parts() -> list[trimesh.Trimesh]:
    scene = trimesh.load(STAND_MESH, force="scene")
    meshes = [mesh.copy() for mesh in scene.geometry.values()]
    if len(meshes) != 1:
        raise ValueError("Piper stand must have exactly one raw COLLADA geometry")
    for mesh in meshes:
        mesh.apply_scale(STAND_MESH_SCALE)
    return meshes


def stand_mesh_correction() -> np.ndarray:
    scene = trimesh.load(STAND_MESH, force="scene")
    if len(scene.graph.nodes_geometry) != 1:
        raise ValueError("Piper stand must have exactly one COLLADA node")
    graph_transform, _ = scene.graph[scene.graph.nodes_geometry[0]]
    raw_scale = np.diag([STAND_MESH_SCALE] * 3 + [1.0])
    return raw_scale @ np.linalg.inv(graph_transform)


def origin_from_matrix(matrix: np.ndarray) -> ET.Element:
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1, 1)))
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return ET.Element("origin", xyz=" ".join(f"{value:.12g}" for value in matrix[:3, 3]),
                      rpy=" ".join(f"{value:.12g}" for value in (roll, pitch, yaw)))


def fixed_joint(root: ET.Element, name: str, parent: str, child: str,
                pose: np.ndarray) -> None:
    joint = ET.SubElement(root, "joint", name=name, type="fixed")
    joint.append(origin_from_matrix(pose))
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)


def piper_description() -> ET.Element:
    for asset in CAMERA_ASSETS:
        if not asset.is_file():
            raise FileNotFoundError(f"Piper camera/stand asset missing: {asset}; initialize robot_control nested assets")
    root = description()
    source = ET.parse(CAMERA_URDF).getroot()
    stand = source.find("./link[@name='camera_stand_link']/visual")
    camera = source.find("./link[@name='camera_link']/visual")
    if stand is None or camera is None:
        raise ValueError("pinned Piper D435 URDF lacks stand or camera visual")

    stand_link = ET.SubElement(root, "link", name="camera_stand_link")
    stand_link.append(copy.deepcopy(stand))
    camera_link = ET.SubElement(root, "link", name="d435i_camera_link")
    camera_link.append(copy.deepcopy(camera))

    stand_pose = transform(ET.Element("origin", xyz="-0.032 -0.002 0.025", rpy="0 3.14 1.57"))
    fixed_joint(root, "camera_stand_joint", "link6", "camera_stand_link", stand_pose)

    ordinary_from_v100 = np.array([[0., -1., 0., -0.00009777],
                                    [1., 0., 0., 0.00141648],
                                    [0., 0., 1., 0.], [0., 0., 0., 1.]])
    mount_v100 = transform(ET.Element("origin", xyz="-0.0315 0.064 0.027",
                                      rpy="0 -1.22 -1.57"))
    camera_offset = transform(ET.Element("origin", xyz="0.0106 0.0175 0.0125"))
    fixed_joint(root, "d435i_camera_joint", "link6", "d435i_camera_link",
                ordinary_from_v100 @ mount_v100 @ camera_offset)
    return root


def assembly_sha256() -> str:
    digest = hashlib.sha256()
    for asset in CAMERA_ASSETS:
        digest.update(asset.read_bytes())
    digest.update(ET.tostring(piper_description()))
    return digest.hexdigest()


def assembly_sources() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in CAMERA_ASSETS}
