"""Derive a robot_control-compatible SIM scene from a Stage 3 MJCF export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def derive(source: Path, output: Path) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = ET.parse(source).getroot()
    mounts = [body for body in root.find("worldbody") if body.tag == "body"
              and body.get("name", "").endswith("_mount")]
    if len(mounts) != 1 or len(mounts[0]) != 1:
        raise ValueError("Expected exactly one embedded robot below a direct worldbody mount")
    mount = mounts[0]
    robot_root = mount[0].get("name")
    if robot_root not in ("base_link", "fr3_link0", "base"):
        raise ValueError(f"Unsupported embedded robot root: {robot_root}")
    mount_attributes = dict(mount.attrib)
    # Unnamed robot elements must not match omitted constraint attributes.
    old_robot_bodies = {body.get("name") for body in mount.iter("body")
                        if body.get("name") is not None}
    old_robot_joints = {joint.get("name") for joint in mount.iter()
                        if joint.tag in ("joint", "freejoint") and joint.get("name") is not None}
    old_robot_geoms = {geom.get("name") for geom in mount.iter("geom")
                       if geom.get("name") is not None}
    for child in list(mount):
        mount.remove(child)

    for section in ("actuator", "equality", "contact"):
        container = root.find(section)
        if container is None:
            continue
        for element in list(container):
            uses_robot = (
                any(element.get(attribute) in old_robot_joints
                    for attribute in ("joint", "joint1", "joint2"))
                or any(element.get(attribute) in old_robot_bodies
                       for attribute in ("body1", "body2"))
                or any(element.get(attribute) in old_robot_geoms
                       for attribute in ("geom1", "geom2"))
            )
            if uses_robot:
                container.remove(element)
        if not len(container):
            root.remove(container)
    keyframes = root.find("keyframe")
    if keyframes is not None:
        root.remove(keyframes)

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    grasp_parent = "gripper_base" if robot_root == "base_link" else "fr3_hand"
    existing_equalities = {item.get("name") for item in equality}
    for body in root.find("worldbody").iter("body"):
        body_name = body.get("name")
        if body_name in (None, mount_attributes["name"]) or not any(
                joint.tag == "freejoint" for joint in body):
            continue
        equality_name = f"sim_grasp_{body_name}"
        if equality_name not in existing_equalities:
            ET.SubElement(equality, "weld", name=equality_name,
                          body1=grasp_parent, body2=body_name, active="false")

    asset = root.find("asset")
    if asset is None:
        raise ValueError("Stage 3 scene has no asset section")
    worldbody = root.find("worldbody")
    referenced_meshes = {geom.get("mesh") for geom in worldbody.iter("geom")}
    referenced_materials = {geom.get("material") for geom in worldbody.iter("geom")}
    referenced_textures = {material.get("texture") for material in asset.findall("material")
                           if material.get("name") in referenced_materials}
    for entry in list(asset):
        asset_file = (entry.get("file") or "").replace("\\", "/")
        robot_mesh = (
            asset_file.startswith(("fr3/", "piper/", "franka/", "Piper/"))
            or "/assets/meshes/fr3/" in asset_file
            or "/assets/meshes/piper/" in asset_file
            or "/assets/meshes/franka/" in asset_file
        )
        if entry.tag == "mesh" and (entry.get("name") not in referenced_meshes or robot_mesh):
            asset.remove(entry)
        elif entry.tag == "material" and entry.get("name") not in referenced_materials:
            asset.remove(entry)
        elif (entry.tag == "texture" and entry.get("name") not in referenced_textures
              and entry.get("type") != "skybox"):
            asset.remove(entry)

    asset_source = source.parent / "assets"
    asset_output = output.parent / "assets"
    if not asset_source.is_dir():
        raise FileNotFoundError(asset_source)
    if asset_output.exists() or asset_output.is_symlink():
        if asset_output.resolve() != asset_source:
            raise FileExistsError(f"Unexpected existing assets path: {asset_output}")
    else:
        asset_output.symlink_to(asset_source, target_is_directory=True)

    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    derived_root = ET.parse(output).getroot()
    derived_mount = derived_root.find(f"worldbody/body[@name='{mount_attributes['name']}']")
    if derived_mount is None or derived_mount.attrib != mount_attributes or len(derived_mount):
        raise AssertionError(f"Derived {mount_attributes['name']} did not remain empty and fixed")
    source_bodies = {body.get("name") for body in ET.parse(source).getroot().find("worldbody")
                     if body.tag == "body" and body.get("name") != "piper_mount"}
    derived_bodies = {body.get("name") for body in worldbody
                      if body.tag == "body" and body.get("name") != "piper_mount"}
    if source_bodies != derived_bodies:
        raise AssertionError("Environment bodies changed during derivation")
    result = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "derived": str(output),
        "derived_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "mount": mount_attributes,
        "preserved_world_bodies": sorted(source_bodies),
        "preserved_cameras": sorted(camera.get("name") for camera in derived_root.iter("camera")),
    }
    (output.parent / "scene_derivation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
