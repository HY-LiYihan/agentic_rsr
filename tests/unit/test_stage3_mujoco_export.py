import json
import hashlib
import xml.etree.ElementTree as ET

import pytest

from real2sim.stage3.mujoco_export import (
    CAMERA_NAMES,
    _stage2_delivery,
    _support_slab,
    _camera_element,
    _add_visual_background,
    _append_task_body,
    _append_visual_attachment,
    _write_obj,
    infer_semantics,
    validate_xml,
)


def test_infer_semantics_prefers_explicit_properties():
    result = infer_semantics("drawer_box", {
        "real2sim_role": "dynamic",
        "real2sim_joint_type": "slide",
        "real2sim_collision": "convex",
        "real2sim_mass_kg": 1.5,
    })
    assert result["role"] == "dynamic"
    assert result["joint_type"] == "slide"
    assert result["collision"] == "convex"
    assert result["mass_kg"] == 1.5
    assert result["inferred_fields"] == []


def test_infer_semantics_marks_operable_objects_dynamic():
    result = infer_semantics("emergency_button", {})
    assert result["role"] == "dynamic"
    assert result["joint_type"] == "slide"
    assert result["collision"] == "box"


def test_storage_box_is_static_by_default():
    result = infer_semantics("green_storage_box", {})
    assert result["role"] == "static"
    assert result["joint_type"] == "fixed"


def test_camera_uses_pixel_intrinsics():
    element = _camera_element({
        "name": CAMERA_NAMES[0], "pos": [1, 2, 3], "quat": [1, 0, 0, 0],
        "lens": 50, "sensor_width": 36, "sensor_height": 24,
        "shift_x": 0, "shift_y": 0, "clip_start": 0.01, "clip_end": 100,
    }, 512, 288)
    assert element.attrib["resolution"] == "512 288"
    assert element.attrib["principalpixel"] == "0 0"
    assert element.attrib["focalpixel"].startswith("711.111")


def test_camera_principalpixel_is_center_relative():
    element = _camera_element({
        "name": CAMERA_NAMES[0], "pos": [1, 2, 3], "quat": [1, 0, 0, 0],
        "lens": 50, "sensor_width": 36, "sensor_height": 24,
        "shift_x": 0.1, "shift_y": -0.1, "clip_start": 0.01, "clip_end": 100,
    }, 512, 288)
    assert element.attrib["principalpixel"] == "-51.2 -51.2"


def test_background_grid_is_visual_only_and_below_robot_base():
    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    _add_visual_background(asset, worldbody, {"pos": [0, 0, 0.3]})
    grid = worldbody.find("geom[@name='stage3_background_grid']")
    assert grid is not None
    assert grid.get("pos") == "0 0 0.2"
    assert grid.get("type") == "plane"
    assert grid.get("contype") == grid.get("conaffinity") == "0"
    assert asset.find("texture[@name='stage3_grid']").get("builtin") == "checker"
    assert asset.find("texture[@name='stage3_sky']").get("builtin") == "gradient"


def test_stage2_delivery_requires_accepted_immutable_scene_and_four_rgb(tmp_path):
    task = tmp_path / "piper" / "stage2" / "task"
    geometry = task / "geometry"
    geometry.mkdir(parents=True)
    scene = geometry / "final.blend"
    scene.write_bytes(b"unchanged")
    (task / "plan").mkdir()
    (task / "plan/plan.md").write_text("plan", encoding="utf-8")
    images = []
    for index in range(4):
        image = task / f"camera_{index}_rgb.png"
        image.write_bytes(b"image")
        images.append({"render_rgb": str(image)})
    manifest = {"accepted": True, "scene": str(scene),
                "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
                "evaluation": {"passed": True, "views": images}}
    (geometry / "stage2_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _stage2_delivery(task)["robot"] == "piper"
    scene.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="scene_sha256"):
        _stage2_delivery(task)


def test_planar_blender_mesh_is_exported_with_tiny_thickness(tmp_path):
    mesh = tmp_path / "wall.obj"
    _write_obj({"name": "wall", "vertices": [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
                "faces": [[0, 1, 2, 3]]}, mesh)
    coordinates = [tuple(float(value) for value in line.split()[1:])
                   for line in mesh.read_text().splitlines() if line.startswith("v ")]
    assert len(coordinates) == 8
    assert max(point[2] for point in coordinates) > min(point[2] for point in coordinates)


def test_write_obj_preserves_per_face_uvs(tmp_path):
    mesh = tmp_path / "packet.obj"
    _write_obj({"name": "packet", "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0.1], [0, 1, 0]],
                "faces": [[0, 1, 2], [0, 2, 3]],
                "face_uvs": [[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]]}, mesh)
    contents = mesh.read_text()
    assert contents.count("vt ") == 6
    assert "f 1/1 2/2 3/3" in contents
    assert "f 1/4 3/5 4/6" in contents


def test_write_obj_keeps_uv_on_planar_mesh_with_generated_thickness(tmp_path):
    mesh = tmp_path / "paper.obj"
    _write_obj({"name": "paper", "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                "faces": [[0, 1, 2, 3]], "face_uvs": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}, mesh)
    contents = mesh.read_text()
    assert contents.count("vt ") == 4
    assert "f 1/1 2/2 3/3 4/4" in contents
    assert contents.count("\nf ") > 1


def test_backdrop_uses_tabletop_face_as_collision_slab():
    vertices = [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0],
                [-0.5, -0.5, 0.003], [0.5, -0.5, 0.003],
                [0.5, 0.5, 0.003], [-0.5, 0.5, 0.003]]
    vertices += [[0.5, index / 30 - 0.5, 0.1 + index / 1000] for index in range(30)]
    center, size = _support_slab({"vertices": vertices, "faces": [[0, 1, 2, 3], [4, 5, 6, 7]]})
    assert center == pytest.approx([0, 0, 0])
    assert size == pytest.approx([0.5, 0.5, 0.003])


def test_decorative_mesh_follows_dynamic_body_without_another_collider():
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    assets = ET.SubElement(root, "asset")
    owner = {"name": "button", "location": [0.0, 0.0, 0.2], "quaternion": [1, 0, 0, 0],
             "vertices": [[-0.02, -0.02, -0.02], [0.02, 0.02, 0.02]],
             "materials": [], "custom_properties": {
                 "real2sim_role": "dynamic", "real2sim_joint_type": "free",
                 "real2sim_collision": "box"}}
    label = {**owner, "name": "button_label", "location": [0.01, 0.0, 0.21],
             "custom_properties": {"real2sim_role": "visual_only", "real2sim_collision": "none"}}
    rows = []
    body = _append_task_body(world, assets, owner, {}, rows)
    _append_visual_attachment(body, owner, assets, label, {}, rows)
    assert len(world.findall("body")) == 1
    assert len(body.findall("geom")) == 3
    assert len([geom for geom in body.findall("geom") if geom.get("contype") != "0"]) == 1
    assert rows[-1]["attached_to"] == "button"


def test_structural_basket_wall_adds_collision_to_moving_owner():
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    assets = ET.SubElement(root, "asset")
    owner = {"name": "basket_floor", "location": [0, 0, 0.2], "quaternion": [1, 0, 0, 0],
             "vertices": [[-0.1, -0.1, -0.005], [0.1, 0.1, 0.005]], "materials": [],
             "custom_properties": {"real2sim_role": "dynamic", "real2sim_joint_type": "free",
                                   "real2sim_collision": "box"}}
    wall = {**owner, "name": "basket_wall", "vertices": [[-0.1, -0.1, 0], [0.1, -0.095, 0.1]],
            "custom_properties": {"real2sim_role": "compound", "real2sim_joint_type": "fixed",
                                  "real2sim_collision": "box"}}
    rows = []
    body = _append_task_body(world, assets, owner, {}, rows)
    _append_visual_attachment(body, owner, assets, wall, {}, rows)
    collider = body.find("body[@name='basket_wall_compound']/geom")
    assert collider.get("type") == "box"
    assert collider.get("mass") == "0"
    assert collider.get("group") == "3"
    assert rows[-1]["semantics"]["role"] == "compound"


def test_drawer_slide_axis_follows_panel_normal():
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    assets = ET.SubElement(root, "asset")
    item = {"name": "UpperDrawer_Front", "location": [0, 0, 0], "quaternion": [1, 0, 0, 0],
            "vertices": [[-0.2, -0.01, -0.1], [0.2, 0.01, 0.1]], "materials": [],
            "custom_properties": {"real2sim_role": "operable", "real2sim_joint_type": "slide",
                                  "real2sim_collision": "box"}}
    rows = []
    body = _append_task_body(world, assets, item, {}, rows)
    assert body.find("joint").get("axis") == "0 1 0"
    assert rows[0]["joint_axis_source"] == "drawer_panel_normal_estimate"


def test_open_container_has_bottom_and_walls_but_no_lid():
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    assets = ET.SubElement(root, "asset")
    item = {"name": "cup", "location": [0, 0, 0], "quaternion": [1, 0, 0, 0],
            "vertices": [[-0.05, -0.05, 0], [0.05, 0.05, 0.1]], "materials": [],
            "custom_properties": {"real2sim_role": "dynamic", "real2sim_joint_type": "free",
                                  "real2sim_collision": "open_container", "real2sim_mass_kg": 0.4}}
    rows = []
    body = _append_task_body(world, assets, item, {}, rows)
    colliders = [geom for geom in body.findall("geom") if geom.get("name", "").endswith("_collision")]
    assert len(colliders) == 5
    assert {geom.get("name") for geom in colliders} == {
        f"cup_{part}_collision" for part in ("floor", "left", "right", "front", "back")}
    assert sum(float(geom.get("mass")) for geom in colliders) == pytest.approx(0.4)
    assert body.find("geom[@name='cup_visual']").get("mass") == "0"
    assert body.find("geom[@name='cup_visual']").get("group") == "0"
    assert all(geom.get("group") == "3" for geom in colliders)
    assert rows[0]["collision_proxy"] == "open_container_five_boxes"


@pytest.mark.parametrize("collision", ["box", "mesh"])
def test_task_colliders_are_hidden_by_default_without_disabling_contact(collision):
    import mujoco

    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    assets = ET.SubElement(root, "asset")
    item = {"name": "paper", "location": [0, 0, 0], "quaternion": [1, 0, 0, 0],
            "vertices": [[-0.1, -0.1, 0], [0.1, 0.1, 0.1]], "materials": [],
            "custom_properties": {"real2sim_role": "dynamic", "real2sim_joint_type": "free",
                                  "real2sim_collision": collision, "real2sim_mass_kg": 0.05}}
    body = _append_task_body(world, assets, item, {}, [])
    visual = body.find("geom[@name='paper_visual']")
    collider = body.find("geom[@name='paper_collision']")
    assert visual.get("group") == "0"
    assert visual.get("contype") == visual.get("conaffinity") == "0"
    assert collider.get("group") == "3"
    assert float(collider.get("mass")) == pytest.approx(0.05)
    assert collider.get("contype") is None and collider.get("conaffinity") is None
    assert mujoco.MjvOption().geomgroup[int(collider.get("group"))] == 0


def test_validate_xml_accepts_external_piper_scene(tmp_path):
    pytest.importorskip("mujoco")
    root = ET.Element("mujoco", {"model": "test"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "512", "offheight": "288"})
    worldbody = ET.SubElement(root, "worldbody")
    mount = ET.SubElement(worldbody, "body", {"name": "piper_mount", "pos": "0 0 0", "quat": "1 0 0 0"})
    current = ET.SubElement(mount, "body", {"name": "base_link"})
    ET.SubElement(current, "geom", {"type": "box", "size": "0.1 0.1 0.1", "mass": "1"})
    for index in range(8):
        ET.SubElement(current, "joint", {"name": f"joint{index}", "type": "hinge", "axis": "0 0 1"})
        current = ET.SubElement(current, "body", {"name": f"link{index}"})
        ET.SubElement(current, "geom", {"type": "box", "size": "0.01 0.01 0.01", "mass": "0.01"})
    camera_body = ET.SubElement(worldbody, "body", {"name": "stage3_cameras"})
    camera_args = {"pos": [0, 0, 1], "quat": [1, 0, 0, 0], "lens": 50,
                   "sensor_width": 36, "sensor_height": 24, "shift_x": 0, "shift_y": 0,
                   "clip_start": 0.01, "clip_end": 100}
    for name in CAMERA_NAMES:
        camera_body.append(_camera_element({"name": name, **camera_args}, 512, 288))
    scene = tmp_path / "scene.xml"
    ET.ElementTree(root).write(scene, encoding="unicode")
    report = validate_xml(scene)
    assert report["passed"] is True
    assert report["joints"] >= 8
