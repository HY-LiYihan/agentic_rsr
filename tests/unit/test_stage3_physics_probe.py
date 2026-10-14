import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from real2sim.stage3.physics_probe import (_grasp_geometry, auto_probe, check_grasp, load_scene, main, patch_scene,
                                           run_probe, task_objects, try_parameters)


PAPER_SCENE = Path(__file__).resolve().parents[1] / "fixtures/stage3/code_as_policy_tabletop.xml"


def write_probe_scene(path):
    root = ET.Element("mujoco", {"model": "probe"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "size": "1 1 0.1"})
    body = ET.SubElement(worldbody, "body", {"name": "cube", "pos": "0 0 0.3"})
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", {
        "name": "cube_geom", "type": "box", "size": "0.02 0.02 0.02",
        "mass": "0.1", "friction": "0.6 0.005 0.0001",
    })
    ET.ElementTree(root).write(path, encoding="utf-8")


def test_franka_robot_bodies_are_not_task_objects(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene.xml"
    write_probe_scene(scene)
    root = ET.parse(scene).getroot()
    mount = ET.SubElement(root.find("worldbody"), "body", {"name": "fr3_mount"})
    arm = ET.SubElement(mount, "body", {"name": "fr3_arm"})
    ET.SubElement(arm, "geom", {"name": "fr3_arm_geom", "type": "box",
                                   "size": "0.01 0.01 0.01"})
    ET.ElementTree(root).write(scene, encoding="utf-8")
    assert [item["body"] for item in task_objects(scene, attach_piper="never")["objects"]] == ["cube"]


def test_run_probe_applies_force_and_rotation(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene.xml"
    write_probe_scene(scene)
    report = run_probe(scene, {
        "duration": 0.12,
        "settle": 0.01,
        "actions": [
            {"body": "cube", "type": "force", "start": 0.02,
             "duration": 0.06, "vector": [0.2, 0, 0]},
            {"body": "cube", "type": "rotate", "at": 0.02,
             "axis_angle": [0, 0, 1, 0.3]},
        ],
        "checks": [
            {"body": "cube", "type": "move", "min_m": 1e-5, "max_m": 0.2},
            {"body": "cube", "type": "rotation", "min_rad": 0.1, "max_rad": 1},
        ],
    }, attach_piper="never")
    assert report["passed"] is True
    assert report["responses"]["cube"]["displacement_m"] > 1e-5
    assert report["responses"]["cube"]["rotation_rad"] > 0.1
    assert report["stability"]["finite_state"] is True


def test_probe_rejects_action_outside_simulation_window(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene.xml"
    write_probe_scene(scene)
    with pytest.raises(ValueError, match="within"):
        run_probe(scene, {"duration": 0.1, "actions": [
            {"body": "cube", "type": "force", "start": 0.2,
             "duration": 0.1, "vector": [0.1, 0, 0]},
        ]}, attach_piper="never")


def test_patch_scene_updates_mass_and_friction(tmp_path):
    pytest.importorskip("mujoco")
    import mujoco

    scene = tmp_path / "scene.xml"
    output = tmp_path / "patched.xml"
    write_probe_scene(scene)
    report = patch_scene(scene, output, {
        "cube": {"mass_kg": 0.2, "friction": 0.25},
    }, attach_piper="never")
    assert report["passed"] is True
    model = mujoco.MjModel.from_xml_path(str(output))
    body_id = model.body("cube").id
    geom_id = next(index for index in range(model.ngeom)
                   if model.geom_bodyid[index] == body_id)
    assert model.body_mass[body_id] == pytest.approx(0.2)
    assert model.geom_friction[geom_id].tolist() == pytest.approx([0.25, 0.005, 0.0001])


def test_patch_scene_updates_free_body_initial_keyframe_pose(tmp_path):
    pytest.importorskip("mujoco")
    import mujoco

    scene = tmp_path / "scene.xml"
    output = tmp_path / "candidate.xml"
    write_probe_scene(scene)
    root = ET.parse(scene).getroot()
    keyframes = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframes, "key", {"name": "stage2_initial", "qpos": "0 0 0.3 1 0 0 0"})
    ET.ElementTree(root).write(scene, encoding="utf-8")
    patch_scene(scene, output, {"cube": {"position": [0.03, 0.01, 0.4],
                                         "quaternion": [0.92387953, 0, 0, 0.38268343]}},
                attach_piper="never")
    original = load_scene(scene, attach_piper="never")
    candidate = load_scene(output, attach_piper="never")
    try:
        body_id = candidate.model.body("cube").id
        assert original.data.xpos[body_id].tolist() == pytest.approx([0, 0, 0.3])
        assert candidate.data.xpos[body_id].tolist() == pytest.approx([0.03, 0.01, 0.4])
        assert candidate.data.xquat[body_id].tolist() == pytest.approx(
            [0.92387953, 0, 0, 0.38268343], abs=1e-7)
        key_id = mujoco.mj_name2id(candidate.model, mujoco.mjtObj.mjOBJ_KEY, "stage2_initial")
        assert candidate.model.key_qpos[key_id][:3].tolist() == pytest.approx([0.03, 0.01, 0.4])
    finally:
        original.close()
        candidate.close()


def test_patch_scene_preserves_other_free_body_keyframe_pose(tmp_path):
    pytest.importorskip("mujoco")

    scene = tmp_path / "scene.xml"
    output = tmp_path / "candidate.xml"
    write_probe_scene(scene)
    root = ET.parse(scene).getroot()
    worldbody = root.find("worldbody")
    other = ET.SubElement(worldbody, "body", {"name": "other", "pos": "0.2 0 0.5"})
    ET.SubElement(other, "freejoint")
    ET.SubElement(other, "geom", {"type": "box", "size": "0.02 0.02 0.02", "mass": "0.1"})
    keyframes = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframes, "key", {"name": "stage2_initial",
                                       "qpos": "0 0 0.3 1 0 0 0 0.2 0 0.5 1 0 0 0"})
    ET.ElementTree(root).write(scene, encoding="utf-8")
    patch_scene(scene, output, {"cube": {"position": [0.01, 0, 0.3]}}, attach_piper="never")
    candidate = load_scene(output, attach_piper="never")
    try:
        assert candidate.data.xpos[candidate.model.body("cube").id].tolist() == pytest.approx([0.01, 0, 0.3])
        assert candidate.data.xpos[candidate.model.body("other").id].tolist() == pytest.approx([0.2, 0, 0.5])
    finally:
        candidate.close()


def test_patch_scene_replaces_collision_proxy_and_moves_center_of_mass(tmp_path):
    pytest.importorskip("mujoco")
    import mujoco

    source = tmp_path / "source.xml"
    candidate = tmp_path / "candidate.xml"
    write_probe_scene(source)
    root = ET.parse(source).getroot()
    root.find("./worldbody/body/geom").set("name", "cube_geom_collision")
    backdrop = ET.SubElement(root.find("worldbody"), "body", {"name": "backdrop"})
    ET.SubElement(backdrop, "geom", {"name": "backdrop_visual", "type": "box",
                                     "size": "0.2 0.2 0.2", "contype": "0", "conaffinity": "0"})
    ET.SubElement(backdrop, "geom", {"name": "backdrop_collision", "type": "box",
                                     "size": "0.2 0.2 0.2"})
    ET.ElementTree(root).write(source, encoding="utf-8")
    patch_scene(source, candidate, {
        "cube": {"center_of_mass": [0, 0.01, -0.01], "mass_kg": 0.005,
                 "collision": {"geom": "cube_geom_collision", "mode": "box",
                               "pos": [0, 0, 0], "size": [0.02, 0.02, 0.005]}},
        "backdrop": {"collision": {"geom": "backdrop_collision", "mode": "none"}},
    }, attach_piper="never")
    model = mujoco.MjModel.from_xml_path(str(candidate))
    assert model.body_ipos[model.body("cube").id].tolist() == pytest.approx([0, 0.01, -0.01])
    assert model.body_mass[model.body("cube").id] == pytest.approx(0.005)
    assert model.geom_size[model.geom("cube_geom_collision").id].tolist() == pytest.approx([0.02, 0.02, 0.005])
    assert "backdrop_visual" in [model.geom(index).name for index in range(model.ngeom)]
    assert "backdrop_collision" not in [model.geom(index).name for index in range(model.ngeom)]
    assert "backdrop_collision" in source.read_text()


def test_patch_scene_rejects_removing_moving_collision(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene.xml"
    write_probe_scene(scene)
    root = ET.parse(scene).getroot()
    root.find("./worldbody/body/geom").set("name", "cube_collision")
    ET.ElementTree(root).write(scene, encoding="utf-8")
    with pytest.raises(ValueError, match="static backdrops"):
        patch_scene(scene, tmp_path / "candidate.xml", {
            "cube": {"collision": {"geom": "cube_collision", "mode": "none"}},
        }, attach_piper="never")


def test_patch_scene_supports_evidence_based_joint_axis(tmp_path):
    pytest.importorskip("mujoco")
    import mujoco

    scene = tmp_path / "drawer.xml"
    output = tmp_path / "corrected.xml"
    write_probe_scene(scene)
    root = ET.parse(scene).getroot()
    body = root.find("worldbody/body[@name='cube']")
    body.remove(body.find("freejoint"))
    ET.SubElement(body, "joint", {"name": "slide", "type": "slide", "axis": "0 0 1"})
    ET.ElementTree(root).write(scene, encoding="utf-8")
    patch_scene(scene, output, {"cube": {"joint_axis": [0, 2, 0]}}, attach_piper="never")
    assert mujoco.MjModel.from_xml_path(str(output)).joint("slide").axis.tolist() == pytest.approx([0, 1, 0])
    assert mujoco.MjModel.from_xml_path(str(scene)).joint("slide").axis.tolist() == pytest.approx([0, 0, 1])


def test_patched_mass_changes_force_response(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene.xml"
    tuned = tmp_path / "heavy.xml"
    write_probe_scene(scene)
    patch_scene(scene, tuned, {"cube": {"mass_kg": 0.2}}, attach_piper="never")
    script = {"duration": 0.08, "settle": 0,
              "actions": [{"body": "cube", "type": "force", "vector": [0.2, 0, 0]}]}
    original = run_probe(scene, script, attach_piper="never")
    heavier = run_probe(tuned, script, attach_piper="never")
    original_x = original["responses"]["cube"]["final_position"][0]
    heavier_x = heavier["responses"]["cube"]["final_position"][0]
    assert heavier_x / original_x == pytest.approx(0.5, rel=0.01)


def test_wrench_action_and_patch_source_guard(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene.xml"
    write_probe_scene(scene)
    report = run_probe(scene, {
        "duration": 0.08, "settle": 0.01,
        "actions": [{"body": "cube", "type": "wrench", "start": 0,
                     "duration": 0.04, "force": [0.2, 0, 0], "torque": [0, 0, 0.01]}],
    }, attach_piper="never")
    assert report["stability"]["max_applied_force_n"] == pytest.approx(0.2)
    assert report["stability"]["max_applied_torque_nm"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="output must differ"):
        patch_scene(scene, scene, {"cube": {"mass_kg": 0.2}}, attach_piper="never")


def test_auto_probe_tests_each_free_body(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene.xml"
    write_probe_scene(scene)
    report = auto_probe(scene, duration=0.08, settle=0.01,
                        force_n=0.2, torque_nm=0.01, attach_piper="never")
    assert report["passed"] is True
    assert [item["body"] for item in report["tested_bodies"]] == ["cube"]
    assert report["tested_bodies"][0]["responded"] is True
    assert report["tested_bodies"][0]["response_vs_baseline"]["displacement_m"] > 1e-6


def test_try_parameters_compares_one_object_without_changing_original(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene.xml"
    write_probe_scene(scene)
    original_xml = scene.read_bytes()
    report = try_parameters(scene, "cube", tmp_path / "trial", mass_kg=0.2,
                            duration=0.08, settle=0, force_n=0.2)
    assert report["passed"] is True
    assert report["before"]["parameters"]["mass_kg"] == pytest.approx(0.1)
    assert report["after"]["parameters"]["mass_kg"] == pytest.approx(0.2)
    assert report["after"]["response_vs_baseline"]["displacement_m"] < (
        report["before"]["response_vs_baseline"]["displacement_m"])
    assert scene.read_bytes() == original_xml
    assert (tmp_path / "trial/candidate.xml").is_file()
    assert json.loads((tmp_path / "trial/report.json").read_text())["body"] == "cube"


def test_try_cli_prints_one_json_document(tmp_path, capsys):
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene.xml"
    write_probe_scene(scene)
    result = main(["try", "--scene", str(scene), "--body", "cube",
                   "--mass-kg", "0.2", "--duration", "0.08", "--settle", "0",
                   "--force-n", "0.2", "--output", str(tmp_path / "cli_trial")])
    assert result == 0
    assert json.loads(capsys.readouterr().out)["body"] == "cube"


def test_try_parameters_changes_only_sliding_friction(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "scene.xml"
    write_probe_scene(scene)
    report = try_parameters(scene, "cube", tmp_path / "friction_trial", friction=0.25,
                            duration=0.08, settle=0.01, force_n=0.2, torque_nm=0.01)
    assert report["before"]["parameters"]["mass_kg"] == pytest.approx(0.1)
    assert report["after"]["parameters"]["mass_kg"] == pytest.approx(0.1)
    assert report["after"]["parameters"]["friction"][0] == pytest.approx(
        [0.25, 0.005, 0.0001])


def test_try_keeps_relative_mesh_assets_loadable(tmp_path):
    pytest.importorskip("mujoco")
    original = tmp_path / "original"
    original.mkdir()
    mesh = original / "assets/meshes/task/cube.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("\n".join([
        "v 0 0 0", "v 0.02 0 0", "v 0 0.02 0", "v 0 0 0.02",
        "f 1 3 2", "f 1 2 4", "f 2 3 4", "f 3 1 4", "",
    ]), encoding="utf-8")
    scene = original / "scene.xml"
    write_probe_scene(scene)
    root = ET.parse(scene).getroot()
    ET.SubElement(root, "compiler", {"meshdir": "assets/meshes"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": "cube_visual", "file": "task/cube.obj"})
    body = root.find("worldbody/body[@name='cube']")
    ET.SubElement(body, "geom", {"type": "mesh", "mesh": "cube_visual",
                                 "contype": "0", "conaffinity": "0", "mass": "0"})
    ET.ElementTree(root).write(scene, encoding="utf-8")
    report = try_parameters(scene, "cube", tmp_path / "trial", mass_kg=0.2,
                            duration=0.08, settle=0, force_n=0.2)
    assert report["passed"] is True
    candidate = ET.parse(tmp_path / "trial/candidate.xml").getroot()
    assert candidate.find("asset/mesh").get("file") == str(mesh)
    assert candidate.find("compiler").get("meshdir") == str(mesh.parent.parent)


def test_auto_probe_tests_slide_and_hinge(tmp_path):
    pytest.importorskip("mujoco")
    root = ET.Element("mujoco", {"model": "jointed_probe"})
    world = ET.SubElement(root, "worldbody")
    for name, joint_type, axis in (("drawer", "slide", "1 0 0"),
                                   ("lid", "hinge", "0 0 1")):
        body = ET.SubElement(world, "body", {"name": name, "pos": "0 0 1"})
        ET.SubElement(body, "joint", {"type": joint_type, "axis": axis,
                                      "range": "-0.5 0.5", "limited": "true"})
        ET.SubElement(body, "geom", {"type": "box", "size": "0.05 0.05 0.05",
                                     "mass": "0.1"})
    scene = tmp_path / "jointed.xml"
    ET.ElementTree(root).write(scene, encoding="utf-8")
    report = auto_probe(scene, duration=0.08, settle=0.01, force_n=0.1,
                        torque_nm=0.01, attach_piper="never")
    assert report["passed"] is True
    assert [item["body"] for item in report["tested_bodies"]] == ["drawer", "lid"]
    assert all(item["responded"] for item in report["tested_bodies"])


def test_code_as_policy_fixture_auto_sweep_and_patch(tmp_path):
    pytest.importorskip("mujoco")
    objects = task_objects(PAPER_SCENE)["objects"]
    assert [item["body"] for item in objects] == [
        "blue_block", "green_block", "red_block", "yellow_bowl", "blue_bowl", "green_bowl",
    ]
    handle = load_scene(PAPER_SCENE)
    try:
        assert handle.model.body("gripper_base").id > 0
        assert handle.model.body_mass[handle.model.body("blue_block").id] == pytest.approx(0.04)
    finally:
        handle.close()
    sweep = auto_probe(PAPER_SCENE, duration=0.08, settle=0.03)
    assert sweep["passed"] is True
    assert [item["body"] for item in sweep["tested_bodies"]] == [
        "blue_block", "green_block", "red_block",
    ]
    assert "yellow_bowl" in sweep["skipped_fixed_bodies"]
    tuned_scene = tmp_path / "tuned.xml"
    patch_scene(PAPER_SCENE, tuned_scene, {
        "blue_block": {"mass_kg": 0.05, "friction": [0.45, 0.005, 0.0001]},
    })
    tuned = load_scene(tuned_scene)
    try:
        assert tuned.model.body_mass[tuned.model.body("blue_block").id] == pytest.approx(0.05)
    finally:
        tuned.close()


def test_grasp_cli_uses_frictional_jaws_and_preserves_source(tmp_path, capsys):
    pytest.importorskip("mujoco")
    original = PAPER_SCENE.read_bytes()
    destination = tmp_path / "grasp"
    status = main(["grasp", "--scene", str(PAPER_SCENE), "--body", "blue_block",
                   "--normal-forces", "0.4", "--output", str(destination)])
    report = json.loads(capsys.readouterr().out)
    assert status == 0
    assert report["method"] == "two_frictional_actuated_box_jaws"
    assert report["control_mode"] == "symmetric_position_servo"
    assert report["jaw_mass_kg"] == 0.025
    assert report["max_actuator_force_per_jaw_n"] == 10.0
    assert report["trials"][0]["lifted"] is True
    assert json.loads((destination / "report.json").read_text())["passed"] is True
    assert PAPER_SCENE.read_bytes() == original
    low_friction = main(["grasp", "--scene", str(PAPER_SCENE), "--body", "blue_block",
                         "--normal-forces", "0.4", "--jaw-friction", "0.01"])
    assert low_friction == 2
    assert json.loads(capsys.readouterr().out)["trials"][0]["lifted"] is False


def test_grasp_includes_compound_walls_but_excludes_visual_geoms(tmp_path):
    pytest.importorskip("mujoco")
    scene = tmp_path / "container.xml"
    write_probe_scene(scene)
    root = ET.parse(scene).getroot()
    box = root.find("worldbody/body[@name='cube']")
    ET.SubElement(box, "geom", {"name": "visual", "type": "box", "size": "0.2 0.2 0.2",
                                    "contype": "0", "conaffinity": "0", "mass": "0"})
    wall = ET.SubElement(box, "body", {"name": "attached_wall", "pos": "0.04 0 0"})
    ET.SubElement(wall, "geom", {"name": "wall_collision", "type": "box",
                                   "size": "0.004 0.02 0.02", "mass": "0"})
    ET.ElementTree(root).write(scene, encoding="utf-8")

    augmented = _grasp_geometry(scene, "cube", 0.1, 1.0, 0.006, "never")
    contact_names = {pair.get("geom2") for pair in augmented.findall("contact/pair")}
    assert contact_names == {"cube_geom", "wall_collision"}
    assert "visual" not in contact_names
    assert any(float(jaw.get("pos").split()[0]) > 0.04
               for jaw in augmented.findall("worldbody/body") if jaw.get("name") == "stage3_probe_jaw_right")
    report = check_grasp(scene, "cube", [1.0], attach_piper="never")
    assert report["trials"][0]["peak_normal_force_per_jaw_n"][1] > 0
    along_y = _grasp_geometry(scene, "cube", 0.1, 1.0, 0.006, "never", "y")
    jaw = next(body for body in along_y.findall("worldbody/body")
               if body.get("name") == "stage3_probe_jaw_left")
    assert jaw.find("joint").get("axis") == "0 1 0"
    assert along_y.find("contact/pair[@geom2='wall_collision']") is not None
