import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

from scripts.prepare_sim_scene import derive


@pytest.mark.parametrize("robot_root,mount", [("base_link", "piper_mount"),
                                             ("fr3_link0", "franka_mount")])
def test_unnamed_robot_elements_do_not_remove_task_constraints(tmp_path, robot_root, mount):
    source = tmp_path / "stage3/scene.xml"
    source.parent.mkdir()
    (source.parent / "assets").mkdir()
    source.write_text(f"""<mujoco>
      <asset/>
      <worldbody>
        <body name="{mount}" pos="1 2 3"><body name="{robot_root}">
          <joint name="robot_joint"/><joint/>
          <geom name="robot_geom"/><geom/>
          <body><geom/></body>
        </body></body>
        <body name="box"><freejoint/><geom name="box_geom"/></body>
        <body name="lid"><joint name="lid_hinge"/><geom name="lid_geom"/></body>
      </worldbody>
      <actuator>
        <motor name="lid_motor" joint="lid_hinge"/>
        <motor name="robot_motor" joint="robot_joint"/>
      </actuator>
      <equality>
        <weld name="box_lid_lock" body1="box" body2="lid" active="false"/>
        <joint name="lid_lock" joint1="lid_hinge"/>
        <weld name="robot_lock" body1="{robot_root}" body2="box"/>
      </equality>
      <contact>
        <exclude name="task_exclude" body1="box" body2="lid"/>
        <pair name="task_pair" geom1="box_geom" geom2="lid_geom"/>
        <exclude name="robot_exclude" body1="{robot_root}" body2="lid"/>
        <pair name="robot_pair" geom1="robot_geom" geom2="lid_geom"/>
      </contact>
    </mujoco>""")
    original = source.read_bytes()
    output = tmp_path / "sim/scene_for_sim.xml"
    report = derive(source, output)
    derived = ET.parse(output).getroot()
    original_root = ET.fromstring(original)
    for section, names in (("actuator", ["lid_motor"]),
                           ("equality", ["box_lid_lock", "lid_lock"]),
                           ("contact", ["task_exclude", "task_pair"])):
        for name in names:
            expected = original_root.find(f"{section}/*[@name='{name}']")
            actual = derived.find(f"{section}/*[@name='{name}']")
            assert actual is not None
            assert actual.tag == expected.tag
            assert actual.attrib == expected.attrib
        assert not any(child.get("name", "").startswith("robot_")
                       for child in derived.find(section))
    empty_mount = derived.find(f"worldbody/body[@name='{mount}']")
    assert len(empty_mount) == 0
    assert empty_mount.get("pos") == "1 2 3"
    grasp = derived.find("equality/weld[@name='sim_grasp_box']")
    assert grasp.get("body1") == ("gripper_base" if robot_root == "base_link" else "fr3_hand")
    assert source.read_bytes() == original
    assert (output.parent / "assets").resolve() == source.parent / "assets"
    saved = json.loads((output.parent / "scene_derivation.json").read_text())
    assert saved == report
    assert report["derived_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
