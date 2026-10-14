import base64
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_stage3_codex_batch import (assess_report, author_policy,
                                             request_policy_execution, task_prompt)
from scripts.stage3_cli import run_command
from sim.codex_sdk import SimCodexSession
from sim.program_runner import run_program
from sim.stage3_harness import Stage3Tools
from sim.stage3_observation import Stage3RGBDObservation, visual_prompt
from sim.stage3_scene import DOWN_QUATERNION, Stage3Scene


def test_stage3_prompt_describes_only_minimal_agent_interface():
    prompt = task_prompt("piper", "put_block", "Put the block in the tray.",
                         ["Block", "Tray"])
    for tool in ("api.state()", "api.observe(", "api.robot.move_pose(",
                 "api.robot.gripper(", "api.robot.move_trajectory(", "api.reset()",
                 "api.return_control(", "api.stop()"):
        assert tool in prompt
    for tool in ("- api.robot.pick_place(", "- api.robot.grasp(",
                 "- api.object.set_joint(", "- api.task.check_success("):
        assert tool not in prompt
    assert '"Block"' in prompt and '"Tray"' in prompt
    assert "link6" not in prompt
    assert "0.1425" not in prompt and "0.13" not in prompt
    assert "0.1425" not in task_prompt("franka", "put_block", "Place it.", ["Block"])


def test_stage3_author_thread_is_ephemeral_not_resumable(monkeypatch, tmp_path):
    options = []

    class FakeCodex:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def thread_start(self, **kwargs):
            options.append(kwargs)
            return SimpleNamespace(id="transient-thread")

    monkeypatch.setitem(sys.modules, "openai_codex", SimpleNamespace(
        Codex=FakeCodex, CodexConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        Sandbox=SimpleNamespace(read_only="read-only"),
        ApprovalMode=SimpleNamespace(deny_all="deny_all"),
    ))
    with SimCodexSession("test", tmp_path, ephemeral=True) as session:
        assert session.id == "transient-thread"
        assert options[0]["ephemeral"] is True
        assert options[0]["sandbox"] == "read-only"
    with pytest.raises(ValueError, match="cannot be resumed"):
        SimCodexSession("test", tmp_path, thread_id="transient-thread", ephemeral=True)


def test_author_can_practice_then_submit_valid_policy(monkeypatch, tmp_path):
    responses = [
        {"mode": "practice", "command": "state", "files": []},
        {"mode": "policy", "command": "", "files": [{
            "path": "policy.py", "source": "def run(api):\n    return {'status': 'failed'}\n"}]},
    ]

    class Session:
        def __init__(self):
            self.prompts = []

        def run(self, prompt, schema, output, images=()):
            self.prompts.append(prompt)
            return responses.pop(0)

    monkeypatch.setattr("scripts.run_stage3_codex_batch.request_command",
                        lambda socket, command: {"attempt": 0, "tool": command})
    session = Session()
    source = author_policy(session, "write policy", tmp_path / "practice.sock", tmp_path,
                           (), time.monotonic() + 10)
    assert "def run(api)" in source
    assert "Practice CLI result" in session.prompts[1]
    assert '"command": "state"' in (tmp_path / "practice_actions.jsonl").read_text()


def test_agent_explicitly_authorizes_policy_execution(tmp_path):
    class Session:
        def __init__(self):
            self.calls = 0

        def run(self, prompt, schema, output, images=()):
            self.calls += 1
            assert "mode=\"execute\"" in prompt
            return {"mode": "execute", "command": "run_policy", "files": []}

    source = "def run(api):\n    return {'status': 'completed'}\n"
    assert request_policy_execution(Session(), source, tmp_path,
                                    time.monotonic() + 10) == source


def test_stage3_reset_and_return_control_are_bounded_controls():
    class Scene:
        def __init__(self):
            self.reset_count = 0
            self.robot = SimpleNamespace(stop=lambda: None)

        def reset(self):
            self.reset_count += 1

        def state(self):
            return {"attempt": self.reset_count}

    class Perception:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    scene = Scene()
    perception = Perception()
    tools = Stage3Tools(scene, "task", "instruction", perception=perception)
    assert tools.dispatch("reset") == {
        "status": "reset", "attempt": 1, "state": {"attempt": 1}}
    assert scene.reset_count == 1 and perception.reset_count == 0
    checkpoint = tools.dispatch("return_control", "need another observation")
    assert checkpoint["status"] == "return_control"
    assert checkpoint["reason"] == "need another observation"
    tools.dispatch("stop")
    with pytest.raises(RuntimeError, match="Episode stopped"):
        tools.dispatch("state")
    assert tools.dispatch("reset")["attempt"] == 2
    assert tools.dispatch("state")["attempt"] == 2
    for _ in range(8):
        assert tools.dispatch("reset")["status"] == "reset"


def test_stage3_cli_shares_tool_state_and_hides_image_base64():
    class Tools:
        def __init__(self):
            self.attempt = 0

        def dispatch(self, name, *args):
            if name == "reset":
                self.attempt += 1
                return {"status": "reset", "attempt": self.attempt}
            if name == "observe":
                return {"images": {"third_person": {
                    "path": "/tmp/third_person.jpg", "rgb_base64": "hidden"}}}
            return {"attempt": self.attempt}

    tools = Tools()
    assert run_command(tools, "reset")["attempt"] == 1
    assert run_command(tools, "state")["attempt"] == 1
    assert run_command(tools, "observe")["images"]["third_person"] == {
        "path": "/tmp/third_person.jpg"}


def test_stage3_return_control_ends_current_program_segment():
    calls = []

    def dispatch(name, *args):
        calls.append((name, args))
        return {"status": "return_control", "reason": args[0]}

    result = run_program(
        'def run(api):\n'
        '    api.return_control("inspect result")\n'
        '    api.stop()\n',
        dispatch, timeout=10, interface="stage3")
    assert result["status"] == "returned"
    assert result["checkpoint"]["reason"] == "inspect result"
    assert [name for name, _ in calls] == ["return_control"]


def test_checkpoint_cannot_be_swallowed_by_policy_exception_handler():
    def dispatch(name, *args):
        return {"status": "return_control", "reason": args[0]}

    result = run_program('def run(api):\n    try:\n'
                         '        api.return_control("inspect")\n'
                         '    except Exception:\n        return {"status": "ignored"}\n',
                         dispatch, timeout=10, interface="stage3")
    assert result["status"] == "returned"


def test_stop_requires_independent_review_to_count_as_success(tmp_path):
    report = {"program": {"status": "finished", "return": {"status": "completed"}},
              "actions": [{"tool": "stop"}]}
    assessed = assess_report(report.copy(), tmp_path)
    assert assessed["policy_claimed_success"]
    assert not assessed["success"]
    assert assessed["verification"] == "pending_video_review"
    (tmp_path / "video_review.json").write_text('{"verdict": "passed"}')
    assessed = assess_report(report, tmp_path)
    assert assessed["success"]
    assert assessed["verification"] == "video_review_passed"


def test_reset_excludes_previous_attempt_stop_from_final_checkpoint(tmp_path):
    report = {"program": {"status": "returned", "checkpoint": {"status": "return_control"}},
              "actions": [{"tool": "stop"}, {"tool": "reset"},
                          {"tool": "return_control"}]}
    assessed = assess_report(report, tmp_path)
    assert not assessed["success"]
    assert assessed["verification"] == "checkpoint_pending"


def test_stage3_program_exposes_only_requested_tools():
    calls = []

    def dispatch(name, *args):
        calls.append((name, args))
        return {"status": "completed"}

    source = '''def run(api):
    api.state()
    api.observe(["block"])
    api.robot.move_pose(0.35, 0.0, 0.2, (0, 0, 1, 0))
    api.robot.gripper(0.02)
    api.robot.move_trajectory([{"xyz_m": [0.35, 0, 0.3], "quaternion_wxyz": [0, 0, 1, 0]}])
    try:
        api.robot.pick_place("block", "tray")
    except Exception:
        pass
    api.stop()
    return {"status": "finished"}
'''
    result = run_program(source, dispatch, timeout=10, interface="stage3")
    assert result["status"] == "finished"
    assert [name for name, _ in calls] == ["state", "observe", "robot.move_pose",
                                          "robot.gripper", "robot.move_trajectory", "stop"]


def test_rgbd_observation_returns_image_and_only_detected_names(tmp_path):
    mujoco = pytest.importorskip("mujoco")

    class Scene:
        def __init__(self):
            self.mujoco = mujoco

        def read_rgbd(self):
            intrinsics = SimpleNamespace(width=8, height=8, fx=8, fy=8, cx=4, cy=4)
            return SimpleNamespace(color=np.full((8, 8, 3), 128, dtype=np.uint8),
                                   depth=np.ones((8, 8)), depth_scale=1.0,
                                   intrinsics=intrinsics)

        def camera_pose(self):
            return {"xyz_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}

        def render_camera(self, name, width, height):
            assert name == "stage3_camera_0" and (width, height) == (512, 288)
            return np.full((height, width, 3), 96, dtype=np.uint8)

    class Segmenter:
        def segment(self, image, prompt, expected_shape):
            assert image.is_file()
            assert expected_shape == (8, 8)
            mask = np.zeros((8, 8), dtype=bool)
            if prompt == "red block":
                mask[2:6, 2:6] = True
            return mask

    perception = Stage3RGBDObservation(Scene(), ("BrownBlock_A", "Tray"),
                                       tmp_path, "unused", 0, segmenter=Segmenter())
    observed = perception.observe({"BrownBlock_A": "red block", "Tray": "hidden tray"})
    assert list(observed["objects"]) == ["BrownBlock_A"]
    assert observed["objects"]["BrownBlock_A"]["xyz_m"][2] == pytest.approx(1)
    assert observed["image"]["mime_type"] == "image/jpeg"
    assert observed["images"]["third_person"]["path"].endswith(
        "observation_000_third_person.jpg")
    assert observed["camera"]["third_person"] == "stage3_camera_0"
    assert base64.b64decode(observed["image"]["rgb_base64"]) == Path(
        observed["image"]["path"]).read_bytes()
    following = perception.observe({"BrownBlock_A": "red block"})
    assert following["image"]["path"].endswith("observation_001_wrist.jpg")
    with pytest.raises(ValueError, match="Unknown object"):
        perception.observe({"not_public": "something"})
    assert visual_prompt("Stage2_EStopHousing") == "emergency stop housing"


def test_trajectory_interpolates_and_rejects_invalid_waypoints_before_motion():
    class Scene:
        def __init__(self):
            self.poses = []
            self.widths = []
            self.robot = SimpleNamespace(state=lambda: SimpleNamespace(
                pose=SimpleNamespace(position=(0, 0, 0), quaternion=(1, 0, 0, 0))),
                stop=lambda: None)

        def move_pose(self, pose, duration_s=None):
            self.poses.append((pose, duration_s))

        def gripper(self, width):
            self.widths.append(width)

        def state(self):
            position = self.poses[-1][0].position if self.poses else (0, 0, 0)
            return {"gripper": {"width_m": self.widths[-1] if self.widths else 0.08},
                    "end_effector_pose": {"xyz_m": position}}

    scene = Scene()
    tools = Stage3Tools(scene, "task", "instruction")
    with pytest.raises(ValueError, match="gripper_width_m"):
        tools.dispatch("robot.move_trajectory", [
            {"xyz_m": [0.08, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
            {"xyz_m": [0.1, 0, 0], "quaternion_wxyz": [1, 0, 0, 0],
             "gripper_width_m": 1.0},
        ])
    assert not scene.poses
    result = tools.dispatch("robot.move_trajectory", [
        {"xyz_m": [0.08, 0, 0], "quaternion_wxyz": [1, 0, 0, 0],
         "gripper_width_m": 0},
    ])
    assert result["motion_segments"] == 2
    assert [round(pose.position[0], 2) for pose, _ in scene.poses] == [0.04, 0.08]
    assert scene.widths == [0]
    with pytest.raises(ValueError, match="not available"):
        tools.dispatch("robot.pick_place", "block", "tray")
    with pytest.raises(ValueError, match="not available"):
        tools.dispatch("object.set_joint", "drawer", 0)


def test_physical_gripper_automatically_attaches_only_on_bilateral_contact():
    pytest.importorskip("robot_control")
    scene_file = (Path(__file__).resolve().parents[2] / "runs/sim_processed_stage3_all/"
                  "piper/put_emergency_stop_button_in_green_storage_box/scene_for_sim.xml")
    if not scene_file.is_file():
        pytest.skip("Derived Stage 3 scene is not available")
    scene = Stage3Scene(scene_file, "piper")
    try:
        ee_id = scene.model.body("link6").id
        ee_rotation = scene.data.xmat[ee_id].reshape(3, 3)
        for finger in ("gripper_link1", "gripper_link2"):
            finger_id = scene.model.body(finger).id
            local = ee_rotation.T @ (scene.data.xpos[finger_id] - scene.data.xpos[ee_id])
            assert local[2] == pytest.approx(0.1425, abs=1e-4)
        name = "Stage2_EStopHousing"
        weld_id = scene.model.eq(f"sim_grasp_{name}").id
        scene.gripper(0.0)
        assert not scene.data.eq_active[weld_id]
        scene.gripper(0.08)
        from sim.randomization import Pose3D
        scene.move_pose(Pose3D((0.3528, -0.16, 0.20), DOWN_QUATERNION))
        scene.move_pose(Pose3D((0.3528, -0.16, 0.15), DOWN_QUATERNION))
        scene.gripper(0.0)
        assert scene.data.eq_active[weld_id]
        scene.move_pose(Pose3D((0.3528, -0.16, 0.20), DOWN_QUATERNION))
        assert scene.body_xyz(name)[2] > 0.05
        scene.gripper(0.08)
        assert not scene.data.eq_active[weld_id]
    finally:
        scene.close()
