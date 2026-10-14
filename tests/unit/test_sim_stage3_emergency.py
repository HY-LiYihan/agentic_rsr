from pathlib import Path

import pytest

from sim.harness import TabletopTask, emergency_inside_box, run_episode
from sim.policy_loop_controller import policy_loop_prompt_for


def test_emergency_goal_requires_interior_height_and_clear_walls():
    box = (0.36, 0.15, 0.003)
    assert emergency_inside_box((0.359, 0.149, 0.023), box)
    assert not emergency_inside_box((0.359, -0.16, 0.023), box)
    assert not emergency_inside_box((0.455, 0.15, 0.023), box)
    assert not emergency_inside_box((0.359, 0.149, 0.15), box)


def test_emergency_prompt_describes_scene_specific_controls():
    task = TabletopTask("emergency stop button", "green storage box",
                        "Place the stop button inside the storage box", "stage3_emergency")
    prompt = policy_loop_prompt_for(task)
    assert "0.14..0.22" in prompt
    assert "api.robot.pick_place" in prompt
    assert "policy.py" in prompt


def test_derived_stage3_scene_exact_episode(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("robot_control")
    scene = (Path(__file__).resolve().parents[2] / "runs/sim_stage3_emergency_trial/"
             "scene_for_sim.xml")
    if not scene.exists():
        pytest.skip("Optional derived Stage 3 scene not installed")
    task = TabletopTask("emergency stop button", "green storage box",
                        "Place emergency stop inside green storage box", "stage3_emergency")
    policy = '''def run(api):
    objects = api.observe()["objects"]
    api.robot.pick_place("emergency stop button", objects["emergency stop button"]["xy_m"], objects["green storage box"]["xy_m"])
    return {"status": "completed"} if api.task.check_success() else {"status": "not_in_box"}
'''
    result = run_episode(scene, task, policy, "exact", 0, tmp_path, "", 0)
    assert result["success"]
    assert emergency_inside_box(result["after_xyz"][task.source],
                                result["after_xyz"][task.target])
