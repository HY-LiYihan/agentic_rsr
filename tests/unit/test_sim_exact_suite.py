import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_sim_exact_suite.py"
spec = importlib.util.spec_from_file_location("evaluate_sim_exact_suite", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SuiteTools = module.SuiteTools


class FakeScene:
    def __init__(self):
        self.positions = {
            "blue block": [-0.05, -0.12, 0.773],
            "green block": [-0.13, 0.04, 0.773],
            "yellow bowl": [0.0, 0.15, 0.75],
        }
        self.robot = SimpleNamespace(step=lambda count: None)

    def object_xyz(self, name):
        return self.positions[name]

    def gripper(self, width):
        return None

    def grasp(self, name):
        return None

    def release(self, name):
        return None


class FakeEpisode:
    def move_pose(self, pose, label):
        return None


def test_suite_tools_expose_all_objects_without_narrow_target_limit():
    scene = FakeScene()
    task = SimpleNamespace(objects=tuple(scene.positions), command="Put all blocks in a line")
    tools = SuiteTools(scene, FakeEpisode(), task, scene.positions.copy(),
                       lambda case, before, after: True)
    observed = tools.dispatch("observe")
    assert set(observed["objects"]) == set(scene.positions)
    assert observed["instruction"] == task.command
    assert observed["workspace_xy_m"] == {"min": [-0.3, -0.3], "max": [0.1, 0.3]}
    assert tools.max_calls == 128
    assert tools.dispatch("robot.pick_place", "blue block", [-0.05, -0.12],
                          [0.022397154, -0.027155275])["status"] == "completed"
    assert tools.dispatch("robot.pick_place", "blue block", [-0.05, -0.12],
                          [-0.20588225, -0.031303254])["status"] == "completed"
    with pytest.raises(ValueError, match="recent public observation"):
        tools.dispatch("robot.pick_place", "blue block", [-0.01, -0.12], [-0.165, -0.165])
    assert tools.dispatch("task.check_success") is True
