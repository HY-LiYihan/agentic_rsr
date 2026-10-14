from pathlib import Path

import numpy as np
import pytest

from sim.episode import SimEpisode
from sim.harness import RANDOMIZED_CONFIG
from sim.scene_adapter import PiperTabletopScene


REFERENCE_SCENE = Path(__file__).resolve().parents[1] / "fixtures/stage3/code_as_policy_tabletop.xml"


def test_reference_tabletop_randomizes_all_objects_without_changing_source():
    pytest.importorskip("mujoco")
    pytest.importorskip("robot_control")
    path = REFERENCE_SCENE
    before = path.read_bytes()
    scene = PiperTabletopScene(path, ("blue block", "yellow bowl"))
    try:
        nominal = scene.nominal_objects()
        episode = SimEpisode(scene, path, seed=7000, config=RANDOMIZED_CONFIG)
        context = episode.reset(nominal, scene.nominal_cameras())
        assert len(context.objects) == 6
        assert any(np.linalg.norm(np.asarray(context.objects[name].position)
                                  - nominal[name].position) > 1e-5 for name in nominal)
        assert all(np.allclose(scene.object_xyz(name), pose.position)
                   for name, pose in context.objects.items())
        assert path.read_bytes() == before
    finally:
        scene.close()


def test_reset_cancels_pending_arm_trajectory():
    pytest.importorskip("mujoco")
    pytest.importorskip("robot_control")
    path = REFERENCE_SCENE
    scene = PiperTabletopScene(path, ("blue block", "yellow bowl"))
    try:
        scene.robot.move_joints([0, 0, 0, 0, 0, 0])
        assert scene.backend._motion is not None
        scene.reset_nominal()
        assert scene.backend._motion is None
        scene.robot.step(1)
        assert np.allclose(scene.backend.data.ctrl, scene.nominal_ctrl)
    finally:
        scene.close()
