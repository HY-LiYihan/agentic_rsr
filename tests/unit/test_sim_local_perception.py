"""SIM observations use the shared model factory without a remote endpoint."""

from types import SimpleNamespace

import numpy as np
import pytest

from sim.observation import SAM3RGBDObservation
from sim.stage3_observation import Stage3RGBDObservation


class FakeScene:
    def __init__(self):
        self.mujoco = SimpleNamespace(mju_quat2Mat=self._rotation)

    @staticmethod
    def _rotation(output, quaternion):
        output[:] = np.eye(3).reshape(-1)

    def read_rgbd(self):
        return SimpleNamespace(
            color=np.full((4, 4, 3), 128, dtype=np.uint8),
            depth=np.full((4, 4), 1000, dtype=np.uint16), depth_scale=0.001,
            intrinsics=SimpleNamespace(fx=2, fy=2, cx=0, cy=0),
        )

    def estimated_world_from_camera(self, pose):
        transform = np.eye(4)
        transform[:3, 3] = (1, 2, 3)
        return transform

    def camera_pose(self):
        return {"xyz_m": [1, 2, 3], "quaternion_wxyz": [1, 0, 0, 0]}

    def render_camera(self, name, width, height):
        return np.full((height, width, 3), 128, dtype=np.uint8)


class FakeSegmenter:
    metadata = {"backend": "local", "model": "sam3", "device": "cpu"}

    def __init__(self, masks=None):
        self.masks = masks or {}
        self.calls = []
        self.closed = 0

    def segment(self, image, prompt, expected_shape):
        assert image.is_file()
        self.calls.append((prompt, expected_shape))
        return self.masks.get(prompt, np.ones(expected_shape, dtype=bool))

    def close(self):
        self.closed += 1


def observation(kind, tmp_path, **kwargs):
    scene = FakeScene()
    names = ("container",)
    if kind == "stage3":
        return Stage3RGBDObservation(scene, names, tmp_path, **kwargs)
    episode = SimpleNamespace(context=SimpleNamespace(
        estimated_camera_pose=lambda camera: None))
    return SAM3RGBDObservation(scene, episode, names, tmp_path, **kwargs)


@pytest.mark.parametrize("kind", ("tabletop", "stage3"))
def test_injected_local_segmenter_projects_depth_without_truth(kind, tmp_path):
    segmenter = FakeSegmenter()
    perception = observation(kind, tmp_path, segmenter=segmenter)
    result = perception.observe()
    assert result["objects"]["container"]["xyz_m"] == pytest.approx([1.75, 2.75, 4])
    assert result["objects"]["container"]["bbox_pixels"] == [0, 0, 4, 4]
    assert perception.metadata == segmenter.metadata
    perception.close()
    assert segmenter.closed == 0


@pytest.mark.parametrize("kind", ("tabletop", "stage3"))
def test_shared_factory_defaults_and_owned_close(kind, tmp_path, monkeypatch):
    from perception import config

    segmenter = FakeSegmenter()
    calls = []

    def create(host=None, port=None):
        calls.append((host, port))
        return segmenter

    monkeypatch.setattr(config, "create_instance_segmenter", create)
    perception = observation(kind, tmp_path)
    assert calls == [(None, None)]
    perception.observe()
    perception.close()
    perception.close()
    assert segmenter.closed == 1


@pytest.mark.parametrize("kind", ("tabletop", "stage3"))
def test_legacy_endpoint_forwarded_to_factory(kind, tmp_path, monkeypatch):
    from perception import config

    calls = []

    def create(host, port):
        calls.append((host, port))
        return FakeSegmenter()

    monkeypatch.setattr(config, "create_instance_segmenter", create)
    perception = observation(kind, tmp_path, host="legacy", port=9000)
    assert calls == [("legacy", 9000)]
    perception.close()


@pytest.mark.parametrize("kind", ("tabletop", "stage3"))
def test_empty_detection_and_bad_mask_grid(kind, tmp_path):
    segmenter = FakeSegmenter({"container": np.zeros((4, 4), dtype=bool)})
    perception = observation(kind, tmp_path, segmenter=segmenter)
    assert perception.observe()["objects"] == {}
    segmenter.masks["container"] = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="does not match RGB-D"):
        perception.observe()


def test_tabletop_fallback_uses_same_segmenter(tmp_path):
    segmenter = FakeSegmenter({"container": np.zeros((4, 4), dtype=bool)})
    perception = observation("tabletop", tmp_path, segmenter=segmenter,
                             fallback_prompts={"container": "tray"})
    assert "container" in perception.observe()["objects"]
    assert segmenter.calls == [("container", (4, 4)), ("tray", (4, 4))]


def test_practice_model_is_closed_before_evaluation(monkeypatch, tmp_path):
    from scripts import run_stage3_codex_batch as batch

    commands = []
    path = tmp_path / "practice.sock"
    path.touch()
    process = SimpleNamespace(poll=lambda: None,
                              wait=lambda timeout: commands.append(("wait", timeout)),
                              terminate=lambda: commands.append(("terminate",)))
    monkeypatch.setattr(batch, "request_command", lambda path, command, timeout:
                        commands.append((command, timeout)))
    batch.stop_practice_cli(process, path)
    assert commands == [("quit", 5), ("wait", 5)]


def test_episode_releases_owned_model_when_program_raises(monkeypatch, tmp_path):
    from perception import config
    from sim import harness

    segmenter = FakeSegmenter()
    scene = SimpleNamespace(
        nominal_objects=lambda: {}, nominal_cameras=lambda: {},
        object_xyz=lambda name: [0, 0, 0], closed=False,
    )
    scene.close = lambda: setattr(scene, "closed", True)
    episode = SimpleNamespace(reset=lambda *args: None)

    def fail(*args, **kwargs):
        raise RuntimeError("program failed")

    monkeypatch.setattr(harness, "PiperTabletopScene", lambda *args, **kwargs: scene)
    monkeypatch.setattr(harness, "SimEpisode", lambda *args: episode)
    monkeypatch.setattr(harness, "run_program", fail)
    monkeypatch.setattr(config, "create_instance_segmenter", lambda *args: segmenter)
    scene_path = tmp_path / "scene.xml"
    scene_path.write_text("<mujoco/>")
    task = harness.TabletopTask("block", "container", "move block")
    with pytest.raises(RuntimeError, match="program failed"):
        harness.run_episode(scene_path, task, "def run(api): pass", "randomized",
                            17, tmp_path / "output")
    assert segmenter.closed == 1
    assert scene.closed
