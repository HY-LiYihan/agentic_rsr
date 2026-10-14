import numpy as np
import pytest

from perception.sam3.local_inference import LocalSAM3
from perception.vggt_omega.local_inference import LocalVGGT
from real2sim.stage1.pipeline import validate


def test_local_backends_fail_without_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        LocalVGGT(tmp_path).infer([tmp_path / "frame.jpg"] * 32)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        LocalSAM3(tmp_path).segment(tmp_path / "frame.jpg")


def test_swappable_backends_share_array_and_mask_contract(tmp_path):
    class FakeBackend:
        def infer(self, frames):
            assert len(frames) == 32
            return {"depth": np.ones((32, 288, 512), np.float32),
                    "depth_conf": np.ones((32, 288, 512), np.float32),
                    "intrinsics": np.tile(np.array([[400., 0, 256], [0, 400, 144], [0, 0, 1]]), (32, 1, 1)),
                    "extrinsics": np.tile(np.column_stack((np.eye(3), [0, 0, 0])), (32, 1, 1))}

        def segment(self, image, prompt="robotic arm with base and camera"):
            assert prompt == "robotic arm with base and camera"
            return np.zeros((288, 512), bool)

    for backend in (FakeBackend(), FakeBackend()):
        depth, intrinsics, extrinsics = validate(backend.infer([tmp_path / "frame.jpg"] * 32), 1.0)
        assert depth.shape == (32, 288, 512) and intrinsics.shape == (32, 3, 3)
        assert extrinsics.shape == (32, 3, 4)
        assert backend.segment(tmp_path / "frame.jpg").shape == (288, 512)
