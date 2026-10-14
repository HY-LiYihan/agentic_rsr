import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from perception.vggt_omega.local_inference import LocalVGGT, aligned_rgb, validate_outputs


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.value, axis))

    def to(self, device):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def outputs():
    return {"depth": np.ones((32, 288, 512), dtype=np.float32),
            "depth_conf": np.ones((32, 288, 512), dtype=np.float32),
            "intrinsics": np.tile([[400, 0, 256], [0, 400, 144], [0, 0, 1]], (32, 1, 1)).astype(np.float32),
            "extrinsics": np.tile(np.column_stack([np.eye(3), [0, 0, 0]]), (32, 1, 1)).astype(np.float32)}


@pytest.fixture
def frames(tmp_path):
    image = np.zeros((288, 512, 3), dtype=np.uint8)
    image[:, :16, 0] = 255
    image[:, -16:, 2] = 255
    path = tmp_path / "frame.png"
    Image.fromarray(image).save(path)
    return [path] * 32


def test_rgb_tensor_preserves_edges_and_pixel_grid(frames):
    tensor = aligned_rgb(frames)
    assert tensor.shape == (32, 3, 288, 512)
    assert tensor.dtype == np.float32
    assert tensor[:, 0, :, 0].min() == 1
    assert tensor[:, 2, :, -1].min() == 1


def test_wrong_shape_or_frame_count_is_rejected(frames, tmp_path):
    with pytest.raises(ValueError, match="32 ordered"):
        aligned_rgb(frames[:10])
    path = tmp_path / "wrong.png"
    Image.new("RGB", (512, 256)).save(path)
    with pytest.raises(ValueError, match="512x288"):
        aligned_rgb([path] * 32)


def test_camera_unprojection_projects_back_to_same_rgb_pixels():
    from real2sim.stage1.registration import camera_world, unproject

    arrays = validate_outputs(outputs())
    pixels = np.array([[0, 0], [256, 144], [511, 287]])
    intrinsic = arrays["intrinsics"][0]
    mask = np.zeros((288, 512), dtype=bool)
    mask[pixels[:, 1], pixels[:, 0]] = True
    camera = unproject(arrays["depth"][0], intrinsic, mask, stride=1)
    extrinsic = np.column_stack(([[0, -1, 0], [1, 0, 0], [0, 0, 1]], [0.3, -0.4, 0.7]))
    world_from_camera = camera_world(extrinsic)
    world = camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    camera_again = world @ extrinsic[:, :3].T + extrinsic[:, 3]
    projected = camera_again @ intrinsic.T
    np.testing.assert_allclose(projected[:, :2] / projected[:, 2:], pixels, atol=1e-9)


@pytest.mark.parametrize("field", ["depth", "depth_conf", "intrinsics", "extrinsics"])
def test_invalid_outputs_are_not_resized(field):
    arrays = outputs()
    arrays[field] = arrays[field][:1]
    with pytest.raises(ValueError, match="no resizing"):
        validate_outputs(arrays)


def test_nonfinite_and_invalid_camera_outputs_are_rejected():
    arrays = outputs()
    arrays["depth"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_outputs(arrays)
    arrays = outputs()
    arrays["extrinsics"][0, 0, 0] = -1
    with pytest.raises(ValueError, match="rotations"):
        validate_outputs(arrays)


def test_lazy_model_reuse_and_release_without_preprocessor(monkeypatch, tmp_path, frames):
    from contextlib import nullcontext
    import perception.vggt_omega.local_inference as local

    created, calls = [], []
    arrays = outputs()

    class Model:
        def __init__(self):
            created.append(self)

        def eval(self):
            return self

        def load_state_dict(self, weights):
            assert weights == "weights"

        def to(self, device):
            return self

        def __call__(self, tensor):
            calls.append(tensor.value)
            return {"depth": Tensor(arrays["depth"][None, ..., None]),
                    "depth_conf": Tensor(arrays["depth_conf"][None]), "pose_enc": None}

    def camera(poses, shape):
        assert shape == (288, 512)
        return Tensor(arrays["extrinsics"][None]), Tensor(arrays["intrinsics"][None])

    torch = SimpleNamespace(from_numpy=Tensor, load=lambda *args, **kwargs: "weights",
                            inference_mode=nullcontext,
                            cuda=SimpleNamespace(is_available=lambda: False, OutOfMemoryError=MemoryError))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vggt_omega", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vggt_omega.models", SimpleNamespace(VGGTOmega=Model))
    monkeypatch.setitem(sys.modules, "vggt_omega.utils", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vggt_omega.utils.pose_enc", SimpleNamespace(encoding_to_camera=camera))
    monkeypatch.setattr(local, "require_device", lambda module, device: device)
    monkeypatch.setattr(local, "local_metadata", lambda *args: {"backend": "local"})
    (tmp_path / "vggt_omega_1b_512.pt").write_bytes(b"weights")
    model = LocalVGGT(tmp_path, device="cpu")
    model.infer(frames)
    model.infer(frames)
    assert len(created) == 1 and len(calls) == 2
    assert calls[0].shape == (1, 32, 3, 288, 512)
    assert calls[0][0, :, 0, :, 0].min() == 1
    model.close()
    model.infer(frames)
    assert len(created) == 2
