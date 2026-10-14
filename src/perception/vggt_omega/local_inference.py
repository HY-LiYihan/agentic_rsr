"""Lazy checkpoint-backed local VGGT-Ω inference."""

from pathlib import Path

import numpy as np
from PIL import Image

from perception.config import local_metadata, require_device


def aligned_rgb(frames: list[Path]) -> np.ndarray:
    if len(frames) != 32:
        raise ValueError("VGGT requires 32 ordered frames")
    images = []
    for frame in frames:
        with Image.open(frame) as image:
            if image.size != (512, 288):
                raise ValueError(f"VGGT input must be aligned 512x288 RGB: {frame}")
            images.append(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)
    return np.stack(images).transpose(0, 3, 1, 2).copy()


def validate_outputs(arrays: dict) -> dict:
    shapes = {"depth": (32, 288, 512), "depth_conf": (32, 288, 512),
              "intrinsics": (32, 3, 3), "extrinsics": (32, 3, 4)}
    for name, shape in shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError(f"VGGT output {name} must have shape {shape}; no resizing is permitted")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"VGGT output {name} contains nonfinite values")
    intrinsics = arrays["intrinsics"]
    if (np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0)
            or not np.allclose(intrinsics[:, 2], [0, 0, 1], atol=1e-3)):
        raise ValueError("VGGT returned invalid camera intrinsics")
    rotations = arrays["extrinsics"][:, :, :3]
    if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-2)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-2)):
        raise ValueError("VGGT returned invalid world-to-camera rotations")
    return arrays


class LocalVGGT:
    def __init__(self, model_dir: Path, device: str = "cuda"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.model = None
        self.metadata = {"backend": "local", "model": "vggt", "device": device}

    def load(self):
        if self.model is not None:
            return
        checkpoint = self.model_dir / "vggt_omega_1b_512.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"local VGGT checkpoint missing: {checkpoint}")
        try:
            import torch
            from vggt_omega.models import VGGTOmega
        except ImportError as exc:
            raise RuntimeError("install torch and the official vggt-omega package for local inference") from exc
        selected = require_device(torch, self.device)
        try:
            model = VGGTOmega().eval()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            self.model = model.to(selected)
            self.metadata = local_metadata("vggt", checkpoint, self.device)
        except torch.cuda.OutOfMemoryError as exc:
            self.close()
            raise RuntimeError(f"VGGT model loading exhausted memory on {self.device}; "
                               "use a GPU with more free memory") from exc
        except Exception:
            self.close()
            raise

    def infer(self, frames: list[Path]) -> dict[str, np.ndarray]:
        checkpoint = self.model_dir / "vggt_omega_1b_512.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"local VGGT checkpoint missing: {checkpoint}")
        pixels = aligned_rgb(frames)
        self.load()
        import torch
        from vggt_omega.utils.pose_enc import encoding_to_camera

        try:
            images = torch.from_numpy(pixels).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                predictions = self.model(images)
                extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], (288, 512))

            def numpy_array(value, depth=False):
                value = value.detach().float().cpu().numpy()
                if depth and value.ndim == 5 and value.shape[-1] == 1:
                    value = value[..., 0]
                return np.asarray(value[0], dtype=np.float32)

            return validate_outputs({
                "depth": numpy_array(predictions["depth"], depth=True),
                "depth_conf": numpy_array(predictions["depth_conf"], depth=True),
                "extrinsics": numpy_array(extrinsics), "intrinsics": numpy_array(intrinsics),
            })
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("VGGT CUDA memory exhausted for 32 aligned 512x288 frames; "
                               "use a GPU with more free memory") from exc

    def close(self):
        self.model = None
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
