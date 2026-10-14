"""Lazy local SAM3 text segmentation using a provided checkpoint."""

import gc
import os
from pathlib import Path

import numpy as np
from perception.sam3 import DEFAULT_PROMPT
from perception.sam3.instances import validate_instances


def _image_shape(image: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image) as source:
        return source.height, source.width


class LocalSAM3:
    def __init__(self, model_dir: Path, device: str = "cuda"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.predictor = None
        self.metadata = {"backend": "local", "model": "sam3", "device": device}
        self._image_key = None

    def _load(self):
        checkpoint = self.model_dir / "sam3.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"local SAM3 checkpoint missing: {checkpoint}")
        if self.predictor is not None:
            return
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_OFFLINE"] = "true"
        try:
            import torch
            from clip.simple_tokenizer import SimpleTokenizer
        except ImportError as exc:
            raise RuntimeError("install the local inference dependencies, including the pinned CLIP tokenizer") from exc
        # The upstream builder otherwise installs CLIP during inference. Also
        # verify the tokenizer's packaged BPE vocabulary before loading weights.
        try:
            SimpleTokenizer()
        except (OSError, ValueError) as exc:
            raise RuntimeError("the local CLIP tokenizer BPE vocabulary is missing or invalid") from exc
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
            from ultralytics.utils import checks
            import ultralytics.utils as utils
        except ImportError as exc:
            raise RuntimeError("install the local inference dependencies and pinned Ultralytics source") from exc
        utils.AUTOINSTALL = checks.AUTOINSTALL = False
        # These flags and the analytics singleton can predate this backend if
        # another component already imported Ultralytics in the same process.
        utils.ONLINE = checks.ONLINE = False
        from ultralytics.utils.events import events

        events.enabled = False
        from perception.config import local_metadata, require_device

        require_device(torch, self.device)
        metadata = local_metadata("sam3", checkpoint, self.device)
        predictor = SAM3SemanticPredictor(overrides={
            "conf": 0.25, "task": "segment", "mode": "predict", "model": str(checkpoint),
            "device": self.device, "save": False, "verbose": False,
        })
        try:
            predictor.setup_model(verbose=False)
        except Exception as exc:
            del predictor
            gc.collect()
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if "out of memory" in str(exc).lower():
                raise RuntimeError(f"SAM3 ran out of memory while loading on {self.device}; "
                                   "use a device with sufficient memory") from exc
            raise
        self.predictor = predictor
        self.metadata = metadata

    def load(self):
        """Load and validate the actual model without running a prediction."""
        self._load()
        return self

    def segment_instances(self, image: Path, prompt: str = DEFAULT_PROMPT,
                          expected_shape: tuple[int, int] | None = None) -> dict:
        self._load()
        image = Path(image)
        shape = _image_shape(image)
        if expected_shape is not None and tuple(expected_shape) != shape:
            raise ValueError(f"SAM3 image shape {shape} does not match expected shape {expected_shape}")
        stat = image.stat()
        key = (str(image.resolve()), stat.st_mtime_ns, stat.st_size)
        try:
            if self._image_key != key:
                self.predictor.reset_image()
                self.predictor.set_image(str(image))
                self._image_key = key
            results = self.predictor(text=[prompt])
        except RuntimeError as exc:
            self.close()
            if "out of memory" in str(exc).lower():
                raise RuntimeError(f"SAM3 ran out of memory on {self.device}; use a device with sufficient memory") from exc
            raise
        if not results or results[0].masks is None:
            return {"masks": np.zeros((0, *shape), dtype=bool), "scores": np.zeros(0, dtype=np.float32)}
        result = results[0]
        # Pinned SAM3 stretches the input (scale_fill=True) and its postprocess
        # restores logits to orig_shape. Resizing here would hide a model mismatch.
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        scores = None if boxes is None else boxes.conf.cpu().numpy()
        return validate_instances(masks, scores, shape)

    def segment(self, image: Path, prompt: str = DEFAULT_PROMPT) -> np.ndarray:
        return np.any(self.segment_instances(image, prompt)["masks"], axis=0)

    def close(self):
        predictor, self.predictor = self.predictor, None
        self._image_key = None
        if predictor is None:
            return
        predictor.reset_image()
        del predictor
        gc.collect()
        import torch

        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
