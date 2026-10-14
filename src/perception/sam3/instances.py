"""Shared SAM3 instance contract and SIM selection policy."""

from pathlib import Path

import numpy as np


def validate_instances(masks, scores, expected_shape: tuple[int, int]) -> dict:
    if (masks is None or scores is None or np.ndim(masks) != 3
            or np.shape(masks)[1:] != tuple(expected_shape)
            or np.shape(scores) != (len(masks),)):
        raise ValueError("SAM3 returned invalid instance masks or scores")
    masks = np.asarray(masks)
    scores = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(masks).all() or not np.isfinite(scores).all():
        raise ValueError("SAM3 returned non-finite instance masks or scores")
    return {"masks": masks > 0, "scores": scores}


def select_instance(instances: dict, prompt: str) -> np.ndarray:
    masks, scores = instances["masks"], instances["scores"]
    instances = validate_instances(masks, scores, masks.shape[1:])
    masks, scores = instances["masks"], instances["scores"]
    if not len(masks):
        return np.zeros(masks.shape[1:], dtype=bool)
    candidates = np.flatnonzero(scores >= np.max(scores) - 0.05)
    areas = np.count_nonzero(masks[candidates], axis=(1, 2))
    selected = np.argmin(areas) if prompt.endswith(" block") else np.argmax(areas)
    return masks[int(candidates[selected])]


class InstanceSegmenter:
    """Adapt either backend's instances to the established SIM mask interface."""

    def __init__(self, backend):
        self.backend = backend

    def segment(self, image: Path, prompt: str, expected_shape: tuple[int, int]) -> np.ndarray:
        instances = self.backend.segment_instances(image, prompt, expected_shape)
        return select_instance(instances, prompt)

    @property
    def metadata(self) -> dict:
        return getattr(self.backend, "metadata", {})

    def close(self):
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()
