"""Wrist RGB-D plus the configured SAM3 backend, without truth-state queries."""

from __future__ import annotations

import json
from pathlib import Path
import socket

import numpy as np

from perception.wire import receive_arrays, send_frame


class ExactObservation:
    """Expose exact task-object positions through the public observation schema."""

    def __init__(self, scene, names: tuple[str, ...]):
        self.scene, self.names = scene, names

    def observe(self) -> dict:
        objects = {}
        for name in self.names:
            xyz = self.scene.object_xyz(name)
            objects[name] = {"xy_m": [xyz[0], xyz[1]], "xyz_m": list(xyz)}
        return {"objects": objects, "camera": "wrist", "image": None}


class SAM3InstanceSegmenter:
    """Stage 1's framed SAM3 TCP protocol, selecting one scored instance."""

    def __init__(self, host: str, port: int, timeout: float = 60):
        self.host, self.port, self.timeout = host, port, timeout

    def segment(self, image: Path, prompt: str, expected_shape: tuple[int, int]) -> np.ndarray:
        blob = image.read_bytes()
        header = {"type": "infer", "model": "sam3", "mode": "text",
                  "image_bytes": len(blob), "text": [prompt], "conf": 0.25}
        with socket.create_connection((self.host, self.port), self.timeout) as connection:
            connection.settimeout(self.timeout)
            send_frame(connection, json.dumps(header).encode())
            send_frame(connection, blob)
            arrays, _ = receive_arrays(connection, framed=True)
        masks, scores = arrays.get("masks"), arrays.get("scores")
        if (masks is None or scores is None or masks.ndim != 3
                or masks.shape[1:] != expected_shape or scores.shape != (len(masks),)):
            raise ValueError("SAM3 returned invalid instance masks or scores")
        if not len(masks):
            return np.zeros(expected_shape, dtype=bool)
        candidates = np.flatnonzero(scores >= np.max(scores) - 0.05)
        areas = np.count_nonzero(masks[candidates], axis=(1, 2))
        selected = np.argmin(areas) if prompt.endswith(" block") else np.argmax(areas)
        return masks[int(candidates[selected])].astype(bool)


class SAM3RGBDObservation:
    def __init__(self, scene, episode, names: tuple[str, ...], image_dir: Path,
                 host: str | None = None, port: int | None = None,
                 prompts: dict[str, str] | None = None,
                 fallback_prompts: dict[str, str] | None = None, segmenter=None):
        self.scene, self.episode, self.names = scene, episode, names
        self.image_dir = image_dir
        self._owns_segmenter = segmenter is None
        if segmenter is None:
            from perception.config import create_instance_segmenter
            segmenter = create_instance_segmenter(host, port)
        self.segmenter = segmenter
        self.prompts = prompts or {}
        self.fallback_prompts = fallback_prompts or {}
        self.calls = 0

    @property
    def metadata(self) -> dict:
        return getattr(self.segmenter, "metadata", {})

    def close(self) -> None:
        if self._owns_segmenter:
            self.segmenter.close()
            self._owns_segmenter = False

    def observe(self) -> dict:
        from PIL import Image

        frame = self.scene.read_rgbd()
        self.image_dir.mkdir(parents=True, exist_ok=True)
        image = self.image_dir / f"observation_{self.calls:03d}.jpg"
        Image.fromarray(frame.color).save(image, format="JPEG", quality=95)
        self.calls += 1
        transform = self.scene.estimated_world_from_camera(
            self.episode.context.estimated_camera_pose("wrist"))
        depth = np.asarray(frame.depth, dtype=float) * frame.depth_scale
        objects = {}
        for name in self.names:
            prompt = self.prompts.get(name, name.replace(" bowl", " container"))
            mask = self.segmenter.segment(image, prompt, depth.shape)
            if mask.shape != depth.shape:
                raise ValueError(f"SAM3 mask for {name} does not match RGB-D frame")
            rows, columns = np.nonzero(mask & np.isfinite(depth) & (depth > 0.05) & (depth < 3.0))
            if not len(rows) and name in self.fallback_prompts:
                mask = self.segmenter.segment(image, self.fallback_prompts[name], depth.shape)
                if mask.shape != depth.shape:
                    raise ValueError(f"SAM3 fallback mask for {name} does not match RGB-D frame")
                rows, columns = np.nonzero(mask & np.isfinite(depth) & (depth > 0.05) & (depth < 3.0))
            if not len(rows):
                continue
            distances = depth[rows, columns]
            points = np.stack(((columns - frame.intrinsics.cx) * distances / frame.intrinsics.fx,
                               (rows - frame.intrinsics.cy) * distances / frame.intrinsics.fy,
                               distances), axis=1)
            world_points = points @ transform[:3, :3].T + transform[:3, 3]
            xyz = np.median(world_points, axis=0)
            if name.endswith(" block"):
                percentile = 60 if np.median(columns) < frame.intrinsics.cx else 25
                xyz[:2] = np.percentile(world_points[:, :2], percentile, axis=0)
            objects[name] = {"xy_m": [float(xyz[0]), float(xyz[1])],
                             "xyz_m": [float(value) for value in xyz],
                             "bbox_pixels": [int(columns.min()), int(rows.min()),
                                             int(columns.max()) + 1, int(rows.max()) + 1]}
        return {"objects": objects, "camera": "wrist", "image": str(image)}
