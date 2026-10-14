"""Wrist RGB-D observations with the configured SAM3 backend."""

from __future__ import annotations

import base64
from pathlib import Path
import re

import numpy as np
from PIL import Image

def visual_prompt(name: str) -> str:
    label = re.sub(r"^Stage\d+_", "", name)
    label = re.sub(r"(_compound|_visual)$", "", label)
    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", label)
    return label.replace("_", " ").lower().replace("estop", "emergency stop")


class Stage3RGBDObservation:
    def __init__(self, scene, names: tuple[str, ...], image_dir: Path,
                 host: str | None = None, port: int | None = None, segmenter=None):
        self.scene = scene
        self.names = names
        self.image_dir = image_dir
        self._owns_segmenter = segmenter is None
        if segmenter is None:
            from perception.config import create_instance_segmenter
            segmenter = create_instance_segmenter(host, port)
        self.segmenter = segmenter
        self.calls = 0

    @property
    def metadata(self) -> dict:
        return getattr(self.segmenter, "metadata", {})

    def close(self) -> None:
        if self._owns_segmenter:
            self.segmenter.close()
            self._owns_segmenter = False

    @staticmethod
    def _image_payload(image: Path, array: np.ndarray) -> dict:
        Image.fromarray(array).save(image, format="JPEG", quality=95)
        return {
            "path": str(image.resolve()),
            "mime_type": "image/jpeg",
            "rgb_base64": base64.b64encode(image.read_bytes()).decode("ascii"),
            "width": int(array.shape[1]),
            "height": int(array.shape[0]),
        }

    def observe(self, queries=None) -> dict:
        if queries is None:
            selected = {name: visual_prompt(name) for name in self.names}
        elif isinstance(queries, dict):
            selected = queries
        elif isinstance(queries, (list, tuple)):
            selected = {name: visual_prompt(name) for name in queries}
        else:
            raise ValueError("observe queries must be a name list or name-to-prompt mapping")
        if not selected or len(selected) > 32:
            raise ValueError("observe requires 1..32 named objects")
        for name, prompt in selected.items():
            if name not in self.names or not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Unknown object or empty SAM3 prompt: {name}")

        frame = self.scene.read_rgbd()
        self.image_dir.mkdir(parents=True, exist_ok=True)
        stem = f"observation_{self.calls:03d}"
        image = self.image_dir / f"{stem}_wrist.jpg"
        third_person_image = self.image_dir / f"{stem}_third_person.jpg"
        wrist_payload = self._image_payload(image, frame.color)
        third_person = self.scene.render_camera("stage3_camera_0", width=512, height=288)
        third_person_payload = self._image_payload(third_person_image, third_person)
        self.calls += 1
        depth = np.asarray(frame.depth, dtype=float) * frame.depth_scale
        camera_pose = self.scene.camera_pose()
        rotation = np.empty(9)
        self.scene.mujoco.mju_quat2Mat(rotation, np.asarray(camera_pose["quaternion_wxyz"]))
        camera_position = np.asarray(camera_pose["xyz_m"])
        objects = {}
        for name, prompt in selected.items():
            mask = self.segmenter.segment(image, prompt, depth.shape)
            if mask.shape != depth.shape:
                raise ValueError(f"SAM3 mask for {name} does not match RGB-D frame")
            rows, columns = np.nonzero(mask & np.isfinite(depth) & (depth > 0.05)
                                       & (depth < 3.0))
            if len(rows) < 10:
                continue
            distances = depth[rows, columns]
            camera_points = np.stack(
                ((columns - frame.intrinsics.cx) * distances / frame.intrinsics.fx,
                 (rows - frame.intrinsics.cy) * distances / frame.intrinsics.fy,
                 distances), axis=1)
            world_points = camera_points @ rotation.reshape(3, 3).T + camera_position
            xyz = np.median(world_points, axis=0)
            objects[name] = {
                "xyz_m": [float(value) for value in xyz],
                "xy_m": [float(value) for value in xyz[:2]],
                "bbox_pixels": [int(columns.min()), int(rows.min()),
                                int(columns.max()) + 1, int(rows.max()) + 1],
            }
        return {
            "objects": objects,
            "camera": {"wrist": "d435i_color_optical_frame",
                       "third_person": "stage3_camera_0"},
            "images": {"wrist": wrist_payload, "third_person": third_person_payload},
            "image": wrist_payload,
            "third_person_image": third_person_payload,
        }
