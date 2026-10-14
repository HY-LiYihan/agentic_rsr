"""Blender-side, read-only privileged mesh ray caster for Stage 1 observations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _bounds(objects, intrinsic, extrinsic, width, height, margin):
    corners = np.asarray([list(obj.matrix_world @ Vector(corner)) for obj in objects
                          for corner in obj.bound_box], dtype=np.float64)
    if not len(corners):
        return None
    camera = corners @ extrinsic[:, :3].T + extrinsic[:, 3]
    if np.any(camera[:, 2] <= 0):
        return 0, width, 0, height
    pixel_x = camera[:, 0] / camera[:, 2] * intrinsic[0, 0] + intrinsic[0, 2]
    pixel_y = camera[:, 1] / camera[:, 2] * intrinsic[1, 1] + intrinsic[1, 2]
    return (max(0, int(np.floor(pixel_x.min())) - margin),
            min(width, int(np.ceil(pixel_x.max())) + margin + 1),
            max(0, int(np.floor(pixel_y.min())) - margin),
            min(height, int(np.ceil(pixel_y.max())) + margin + 1))


def export(config: dict) -> None:
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    camera = config["camera"]
    frames = config["frames"]
    stride = config["stride"]
    groups = config["groups"]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    inventory = {}
    for label, names in groups.items():
        objects = [bpy.data.objects.get(name) for name in names]
        if any(obj is None or obj.type != "MESH" for obj in objects):
            raise ValueError(f"missing mesh in {label}: {names}")
        inventory[label] = set(names)
    written = {}
    sampled_strides = {}
    for frame, scene_name in zip(frames, config["scenes"]):
        scene = bpy.data.scenes[scene_name]
        view_index = config["reference_indices"].index(frame)
        expected_camera = f"Reference_{view_index:02d}"
        if scene.camera is None or scene.camera.name != expected_camera:
            raise ValueError(f"camera mismatch: {scene_name} expected {expected_camera}")
        width, height = config["resolution"]
        if (scene.render.resolution_x, scene.render.resolution_y) != (width, height):
            raise ValueError(f"render resolution mismatch for {scene_name}")
        intrinsic = np.asarray(camera["intrinsics"][frame], dtype=np.float64)
        extrinsic = np.asarray(camera["extrinsics"][frame], dtype=np.float64)
        rotation, translation = extrinsic[:, :3], extrinsic[:, 3]
        origin = -rotation.T @ translation
        origin_vec = Vector(origin)
        for label, names in groups.items():
            region = _bounds([bpy.data.objects[name] for name in names], intrinsic,
                             extrinsic, width, height, margin=12)
            for sample_stride in (stride, 1) if stride > 1 else (stride,):
                mask = np.zeros((height, width), dtype=bool)
                points = []
                if region is not None:
                    xmin, xmax, ymin, ymax = region
                    for row in range(ymin, ymax, sample_stride):
                        for col in range(xmin, xmax, sample_stride):
                            camera_ray = np.array([(col - intrinsic[0, 2]) / intrinsic[0, 0],
                                                   (row - intrinsic[1, 2]) / intrinsic[1, 1], 1.0])
                            direction = rotation.T @ camera_ray
                            direction /= np.linalg.norm(direction)
                            hit, position, _, _, obj, _ = scene.ray_cast(
                                depsgraph, origin_vec, Vector(direction), distance=5.0)
                            if hit and obj.name in inventory[label]:
                                mask[row, col] = True
                                points.append(tuple(position))
                if len(points) >= config["minimum_points"]:
                    break
            directory = output / label
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / f"predicted_mask_{frame:02d}.npy", mask)
            np.save(directory / f"predicted_points_{frame:02d}.npy",
                    np.asarray(points, dtype=np.float32).reshape(-1, 3))
            written.setdefault(label, {})[str(frame)] = int(mask.sum())
            sampled_strides.setdefault(label, {})[str(frame)] = sample_stride
    (output / "blend_inventory.json").write_text(json.dumps({
        "scene": bpy.data.filepath, "groups": groups, "hit_pixels": written,
        "stride": stride, "sampled_strides": sampled_strides,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    separator = sys.argv.index("--")
    export(json.loads(Path(sys.argv[separator + 1]).read_text(encoding="utf-8")))
