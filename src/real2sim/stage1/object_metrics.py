"""Multi-view geometry metrics for Stage 1 task objects."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from real2sim.blender import resolve_blender
ROOT = Path(__file__).resolve().parents[3]
REFERENCE = (0, 10, 21, 31)
F1_THRESHOLD_M = 0.01
DEFAULT_VOXEL_SIZE_M = 0.005
DEFAULT_PIXEL_STRIDE = 2
MIN_OBJECT_POINTS = 30


class Segmenter(Protocol):
    def segment(self, image: Path, prompt: str) -> np.ndarray:
        ...


OPERABLE_OBJECT_PROMPTS: dict[str, tuple[tuple[str, str], ...]] = {
    "close_drawer": (("white_drawer_cabinet", "white storage cabinet with drawer"),),
    "put_corn_in_bucket": (("corn", "corn"), ("green_bucket", "green bucket")),
    "put_cup_in_box_and_close_box": (("cup", "cup"), ("storage_box", "storage box")),
    "stack_snacks": (("snack_carton_a", "snacks"), ("snack_carton_b", "snacks")),
    "stack_two_bowls_1": (("blue_bowl", "blue bowl"), ("green_bowl", "green bowl")),
    "store_fruit_in_bamboo_tray": (("red_fruit", "red apple"), ("green_fruit", "green fruit"),
                                   ("orange_fruit", "small orange"), ("bamboo_tray", "woven bamboo tray")),
    "store_fruit_on_plate": (("red_apple", "red apple"), ("green_lime", "green lime"),
                             ("orange_fruit", "small orange"), ("plate", "green plate")),
    "store_three_tools": (("blue_scoop", "blue scoop"), ("green_rake", "green rake"),
                          ("orange_shovel", "orange shovel"), ("white_storage_box", "white storage box")),
    "store_two_beanbags": (("bamboo_tray", "wicker tray"), ("red_beanbag", "red beanbag"),
                            ("green_beanbag", "blue beanbag")),
    "close_cup_lid": (("cup_lid", "white plastic lid"), ("cup_body", "white bottle")),
    "put_brown_block_in_white_and_green_storage_box": (("brown_block", "brown block"),
                                                         ("storage_box", "white and green storage box")),
    "put_emergency_stop_button_in_green_storage_box": (("emergency_stop_switch", "black and red switch"),
                                                        ("green_storage_box", "green storage box")),
    "put_glue_in_clear_cup": (("glue", "red and white glue bottle"), ("clear_cup", "clear cup")),
    "put_glue_in_pen_holder": (("glue", "red and white glue bottle"),
                               ("white_pen_holder", "white pen holder")),
    "stack_brown_blocks": (("brown_block_a", "brown blocks"), ("brown_block_b", "brown blocks")),
    "stack_cones": (("blue_cone", "blue cone"), ("pink_cone", "pink cone")),
    "stack_storage_boxes": (("tan_box", "tan storage box"), ("white_box", "white storage box")),
    "take_out_tissue": (("tissue", "white tissue sticking out"), ("tissue_box", "tissue box")),
}
SAME_APPEARANCE_INSTANCES = {
    "stack_snacks": ("snack_carton_a", "snack_carton_b"),
    "stack_brown_blocks": ("brown_block_a", "brown_block_b"),
}
TARGET_SCOPE: dict[str, str] = {
    "put_emergency_stop_button_in_green_storage_box": (
        "Entire emergency-stop switch: red button, yellow collar, black housing and label; "
        "separately evaluate the entire green storage box; exclude the robotic gripper."
    ),
}
BLEND_GROUP_PREFIXES: dict[str, dict[str, tuple[str, ...]]] = {
    "close_drawer": {"white_drawer_cabinet": ("Cabinet_", "UpperDrawer_", "Drawer_Groove_",
                                               "DrawerFrontInset", "Lower_Rib_", "Tray_RoundedRecess")},
    "put_corn_in_bucket": {"corn": ("Corn_",), "green_bucket": ("Bucket_",)},
    "put_cup_in_box_and_close_box": {"cup": ("Cup_Red",),
                                     "storage_box": ("Box_Floor", "Box_Wall_", "Box_Lid")},
    "stack_snacks": {"snack_carton_a": ("Snack carton A", "A "),
                     "snack_carton_b": ("Snack carton B", "B ")},
    "stack_two_bowls_1": {"blue_bowl": ("Bowl_Blue",), "green_bowl": ("Bowl_Green",)},
    "store_fruit_in_bamboo_tray": {
        "red_fruit": ("Red_Pink_Fruit", "Red_Fruit_Stem"),
        "green_fruit": ("Green_Fruit",), "orange_fruit": ("Orange_Fruit",),
        "bamboo_tray": ("Bamboo_Tray_",),
    },
    "store_fruit_on_plate": {
        "red_apple": ("Stage2_RedApple",), "green_lime": ("Stage2_GreenLime",),
        "orange_fruit": ("Stage2_OrangeFruit",), "plate": ("Stage2_GreenPlate",),
    },
    "store_three_tools": {
        "blue_scoop": ("Tool_Cyan_Scoop",), "green_rake": ("Tool_Lime_Rake",),
        "orange_shovel": ("Tool_Orange_Shovel",), "white_storage_box": ("StorageBox",),
    },
    "store_two_beanbags": {"bamboo_tray": ("Woven_Tray_",), "red_beanbag": ("Red_Beanbag",),
                            "green_beanbag": ("Cyan_Beanbag",)},
    "close_cup_lid": {"cup_lid": ("Lid",), "cup_body": ("Cup",)},
    "put_brown_block_in_white_and_green_storage_box": {
        "brown_block": ("Brown_Block",), "storage_box": ("StorageBox_",),
    },
    "put_emergency_stop_button_in_green_storage_box": {
        "emergency_stop_switch": ("Stage2_EStop",), "green_storage_box": ("Stage2_Tray",),
    },
    "put_glue_in_clear_cup": {"glue": ("GlueBottle_",), "clear_cup": ("ClearCup",)},
    "put_glue_in_pen_holder": {"glue": ("GlueContainer",), "white_pen_holder": ("PenHolder",)},
    "stack_brown_blocks": {"brown_block_a": ("BrownBlock_A",), "brown_block_b": ("BrownBlock_B",)},
    "stack_cones": {"blue_cone": ("Stage2_Blue_Cone",), "pink_cone": ("Stage2_Pink_Cone",)},
    "stack_storage_boxes": {
        "tan_box": ("Tan_Storage_Box_",), "white_box": ("White_Storage_Box_",),
    },
    "take_out_tissue": {"tissue": ("Exposed_White_Tissue",), "tissue_box": ("Tissue_Packet",)},
}


def match_blend_groups(task: str, mesh_names: list[str]) -> dict[str, list[str]]:
    rules = BLEND_GROUP_PREFIXES.get(task)
    if not rules or set(rules) != {name for name, _ in OPERABLE_OBJECT_PROMPTS[task]}:
        raise MetricsError(f"missing Stage 2 mesh mapping for {task}")
    groups = {label: sorted(name for name in mesh_names if name.startswith(prefixes))
              for label, prefixes in rules.items()}
    if any(not names for names in groups.values()):
        raise MetricsError(f"Stage 2 Blend missing target mesh(es) for {task}: {groups}")
    if sum(map(len, groups.values())) != len({name for names in groups.values() for name in names}):
        raise MetricsError(f"a Blend mesh belongs to multiple target groups in {task}")
    return groups


def _extract_model(blend: Path, output: Path, camera: dict, frames: tuple[int, ...],
                   stride: int, blender: Path | None) -> dict[str, list[str]]:
    blender = resolve_blender(blender)
    inventory_expression = ('import bpy,json; print("STAGE2_MESH_NAMES=" + '
                            'json.dumps([obj.name for obj in bpy.data.objects if obj.type == "MESH"]))')
    command = [str(blender), "-b", str(blend), "--python-exit-code", "1",
               "--python-expr", inventory_expression]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    marker = "STAGE2_MESH_NAMES="
    names_line = next((line for line in result.stdout.splitlines() if line.startswith(marker)), None)
    if result.returncode or names_line is None:
        raise MetricsError(f"cannot inspect Stage 2 Blend: {(result.stdout + result.stderr)[-2000:]}")
    groups = match_blend_groups(blend.parents[1].name, json.loads(names_line[len(marker):]))
    specification = {"output": str(output), "camera": camera, "frames": list(frames),
                     "reference_indices": list(REFERENCE), "scenes": [f"Reference_{REFERENCE.index(frame):02d}" for frame in frames],
                     "resolution": [512, 288], "stride": stride, "minimum_points": MIN_OBJECT_POINTS,
                     "groups": groups}
    spec_path = output / "model_render_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    previous_report = output.parent / "report.json"
    if (spec_path.is_file() and (output / "blend_inventory.json").is_file() and
            previous_report.is_file() and _json(previous_report).get("blend_sha256") == _sha256(blend) and
            _json(spec_path) == specification and
            all((output / label / f"predicted_mask_{frame:02d}.npy").is_file() and
                (output / label / f"predicted_points_{frame:02d}.npy").is_file()
                for label in groups for frame in frames)):
        return groups
    spec_path.write_text(json.dumps(specification, indent=2) + "\n", encoding="utf-8")
    worker = Path(__file__).with_name("blend_object_worker.py")
    result = subprocess.run([str(blender), "-b", str(blend), "--python-exit-code", "1",
                             "--python", str(worker), "--", str(spec_path)],
                            capture_output=True, text=True, timeout=1200)
    (output / "model_render.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode or not (output / "blend_inventory.json").is_file():
        raise MetricsError(f"model ray cast failed: {(result.stdout + result.stderr)[-2000:]}")
    return groups


class MetricsError(ValueError):
    """Raised when a scene cannot provide enough data for the metrics."""


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("_").lower()


def _as_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2 or value.shape != shape:
        raise MetricsError(f"mask shape {value.shape} does not match image/depth {shape}")
    return value.astype(bool, copy=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _world_points(depth: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray,
                  extrinsic: np.ndarray, stride: int, robot_mask: np.ndarray | None) -> np.ndarray:
    if depth.ndim != 2 or intrinsic.shape != (3, 3) or extrinsic.shape != (3, 4):
        raise MetricsError("invalid metric depth or camera shape")
    mask = _as_bool_mask(mask, depth.shape)
    if robot_mask is not None:
        mask = mask & ~cv2.dilate(robot_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    rows, cols = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    values = depth[::stride, ::stride]
    selected = mask[::stride, ::stride] & np.isfinite(values) & (values > 0.05) & (values < 5.0)
    if not selected.any():
        return np.empty((0, 3), dtype=np.float32)
    x = (cols[selected] - intrinsic[0, 2]) / intrinsic[0, 0] * values[selected]
    y = (rows[selected] - intrinsic[1, 2]) / intrinsic[1, 1] * values[selected]
    camera = np.column_stack((x, y, values[selected])).astype(np.float32)
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    return ((camera - translation) @ rotation).astype(np.float32)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 3).astype(np.float32)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)].astype(np.float32, copy=False)


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 0 or len(target) == 0:
        return np.full(len(source), np.inf, dtype=np.float32)
    index = cv2.flann_Index(np.ascontiguousarray(target, dtype=np.float32), {"algorithm": 1, "trees": 4})
    _, distances = index.knnSearch(np.ascontiguousarray(source, dtype=np.float32), 1,
                                   params={"checks": 256})
    return np.sqrt(np.maximum(distances[:, 0], 0)).astype(np.float32)


def point_cloud_metrics(predicted: np.ndarray, reference: np.ndarray,
                        threshold_m: float = F1_THRESHOLD_M) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float32).reshape(-1, 3)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1, 3)
    if len(predicted) == 0 or len(reference) == 0:
        raise MetricsError("both point clouds must contain at least one point")
    pred_to_ref = _nearest_distances(predicted, reference)
    ref_to_pred = _nearest_distances(reference, predicted)
    precision = float(np.mean(pred_to_ref <= threshold_m))
    recall = float(np.mean(ref_to_pred <= threshold_m))
    denominator = precision + recall
    f1 = 2 * precision * recall / denominator if denominator else 0.0
    return {
        "chamfer_distance_m": float((pred_to_ref.mean() + ref_to_pred.mean()) / 2),
        "f1_score": f1,
        "f1_threshold_m": threshold_m,
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
        "predicted_points": int(len(predicted)),
        "reference_points": int(len(reference)),
    }


def bbox_center(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        raise MetricsError("cannot compute a bounding box for an empty point cloud")
    return (points.min(axis=0) + points.max(axis=0)) / 2


def bbox_position_error(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(bbox_center(predicted) - bbox_center(reference)))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _load_source(scene: Path, frames: list[int]) -> tuple[list[Path], list[np.ndarray], dict[str, Any], list[np.ndarray]]:
    camera = _json(scene / "camera_metric.json")
    intrinsics = [np.asarray(value, dtype=np.float32) for value in camera["intrinsics"]]
    extrinsics = [np.asarray(value, dtype=np.float32) for value in camera["extrinsics"]]
    if len(intrinsics) != 32 or len(extrinsics) != 32:
        raise MetricsError(f"expected 32 metric cameras in {scene}")
    frame_paths = [scene / "frames" / f"{frame:02d}.jpg" for frame in frames]
    depths = [np.load(scene / "depth_metric" / f"depth_{frame:02d}.npy", allow_pickle=False)
              for frame in frames]
    robot_masks = [np.load(scene / "sam3_masks" / f"mask_{frame:02d}.npy", allow_pickle=False)
                   for frame in frames]
    for image, depth, mask in zip(frame_paths, depths, robot_masks):
        if not image.is_file() or depth.shape != (288, 512) or mask.shape != depth.shape:
            raise MetricsError(f"incomplete frame/depth/mask data for {image}")
    return frame_paths, depths, camera, robot_masks


def _cached_mask(path: Path, shape: tuple[int, int], segmenter: Segmenter | None,
                 image: Path, prompt: str, force: bool) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(".json")
    image_hash = _sha256(image)
    metadata = {"prompt": prompt, "image_sha256": image_hash}
    if path.is_file() and metadata_path.is_file() and not force:
        if _json(metadata_path) == metadata:
            return _as_bool_mask(np.load(path, allow_pickle=False), shape)
    if segmenter is None:
        raise MetricsError(f"missing cached SAM3 mask and no segmenter was supplied: {path}")
    mask = _as_bool_mask(segmenter.segment(image, prompt), shape)
    np.save(path, mask)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return mask


def _split_same_appearance(mask: np.ndarray, predicted: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count, components, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    candidates = sorted(range(1, count), key=lambda index: stats[index, cv2.CC_STAT_AREA], reverse=True)
    candidates = [index for index in candidates if stats[index, cv2.CC_STAT_AREA] >= 40][:len(predicted)]
    result = {label: np.zeros_like(mask, dtype=bool) for label in predicted}
    if not candidates:
        return result
    labels = list(predicted)
    centers = []
    for label in labels:
        rows, cols = np.nonzero(predicted[label])
        centers.append(np.array([cols.mean(), rows.mean()]) if len(rows) else None)
    if len(candidates) == 1:
        distances = [np.linalg.norm(centers[index] - centroids[candidates[0]])
                     if centers[index] is not None else float("inf") for index in range(len(labels))]
        result[labels[int(np.argmin(distances))]] = components == candidates[0]
    else:
        direct = sum(np.linalg.norm(centers[index] - centroids[candidates[index]])
                     if centers[index] is not None else 0 for index in range(2))
        reversed_order = sum(np.linalg.norm(centers[index] - centroids[candidates[1 - index]])
                             if centers[index] is not None else 0 for index in range(2))
        if reversed_order < direct:
            candidates.reverse()
        for label, candidate in zip(labels, candidates):
            result[label] = components == candidate
    return result


def evaluate_scene(scene: Path, output: Path, *, blend: Path | None = None,
                   segmenter: Segmenter | None = None,
                   frames: tuple[int, ...] = REFERENCE,
                   voxel_size: float = DEFAULT_VOXEL_SIZE_M,
                   pixel_stride: int = DEFAULT_PIXEL_STRIDE,
                   threshold_m: float = F1_THRESHOLD_M,
                   force_masks: bool = False, blender: Path | None = None) -> dict[str, Any]:
    scene = scene.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if voxel_size <= 0 or pixel_stride < 1 or threshold_m <= 0:
        raise MetricsError("voxel size, pixel stride, and F1 threshold must be positive")
    if not frames or len(set(frames)) != len(frames) or any(frame not in REFERENCE for frame in frames):
        raise MetricsError("only the four Stage 2 reference camera frames may be evaluated")
    task = scene.name
    object_prompts = OPERABLE_OBJECT_PROMPTS.get(task)
    if not object_prompts:
        raise MetricsError(f"no operable-object prompt configuration for task {task}")
    blend = (blend or scene.parent.parent / "stage2" / task / "geometry" / "final.blend").expanduser().resolve(strict=True)
    frame_paths, depths, camera, robot_masks = _load_source(scene, list(frames))
    intrinsics = [np.asarray(camera["intrinsics"][frame], dtype=np.float32) for frame in frames]
    extrinsics = [np.asarray(camera["extrinsics"][frame], dtype=np.float32) for frame in frames]
    scene_output = output / scene.parent.parent.name / task
    mask_root = scene_output / "masks"
    model_root = scene_output / "model"
    groups = _extract_model(blend, model_root, camera, frames, pixel_stride, blender)
    instance_masks: dict[tuple[str, int], np.ndarray] = {}
    if task in SAME_APPEARANCE_INSTANCES:
        labels = SAME_APPEARANCE_INSTANCES[task]
        prompt = dict(object_prompts)[labels[0]]
        for frame, image, depth in zip(frames, frame_paths, depths):
            source_path = mask_root / "_shared_instances" / f"mask_{frame:02d}.npy"
            union = _cached_mask(source_path, depth.shape, segmenter, image, prompt, force_masks)
            predicted = {label: np.load(model_root / label / f"predicted_mask_{frame:02d}.npy",
                                        allow_pickle=False) for label in labels}
            separated = _split_same_appearance(union, predicted)
            for label in labels:
                path = mask_root / label / f"mask_{frame:02d}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, separated[label])
                path.with_suffix(".json").write_text(json.dumps({
                    "prompt": prompt, "image_sha256": _sha256(image),
                    "derivation": "two_largest_connected_components_assigned_to_model_projection",
                    "source_mask": str(source_path),
                }, indent=2) + "\n", encoding="utf-8")
                instance_masks[label, frame] = separated[label]
    object_results: dict[str, Any] = {}
    for label, prompt in object_prompts:
        mask_counts: dict[str, int] = {}
        raw_mask_counts: dict[str, int] = {}
        view_metrics: dict[str, Any] = {}
        invalid_views: dict[str, str] = {}
        for frame, image, depth, intrinsic, extrinsic, robot_mask in zip(
                frames, frame_paths, depths, intrinsics, extrinsics, robot_masks):
            mask_path = mask_root / _safe_label(label) / f"mask_{frame:02d}.npy"
            mask = instance_masks.get((label, frame))
            if mask is None:
                mask = _cached_mask(mask_path, depth.shape, segmenter, image, prompt, force_masks)
            raw_mask = mask
            raw_mask_pixels = int(mask.sum())
            if task == "take_out_tissue" and label == "tissue_box":
                tissue = np.load(mask_root / "tissue" / f"mask_{frame:02d}.npy", allow_pickle=False)
                mask = mask & ~tissue
                np.save(mask_path.with_name(f"evaluation_mask_{frame:02d}.npy"), mask)
                cv2.imwrite(str(mask_path.with_name(f"evaluation_mask_{frame:02d}.png")),
                            mask.astype(np.uint8) * 255)
            cv2.imwrite(str(mask_path.with_suffix(".png")), raw_mask.astype(np.uint8) * 255)
            effective = mask & ~cv2.dilate(robot_mask.astype(np.uint8),
                                           np.ones((5, 5), np.uint8)).astype(bool)
            cv2.imwrite(str(mask_path.with_name(f"effective_mask_{frame:02d}.png")),
                        effective.astype(np.uint8) * 255)
            preview_path = mask_path.with_name(f"overlay_{frame:02d}.png")
            rgb = cv2.imread(str(image))
            if rgb is None:
                raise MetricsError(f"cannot decode RGB frame: {image}")
            rgb[mask] = (rgb[mask].astype(np.float32) * 0.4 +
                         np.array([0, 255, 0], dtype=np.float32) * 0.6).astype(np.uint8)
            if not cv2.imwrite(str(preview_path), rgb):
                raise OSError(f"cannot write SAM3 mask preview: {preview_path}")
            cloud = _voxel_downsample(_world_points(depth, mask, intrinsic, extrinsic,
                                                     pixel_stride, robot_mask), voxel_size)
            prediction = _voxel_downsample(np.load(model_root / label / f"predicted_points_{frame:02d}.npy",
                                                    allow_pickle=False), voxel_size)
            predicted_mask = np.load(model_root / label / f"predicted_mask_{frame:02d}.npy",
                                     allow_pickle=False)
            cv2.imwrite(str(model_root / label / f"predicted_mask_{frame:02d}.png"),
                        predicted_mask.astype(np.uint8) * 255)
            comparison = cv2.imread(str(image))
            reference_only = mask & ~predicted_mask
            model_only = predicted_mask & ~mask
            overlap = predicted_mask & mask
            comparison[reference_only] = (0, 255, 0)
            comparison[model_only] = (0, 0, 255)
            comparison[overlap] = (0, 255, 255)
            if not cv2.imwrite(str(model_root / label / f"comparison_{frame:02d}.png"), comparison):
                raise OSError(f"cannot write model/reference comparison for {label}, frame {frame}")
            if len(cloud) < MIN_OBJECT_POINTS or len(prediction) < MIN_OBJECT_POINTS:
                invalid_views[str(frame)] = (f"observation={len(cloud)} model={len(prediction)}; "
                                             f"each must have at least {MIN_OBJECT_POINTS} points")
            else:
                metrics = point_cloud_metrics(prediction, cloud, threshold_m)
                metrics["bbox_position_error_m"] = bbox_position_error(prediction, cloud)
                view_metrics[str(frame)] = metrics
            mask_counts[str(frame)] = int(mask.sum())
            if task == "take_out_tissue" and label == "tissue_box":
                raw_mask_counts[str(frame)] = raw_mask_pixels
        fields = ("chamfer_distance_m", "f1_score", "bbox_position_error_m")
        result = {
            "status": "measured_unverified" if view_metrics else "insufficient_model_or_observation",
            "valid_views": [int(frame) for frame in view_metrics],
            "invalid_views": invalid_views,
            "views": view_metrics,
            "mean": ({field: _mean([values[field] for values in view_metrics.values()])
                      for field in fields} if view_metrics else None),
            "blend_meshes": groups[label],
        }
        result["label"] = label
        result["instance_scope"] = ("model_assigned_same_appearance_instance"
                                    if task in SAME_APPEARANCE_INSTANCES else "single_object")
        result["prompt"] = prompt
        result["target_scope"] = TARGET_SCOPE.get(task, "Complete visible manipulable object(s), not an isolated accessory")
        result["mask_pixels"] = mask_counts
        if raw_mask_counts:
            result["raw_mask_pixels"] = raw_mask_counts
        result["artifacts"] = {"reference_masks": str(mask_root / label),
                               "model_masks_and_clouds": str(model_root / label)}
        result["voxel_size_m"] = voxel_size
        result["pixel_stride"] = pixel_stride
        object_results[label] = result
    review_flags = []
    for frame in frames:
        for position, (first, _) in enumerate(object_prompts):
            left_path = mask_root / first / f"mask_{frame:02d}.npy"
            left_evaluation_path = left_path.with_name(f"evaluation_mask_{frame:02d}.npy")
            left = np.load(left_evaluation_path if left_evaluation_path.is_file() else left_path,
                           allow_pickle=False)
            for second, _ in object_prompts[position + 1:]:
                right_path = mask_root / second / f"mask_{frame:02d}.npy"
                right_evaluation_path = right_path.with_name(f"evaluation_mask_{frame:02d}.npy")
                right = np.load(right_evaluation_path if right_evaluation_path.is_file() else right_path,
                                allow_pickle=False)
                smaller = min(int(left.sum()), int(right.sum()))
                overlap = int((left & right).sum())
                if smaller and overlap / smaller > 0.1:
                    left_count, right_count = int(left.sum()), int(right.sum())
                    excluded = ([first] if left_count > right_count * 1.1 else
                                [second] if right_count > left_count * 1.1 else
                                [first, second])
                    for label in excluded:
                        result = object_results[label]
                        result["views"].pop(str(frame), None)
                        result["invalid_views"][str(frame)] = "SAM3 mask overlaps another task object"
                    review_flags.append({"objects": [first, second], "frame": frame,
                                         "reason": "sam3_instance_masks_overlap",
                                         "excluded_objects": excluded,
                                         "overlap_pixels": overlap,
                                         "fraction_of_smaller_mask": overlap / smaller})
    fields = ("chamfer_distance_m", "f1_score", "bbox_position_error_m")
    for result in object_results.values():
        result["valid_views"] = [int(frame) for frame in result["views"]]
        result["mean"] = ({field: _mean([view[field] for view in result["views"].values()])
                           for field in fields} if result["views"] else None)
        result["status"] = "measured_unverified" if result["mean"] else "insufficient_model_or_observation"
    valid_objects = [result for result in object_results.values() if result["mean"]]
    scene_mean = {field: _mean([result["mean"][field] for result in valid_objects]) for field in fields}
    review_flags.extend(
        {"object": label, "reason": "large_model_vs_observation_bbox_center_difference",
         "max_view_bbox_position_error_m": max(view["bbox_position_error_m"] for view in result["views"].values())}
        for label, result in object_results.items() if result["views"] and
        max(view["bbox_position_error_m"] for view in result["views"].values()) > 0.2
    )
    report = {
        "schema": "stage1-vs-stage2-object-metrics-v2",
        "status": ("insufficient_object_observations" if not valid_objects else
                   "partially_measured_unverified" if len(valid_objects) != len(object_results) else
                   "measured_unverified"),
        "evaluation_mode": "stage2_blend_visible_mesh_vs_stage1_sam3_metric_depth",
        "reference_source": "SAM3 masks and robot-base metric depth (pseudo-reference; no ground-truth sensor)",
        "interpretation": "Per-view visible Stage 2 mesh surface vs SAM3-filtered Stage 1 metric-depth points. BBox position is the center of the visible point-cloud AABB. Scene is not modified; movement between frames, occlusion, and depth/segmentation errors affect results.",
        "robot": scene.parent.parent.name,
        "task": task,
        "source_scene": str(scene),
        "source_blend": str(blend),
        "blend_sha256": _sha256(blend),
        "frames": list(frames),
        "camera_world_frame": camera.get("world_frame"),
        "thresholds": {"f1_threshold_m": threshold_m, "voxel_size_m": voxel_size},
        "objects": object_results,
        "scene_mean": scene_mean,
        "review_flags": review_flags,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    scene_output.mkdir(parents=True, exist_ok=True)
    (scene_output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    return report


def _make_segmenter() -> Segmenter:
    from perception.config import create_sam3
    return create_sam3()


def discover_scenes(processed_root: Path) -> list[tuple[str, str, Path]]:
    scenes = []
    for robot in ("franka", "piper"):
        root = processed_root / robot / "stage1"
        for scene in sorted(path for path in root.iterdir() if path.is_dir()):
            scenes.append((robot, scene.name, scene))
    return scenes


def evaluate_all(processed_root: Path, output: Path, *, segmenter: Segmenter | None = None,
                 frames: tuple[int, ...] = REFERENCE, force_masks: bool = False,
                 write_sidecars: bool = True, blender: Path | None = None,
                 **kwargs: Any) -> dict[str, Any]:
    processed_root = processed_root.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if segmenter is None:
        segmenter = _make_segmenter()
    reports = []
    for robot, task, scene in discover_scenes(processed_root):
        report = evaluate_scene(scene, output, segmenter=segmenter, frames=frames,
                                force_masks=force_masks, blender=blender, **kwargs)
        reports.append(report)
        if write_sidecars:
            sidecar = scene / "object_metrics.json"
            sidecar.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    summary_rows = []
    object_rows = []
    for report in reports:
        row = {"robot": report["robot"], "task": report["task"], "status": report["status"],
               "object_count": len(report["objects"])}
        row.update(report["scene_mean"])
        summary_rows.append(row)
        for label, result in report["objects"].items():
            object_rows.append({"robot": report["robot"], "task": report["task"],
                                "object": label, "prompt": result["prompt"],
                                "status": result["status"], "valid_views": len(result["valid_views"]),
                                **(result.get("mean") or {})})
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({
        "schema": "stage1-vs-stage2-object-metrics-summary-v2",
        "scene_count": len(reports),
        "reports": summary_rows,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["robot", "task", "status", "object_count", "chamfer_distance_m", "f1_score",
              "bbox_position_error_m"]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output / "objects.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["robot", "task", "object", "prompt", "status",
                                                      "valid_views", "chamfer_distance_m", "f1_score",
                                                      "bbox_position_error_m"])
        writer.writeheader()
        writer.writerows(object_rows)
    return {"scene_count": len(reports), "reports": summary_rows}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare Stage 2 Blender object meshes with Stage 1 SAM3 and metric depth")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/stage1_vs_stage2_objects")
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE_M)
    parser.add_argument("--pixel-stride", type=int, default=DEFAULT_PIXEL_STRIDE)
    parser.add_argument("--f1-threshold", type=float, default=F1_THRESHOLD_M)
    parser.add_argument("--force-masks", action="store_true")
    parser.add_argument("--no-sidecars", action="store_true",
                        help="leave Stage 1 delivery folders unchanged")
    args = parser.parse_args()
    result = evaluate_all(args.processed, args.output, frames=tuple(REFERENCE),
                          voxel_size=args.voxel_size, pixel_stride=args.pixel_stride,
                          threshold_m=args.f1_threshold, force_masks=args.force_masks,
                          write_sidecars=not args.no_sidecars, blender=args.blender)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
