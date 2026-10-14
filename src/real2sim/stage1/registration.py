"""Conservative automatic arm registration with held-out visual checks."""

import itertools

import cv2
import numpy as np


REFERENCE = (0, 10, 21, 31)


def camera_world(extrinsic: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3] = extrinsic
    return np.linalg.inv(matrix)


def unproject(depth: np.ndarray, intrinsic: np.ndarray, mask: np.ndarray, stride: int = 5) -> np.ndarray:
    vertical, horizontal = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    values = depth[::stride, ::stride]
    selected = mask[::stride, ::stride] & np.isfinite(values) & (values > 0) & (values < 5)
    pixels_x, pixels_y, distance = horizontal[selected], vertical[selected], values[selected]
    return np.column_stack(((pixels_x - intrinsic[0, 2]) / intrinsic[0, 0] * distance,
                            (pixels_y - intrinsic[1, 2]) / intrinsic[1, 1] * distance, distance))


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    left, _, right = np.linalg.svd((source - source_center).T @ (target - target_center))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right)
    rotation = left @ correction @ right
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.T
    matrix[:3, 3] = target_center - source_center @ rotation
    return matrix


def nearest(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = cv2.flann_Index(np.ascontiguousarray(target, dtype=np.float32), {"algorithm": 1, "trees": 4})
    matches, distances = index.knnSearch(np.ascontiguousarray(source, dtype=np.float32), 1, params={"checks": 64})
    return matches[:, 0], np.sqrt(distances[:, 0])


def fit_pose(model: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, float]:
    model_center = np.median(model, axis=0)
    target_center = np.median(observed, axis=0)
    _, source_basis = np.linalg.eigh(np.cov(model.T))
    _, target_basis = np.linalg.eigh(np.cov(observed.T))
    best = None
    for signs in itertools.product((-1, 1), repeat=3):
        rotation = target_basis @ np.diag(signs) @ source_basis.T
        if np.linalg.det(rotation) < 0:
            continue
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, target_center - rotation @ model_center
        for _ in range(18):
            predicted = model @ pose[:3, :3].T + pose[:3, 3]
            match, distance = nearest(predicted, observed)
            keep = distance <= np.quantile(distance, 0.6)
            if keep.sum() < 30:
                break
            adjustment = rigid_fit(predicted[keep], observed[match[keep]])
            pose = adjustment @ pose
        match, distance = nearest(model @ pose[:3, :3].T + pose[:3, 3], observed)
        score = float(np.median(distance))
        if best is None or score < best[1]:
            best = pose, score
    if best is None:
        raise ValueError("unable to fit a rigid Piper pose")
    return best


def score_view(model: np.ndarray, pose: np.ndarray, intrinsic: np.ndarray,
               extrinsic: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> dict:
    world = model @ pose[:3, :3].T + pose[:3, 3]
    camera = world @ extrinsic[:, :3].T + extrinsic[:, 3]
    visible = camera[:, 2] > 0
    camera = camera[visible]
    horizontal = np.rint(camera[:, 0] * intrinsic[0, 0] / camera[:, 2] + intrinsic[0, 2]).astype(int)
    vertical = np.rint(camera[:, 1] * intrinsic[1, 1] / camera[:, 2] + intrinsic[1, 2]).astype(int)
    inside = (horizontal >= 0) & (horizontal < depth.shape[1]) & (vertical >= 0) & (vertical < depth.shape[0])
    projected = np.full(depth.shape, np.inf, dtype=np.float32)
    np.minimum.at(projected, (vertical[inside], horizontal[inside]), camera[inside, 2])
    footprint = cv2.dilate(np.isfinite(projected).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    predicted = cv2.erode(projected, np.ones((7, 7), np.uint8))
    intersection = int((footprint & mask).sum())
    union = int((footprint | mask).sum())
    valid = footprint & mask & np.isfinite(depth) & (depth > 0)
    residual = np.abs(predicted[valid] - depth[valid])
    return {"mask_iou": intersection / union if union else 0.0,
            "coverage": intersection / int(mask.sum()) if mask.any() else 0.0,
            "median_depth_residual_m": float(np.median(residual)) if len(residual) else None,
            "valid_pixels": int(valid.sum()), "projection": footprint}


def register(model: np.ndarray, depths: np.ndarray, intrinsics: np.ndarray,
             extrinsics: np.ndarray, masks: list[np.ndarray],
             thresholds: dict | None = None) -> tuple[np.ndarray, dict, list[np.ndarray]]:
    cloud = []
    for index in range(32):
        if index in REFERENCE:
            continue
        points = unproject(depths[index], intrinsics[index], masks[index])
        if len(points) >= 30:
            transform = camera_world(extrinsics[index])
            cloud.append(points @ transform[:3, :3].T + transform[:3, 3])
    if len(cloud) < 20:
        raise ValueError("fewer than 20 training frames contain valid robot observations")
    observed = np.concatenate(cloud).astype(np.float32)
    if len(observed) > 60000:
        observed = observed[np.linspace(0, len(observed) - 1, 60000, dtype=int)]
    pose, fit_score = fit_pose(model, observed)
    evaluations = [score_view(model, pose, intrinsics[index], extrinsics[index],
                              depths[index], masks[index]) for index in REFERENCE]
    thresholds = thresholds or {"mask_iou_min": 0.35, "coverage_min": 0.55,
                                "median_depth_residual_m_max": 0.05, "valid_pixels_min": 100}
    passed = all(result["mask_iou"] >= thresholds["mask_iou_min"] and
                 result["coverage"] >= thresholds["coverage_min"] and
                 result["valid_pixels"] >= thresholds["valid_pixels_min"] and
                 result["median_depth_residual_m"] is not None and
                 result["median_depth_residual_m"] <= thresholds["median_depth_residual_m_max"]
                 for result in evaluations)
    projections = [result.pop("projection") for result in evaluations]
    report = {"accepted": passed, "status": "passed" if passed else "failed",
              "thresholds": thresholds, "fit_median_nearest_m": fit_score,
              "held_out_frames": list(REFERENCE), "evaluations": evaluations,
              "world_from_base": pose.tolist()}
    return pose, report, projections
