"""Multiscale SO(3) search with translation and scale readjustment."""

from __future__ import annotations

import math

import cv2
import numpy as np


ROTATION_STEPS_DEGREES = (15., 8., 4., 2.)
MIN_LOSS_IMPROVEMENT = 1e-4


def adjust_translation_scale(pose: tuple, center: np.ndarray, radius: float,
                             bounds: tuple[float, float], score) -> tuple[tuple, float, dict]:
    best_loss, best_metrics = score(pose)
    for _ in range(2):
        improved = False
        for axis in range(4):
            for direction in (-1, 1):
                rotation, translation, scale = pose
                if axis < 3:
                    offset = np.zeros(3)
                    offset[axis] = direction * .0125 * radius
                    candidate = (rotation, translation + offset, scale)
                else:
                    next_scale = scale * math.exp(direction * .025)
                    if not bounds[0] <= next_scale <= bounds[1]:
                        continue
                    next_translation = translation + (scale - next_scale) * rotation @ center
                    candidate = (rotation, next_translation, next_scale)
                loss, metrics = score(candidate)
                if loss < best_loss - MIN_LOSS_IMPROVEMENT:
                    pose, best_loss, best_metrics = candidate, loss, metrics
                    improved = True
        if not improved:
            break
    return pose, best_loss, best_metrics


def refine_rotation(pose: tuple, center: np.ndarray, radius: float,
                    bounds: tuple[float, float], score,
                    steps: tuple[float, ...] = ROTATION_STEPS_DEGREES,
                    initial_score: tuple[float, dict] | None = None) -> tuple[tuple, float, dict, list[dict]]:
    best_loss, best_metrics = initial_score if initial_score is not None else score(pose)
    history = []
    for degrees in steps:
        changes = 0
        while True:
            best_neighbor = None
            for axis in range(3):
                for direction in (-1, 1):
                    offset = np.zeros(3)
                    offset[axis] = direction * math.radians(degrees)
                    rotation = cv2.Rodrigues(offset)[0] @ pose[0]
                    translation = pose[1] + pose[2] * (pose[0] - rotation) @ center
                    candidate = (rotation, translation, pose[2])
                    candidate, loss, metrics = adjust_translation_scale(
                        candidate, center, radius, bounds, score)
                    if loss < best_loss - MIN_LOSS_IMPROVEMENT \
                            and (best_neighbor is None or loss < best_neighbor[1]):
                        best_neighbor = (candidate, loss, metrics)
            if best_neighbor is None:
                break
            pose, best_loss, best_metrics = best_neighbor
            changes += 1
            if changes >= 30:
                raise RuntimeError(f"rotation level {degrees}° did not converge after 30 improvements")
        history.append({"rotation_step_degrees": degrees, "accepted_moves": changes,
                        "loss_after_level": best_loss})
    return pose, best_loss, best_metrics, history
