"""Private task checks; never expose scene internals to a policy."""

from __future__ import annotations

import numpy as np


def verify_task(scene, task: str) -> str | None:
    """Return passed/failed for supported goals, or None if human review is required."""
    if task != "put_emergency_stop_button_in_green_storage_box":
        return None
    model, data = scene.model, scene.data
    object_name = "Stage2_EStopHousing"
    floor_name = "Stage2_TrayFloor"
    if object_name not in scene.free_joints or floor_name not in scene.object_bodies:
        return "failed"
    if data.eq_active[model.eq(f"sim_grasp_{object_name}").id]:
        return "failed"
    scene.mujoco.mj_forward(model, data)
    floor_id = scene.object_bodies[floor_name]
    object_id = scene.object_bodies[object_name]
    geom_id = model.geom(f"{floor_name}_collision").id
    local = data.xmat[floor_id].reshape(3, 3).T @ (
        data.xpos[object_id] - data.geom_xpos[geom_id])
    half_extent = model.geom_size[geom_id]
    # Keep the object's centre clear of the rim and above the floor, not floating high.
    inside = (np.all(np.abs(local[:2]) < half_extent[:2] - 0.02)
              and half_extent[2] < local[2] < half_extent[2] + 0.07)
    return "passed" if inside else "failed"
