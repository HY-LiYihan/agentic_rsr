"""Render an RGB-colored, metric Piper point cloud from camera 0.

Run with: uv run --with matplotlib python scripts/visualize_stage1_arm_cloud.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "data/processed/piper/stage1/put_emergency_stop_button_in_green_storage_box"
DEFAULT_OUTPUT = ROOT / "runs/stage1_arm_cloud/put_emergency_stop_button_in_green_storage_box"


def load_cloud(scene: Path, frame: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cameras = json.loads((scene / "camera_metric.json").read_text(encoding="utf-8"))
    if (cameras.get("extrinsics_direction") != "world_to_camera"
            or cameras.get("world_frame") != "robot_base"
            or cameras.get("depth_unit") != "m"):
        raise ValueError("expected metric world-to-camera extrinsics in robot_base coordinates")
    intrinsic = np.asarray(cameras["intrinsics"][frame], dtype=float)
    extrinsic = np.asarray(cameras["extrinsics"][frame], dtype=float)
    depth = np.load(scene / "depth_metric" / f"depth_{frame:02d}.npy", allow_pickle=False)
    mask = np.load(scene / "sam3_masks" / f"mask_{frame:02d}.npy", allow_pickle=False)
    image = cv2.imread(str(scene / "frames" / f"{frame:02d}.jpg"))
    if image is None or depth.shape != mask.shape or depth.shape != image.shape[:2]:
        raise ValueError("RGB, metric depth and mask must have identical image dimensions")
    if intrinsic.shape != (3, 3) or extrinsic.shape != (3, 4):
        raise ValueError("invalid camera calibration dimensions")
    rotation, translation = extrinsic[:, :3], extrinsic[:, 3]
    if (not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3)
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
        raise ValueError("invalid camera intrinsics or rotation")
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
    vertical, horizontal = np.nonzero(valid)
    if len(horizontal) == 0:
        raise ValueError("no valid masked metric depth pixels")
    z = depth[vertical, horizontal].astype(float)
    camera_points = np.column_stack(((horizontal - intrinsic[0, 2]) * z / intrinsic[0, 0],
                                      (vertical - intrinsic[1, 2]) * z / intrinsic[1, 1], z))
    world_points = (camera_points - translation) @ rotation
    return world_points, image[vertical, horizontal, ::-1], intrinsic, extrinsic, image


def project(points: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    camera_points = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixel = np.column_stack((intrinsic[0, 0] * camera_points[:, 0] / camera_points[:, 2] + intrinsic[0, 2],
                                 intrinsic[1, 1] * camera_points[:, 1] / camera_points[:, 2] + intrinsic[1, 2]))
    pixel[camera_points[:, 2] <= 0] = np.nan
    return pixel


def fuse_cloud(scene: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clouds = []
    colors = []
    first_intrinsic = None
    first_extrinsic = None
    for frame in range(32):
        points, rgb, intrinsic, extrinsic, _ = load_cloud(scene, frame)
        clouds.append(points)
        colors.append(rgb)
        if frame == 0:
            first_intrinsic, first_extrinsic = intrinsic, extrinsic
    return np.concatenate(clouds), np.concatenate(colors), first_intrinsic, first_extrinsic


def camera_view(extrinsic: np.ndarray) -> tuple[float, float, float]:
    rotation = extrinsic[:, :3]
    eye_direction = -rotation.T[:, 2]
    elevation = np.degrees(np.arctan2(eye_direction[2], np.linalg.norm(eye_direction[:2])))
    azimuth = np.degrees(np.arctan2(eye_direction[1], eye_direction[0]))
    vertical = np.array([0., 0., 1.])
    screen_right = np.cross(vertical, eye_direction)
    if np.linalg.norm(screen_right) < 1e-8:
        vertical = np.array([0., 1., 0.])
        screen_right = np.cross(vertical, eye_direction)
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(eye_direction, screen_right)
    desired_up = -rotation.T[:, 1]
    roll = np.degrees(np.arctan2(desired_up @ screen_right, desired_up @ screen_up))
    return float(elevation), float(azimuth), float(roll)


def render(points: np.ndarray, colors: np.ndarray, extrinsic: np.ndarray, output: Path,
           bounds: tuple[np.ndarray, np.ndarray], *, max_points: int = 80000,
           show_axes: bool = True) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    low, high = bounds
    within = np.all((points >= low) & (points <= high), axis=1)
    selected = np.flatnonzero(within)
    if not len(selected):
        raise ValueError("no robot points in the URDF visualization bounds")
    if len(selected) > max_points:
        selected = np.sort(np.random.default_rng(2027).choice(selected, max_points, replace=False))

    figure = plt.figure(figsize=(9, 9), facecolor="white")
    axes = figure.add_subplot(111, projection="3d", facecolor="white")
    axes.scatter(points[selected, 0], points[selected, 1], points[selected, 2],
                 c=colors[selected].astype(float) / 255, s=1.0, alpha=.9,
                 depthshade=False, linewidths=0, rasterized=True)
    axes.set_xlim(low[0], high[0])
    axes.set_ylim(low[1], high[1])
    axes.set_zlim(low[2], high[2])
    axes.set_box_aspect(high - low, zoom=1.12)
    axes.view_init(*camera_view(extrinsic), vertical_axis="z")
    axes.set_proj_type("persp")
    if show_axes:
        axes.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
        for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
            axis.set_major_locator(MultipleLocator(.1))
            axis.pane.fill = False
            axis._axinfo["grid"].update(color=(.87, .87, .87, 1), linewidth=.5)
    else:
        axes.set_axis_off()
    figure.savefig(output, dpi=240, facecolor="white")
    plt.close(figure)
    image = cv2.imread(str(output))
    vertical, horizontal = np.nonzero(np.any(image < 250, axis=2))
    if len(horizontal):
        side = int(np.ceil(max(np.ptp(horizontal), np.ptp(vertical)) * 1.18))
        left = int(round((horizontal.min() + horizontal.max() - side) / 2))
        top = int(round((vertical.min() + vertical.max() - side) / 2))
        square = np.full((side, side, 3), 255, dtype=np.uint8)
        source_x = max(0, left)
        source_y = max(0, top)
        source_right = min(image.shape[1], left + side)
        source_bottom = min(image.shape[0], top + side)
        square[source_y - top:source_bottom - top, source_x - left:source_right - left] = image[
            source_y:source_bottom, source_x:source_right]
        if not cv2.imwrite(str(output), cv2.resize(square, (2160, 2160), interpolation=cv2.INTER_AREA)):
            raise OSError(f"cannot write {output}")
    return {"points_in_robot_region": int(within.sum()), "points_drawn": len(selected),
            "visualization_bounds_m": {"min": low.tolist(), "max": high.tolist()},
            "camera_view_elev_azim_roll_degrees": camera_view(extrinsic)}


def main() -> None:
    parser = argparse.ArgumentParser(description="32-frame RGB metric robot cloud from first camera")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    points, colors, _, extrinsic = fuse_cloud(arguments.scene)
    arguments.output.mkdir(parents=True, exist_ok=True)
    template = json.loads((ROOT / "asset/robot_surfaces/piper/metadata.json").read_text(encoding="utf-8"))
    margin = .1
    bounds = (np.asarray(template["bounds_min_m"]) - margin,
              np.asarray(template["bounds_max_m"]) + margin)
    figure_metadata = render(points, colors, extrinsic, arguments.output / "camera_00_metric.png", bounds)
    render(points, colors, extrinsic, arguments.output / "camera_00_cloud_only.png", bounds,
           show_axes=False)
    single_points, single_colors, _, _, _ = load_cloud(arguments.scene, 0)
    render(single_points, single_colors, extrinsic,
           arguments.output / "camera_00_only_metric.png", bounds)
    trimesh.points.PointCloud(points, colors=np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))).export(
        arguments.output / "arm_metric_robot_base.ply")
    metadata = {"scene": str(arguments.scene), "frames_fused": list(range(32)),
                "view_camera_frame": 0, "points": len(points),
                "unit": "m", "world_frame": "robot_base", "world_z": "up",
                "camera_z": "forward (OpenCV)", "extrinsics_direction": "world_to_camera",
                "camera_origin_robot_base_m": (-extrinsic[:, :3].T @ extrinsic[:, 3]).tolist(),
                "rgb_source": "frames/XX.jpg (original RGB)",
                "visualization_region": "URDF surface bounds expanded 0.10 m on each side; full cloud preserved in PLY",
                **figure_metadata,
                "artifacts": ["camera_00_metric.png", "camera_00_cloud_only.png",
                              "camera_00_only_metric.png",
                              "arm_metric_robot_base.ply"]}
    (arguments.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{arguments.output}: {len(points)} RGB metric points; {figure_metadata['points_drawn']} shown in white-background camera-00 view")


if __name__ == "__main__":
    main()
