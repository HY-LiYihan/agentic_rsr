"""Render a third-person observation or the four Stage 2 reference camera views."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from real2sim.stage3.physics_probe import _robot_body, load_scene
from real2sim.stage3.mujoco_export import CAMERA_NAMES


def render_overview(scene: Path, output: Path, settle: float = 0.1,
                    width: int = 640, height: int = 480,
                    attach_piper: str = "auto", camera_name: str | None = None) -> Path:
    import mujoco

    if not np.isfinite(settle) or settle < 0 or width < 1 or height < 1:
        raise ValueError("settle and image dimensions must be finite and nonnegative")
    handle = load_scene(scene, attach_piper=attach_piper)
    try:
        model = handle.model
        data = handle.data if handle.data is not None else mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for _ in range(int(round(settle / model.opt.timestep))):
            mujoco.mj_step(model, data)
        if camera_name is None:
            ids = [index for index in range(1, model.nbody)
                   if not _robot_body(model, index) and model.body_geomnum[index] > 0]
            if not ids:
                raise ValueError("scene has no task objects to frame")
            positions = data.xpos[ids]
            center = np.mean(positions, axis=0)
            center[2] = max(0.2, float(center[2]))
            span = np.ptp(positions, axis=0)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = center
            camera.distance = max(1.1, float(np.linalg.norm(span)) * 2.5)
            camera.azimuth = 135
            camera.elevation = -35
        else:
            if camera_name not in CAMERA_NAMES or model.camera(camera_name).id < 0:
                raise ValueError(f"missing Stage 2 reference camera: {camera_name}")
            camera = camera_name
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name
            if name.endswith("_collision"):
                model.geom_group[geom_id] = 3
        preview = mujoco.MjvOption()
        preview.geomgroup[3] = 0
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            renderer.update_scene(data, camera=camera, scene_option=preview)
            rgb = renderer.render().copy()
        finally:
            renderer.close()
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not write overview image: {output}")
        return output
    finally:
        handle.close()


def render_reference_views(scene: Path, output: Path, settle: float = 0.0,
                           attach_piper: str = "auto") -> list[Path]:
    handle = load_scene(scene, attach_piper=attach_piper)
    try:
        dimensions = [(int(handle.model.cam_resolution[handle.model.camera(name).id][0]),
                       int(handle.model.cam_resolution[handle.model.camera(name).id][1]))
                      for name in CAMERA_NAMES]
    finally:
        handle.close()
    if any(width <= 0 or height <= 0 for width, height in dimensions):
        raise ValueError("all reference cameras need a positive resolution")
    return [render_overview(scene, output / f"camera_{index}_rgb.png", settle, width, height,
                            attach_piper, name)
            for index, (name, (width, height)) in enumerate(zip(CAMERA_NAMES, dimensions))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--settle", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--attach-piper", choices=("auto", "never", "always"), default="auto")
    parser.add_argument("--reference-views", action="store_true",
                        help="write camera_0..3_rgb.png to --output directory using Stage 2 cameras")
    args = parser.parse_args(argv)
    if args.reference_views:
        for path in render_reference_views(args.scene, args.output, 0.0, args.attach_piper):
            print(path)
    else:
        print(render_overview(args.scene, args.output, args.settle, args.width, args.height,
                              args.attach_piper))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
