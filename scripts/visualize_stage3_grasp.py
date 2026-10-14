#!/usr/bin/env python3
"""Create explainable Stage 3 workflow and grasp visualizations."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont

from real2sim.stage3.physics_probe import (
    _grasp_geometry,
    _rebase_scene_assets,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = ROOT / "data/processed/piper/stage3/put_emergency_stop_button_in_green_storage_box"
FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"
if Path(FONT_PATH).is_file():
    plt.rcParams["font.family"] = [FontProperties(fname=FONT_PATH).get_name()]
    plt.rcParams["axes.unicode_minus"] = False


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _read_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#111827")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _rounded_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str,
                  outline: str = "#334155", radius: int = 18) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=3)


def _workflow(task_dir: Path, output: Path) -> Path:
    stage2 = ROOT / "data/processed/piper/stage2/put_emergency_stop_button_in_green_storage_box"
    images = [
        (stage2 / "geometry/baseline_render/camera_0_rgb.png", "Stage 2 accepted RGB", "四相机视觉一致性输入"),
        (task_dir / "mujoco/default_views/camera_0_rgb.png", "MuJoCo scene", "外部 Piper + task geometry"),
        (task_dir / "final_views/camera_0_rgb.png", "Final visible scene", "碰撞体隐藏，视觉网格保留"),
    ]
    width, height = 1680, 980
    canvas = Image.new("RGB", (width, height), "#0b1120")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, True)
    section_font = _font(25, True)
    body_font = _font(21)
    small_font = _font(18)
    draw.text((60, 38), "Stage 3：急停开关场景的编译、物理探针与交付", font=title_font, fill="#f8fafc")
    draw.text((62, 94), "从 Stage 2 的可视化结果出发，分离 visual mesh / collision proxy，再用双夹块验证物理合理性。",
              font=body_font, fill="#cbd5e1")
    card_w, card_h = 430, 330
    positions = [(60, 170), (625, 170), (1190, 170)]
    for (image_path, title, subtitle), (x, y) in zip(images, positions):
        _rounded_card(draw, (x, y, x + card_w, y + card_h), "#111827")
        if image_path.is_file():
            image = _read_rgb(image_path, (card_w - 24, 230))
            canvas.paste(image, (x + 12, y + 12))
        draw.text((x + 18, y + 250), title, font=section_font, fill="#f8fafc")
        draw.text((x + 18, y + 287), subtitle, font=small_font, fill="#94a3b8")
    for x in (500, 1065):
        draw.line((x, 325, x + 95, 325), fill="#38bdf8", width=6)
        draw.polygon([(x + 95, 325), (x + 77, 314), (x + 77, 336)], fill="#38bdf8")

    probe_x, probe_y, probe_w, probe_h = 60, 575, 1560, 300
    _rounded_card(draw, (probe_x, probe_y, probe_x + probe_w, probe_y + probe_h), "#111827")
    draw.text((probe_x + 24, probe_y + 20), "Physical probe：two frictional box jaws", font=section_font, fill="#f8fafc")
    steps = [
        ("1", "静置", "确认急停开关不会自行漂移"),
        ("2", "接近 / 闭合", "左右对称夹块，摩擦系数 1.0"),
        ("3", "抬升", "测试每指 1 N、3 N、10 N"),
        ("4", "独立复验", "记录 contact / hold / rise / slip"),
    ]
    sx = probe_x + 30
    for index, (number, heading, detail) in enumerate(steps):
        x = sx + index * 375
        draw.ellipse((x, probe_y + 90, x + 58, probe_y + 148), fill="#0ea5e9")
        draw.text((x + 20, probe_y + 103), number, font=section_font, fill="#082f49")
        draw.text((x + 75, probe_y + 88), heading, font=section_font, fill="#e0f2fe")
        draw.text((x + 75, probe_y + 132), detail, font=small_font, fill="#94a3b8")
        if index < 3:
            draw.line((x + 255, probe_y + 119, x + 335, probe_y + 119), fill="#64748b", width=4)
            draw.polygon([(x + 335, probe_y + 119), (x + 322, probe_y + 111), (x + 322, probe_y + 127)], fill="#64748b")
    draw.text((probe_x + 24, probe_y + 235),
              "本图中的白色碰撞体不会出现在默认渲染中；它们只参与接触、摩擦、质量和动力学计算。",
              font=body_font, fill="#fbbf24")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _render_grasp_frames(scene: Path, body: str, output: Path, force: float) -> tuple[list[Path], list[dict]]:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="real2sim_visual_grasp_") as directory:
        augmented = Path(directory) / "grasp.xml"
        root = _grasp_geometry(scene, body, 0.1, 1.0, 0.006, "auto", "y", 2.0)
        _rebase_scene_assets(root, scene, augmented)
        ET.ElementTree(root).write(augmented, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(augmented))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stage2_initial")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        close_motors = [model.actuator(f"stage3_probe_close_motor_{side}").id for side in ("left", "right")]
        raise_motors = [model.actuator(f"stage3_probe_raise_motor_{side}").id for side in ("left", "right")]
        jaw_bodies = [model.body(f"stage3_probe_jaw_{side}").id for side in ("left", "right")]
        target_id = model.body(body).id
        for motor in close_motors:
            model.actuator_forcerange[motor] = [-force, force]
        camera = "stage3_camera_0"
        width, height = (int(v) for v in model.cam_resolution[model.camera(camera).id])
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        close_camera = mujoco.MjvCamera()
        close_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        close_camera.distance = 0.18
        close_camera.azimuth = 90
        close_camera.elevation = -8
        frames = [(0.0, "Initial", "夹块尚未闭合"), (0.12, "Closing", "对称闭合"),
                  (0.25, "Contact", "建立双侧接触"), (0.40, "Lifting", "抬升 0.08 m"),
                  (0.65, "Held", "检查保持与滑移")]
        rendered: list[Path] = []
        metrics: list[dict] = []
        current = 0.0
        for target_time, label, detail in frames:
            while current < target_time - 1e-9:
                step = current < 0.25
                for motor in close_motors:
                    data.ctrl[motor] = 0.05 if step else 0.05
                lift = 0.0 if current < 0.25 else min(0.08, 0.08 * (current - 0.25) / 0.4)
                for motor in raise_motors:
                    data.ctrl[motor] = lift
                mujoco.mj_step(model, data)
                current += model.opt.timestep
            contacts = []
            for contact_id in range(data.ncon):
                name1 = model.geom(data.contact[contact_id].geom1).name
                name2 = model.geom(data.contact[contact_id].geom2).name
                if "stage3_probe_pad" in name1 or "stage3_probe_pad" in name2:
                    contacts.append((name1, name2))
            target_z = float(data.xpos[target_id, 2])
            jaw_z = float(np.mean(data.xpos[jaw_bodies, 2]))
            close_camera.lookat[:] = data.xpos[target_id]
            close_camera.lookat[2] += 0.015
            metrics.append({"label": label, "detail": detail, "time_s": target_time,
                            "target_z": target_z, "jaw_z": jaw_z,
                            "contacts": len(contacts), "rise_m": target_z - float(data.xpos[target_id, 2])})
            with mujoco.Renderer(model, height=height, width=width) as renderer:
                renderer.update_scene(data, camera=close_camera)
                rgb = renderer.render().copy()
            path = output / f"{len(rendered):02d}_{label.lower()}.png"
            cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            rendered.append(path)
        return rendered, metrics


def _grasp_montage(scene: Path, body: str, task_dir: Path, output: Path) -> Path:
    report_path = task_dir / "physics/physics_group_20260925/grasp_Stage2_EStopHousing/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames, _ = _render_grasp_frames(scene, body, output.parent / "frames", force=3.0)
    plt.style.use("dark_background")
    figure = plt.figure(figsize=(18, 11), facecolor="#0b1120")
    grid = figure.add_gridspec(3, 6, height_ratios=[1.1, 1.1, 0.8], hspace=0.32, wspace=0.18)
    for index, path in enumerate(frames):
        axis = figure.add_subplot(grid[0:2, index])
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        axis.imshow(image)
        axis.set_title(["初始", "闭合", "接触", "抬升", "保持"][index], fontsize=17, color="#e2e8f0")
        axis.axis("off")
    ax_force = figure.add_subplot(grid[2, 0:2])
    forces = [item["target_normal_force_per_jaw_n"] for item in report["trials"]]
    lifts = [item["rise_m"] * 1000 for item in report["trials"]]
    colors = ["#22c55e" if item["lifted"] else "#ef4444" for item in report["trials"]]
    ax_force.bar([str(int(v)) for v in forces], lifts, color=colors)
    ax_force.axhline(10, color="#fbbf24", linestyle="--", linewidth=1.5)
    ax_force.set_title("每指夹持力 vs 物体抬升", fontsize=14)
    ax_force.set_xlabel("目标力 / N per jaw")
    ax_force.set_ylabel("抬升 / mm")
    ax_force.grid(axis="y", alpha=0.2)
    ax_contact = figure.add_subplot(grid[2, 2:4])
    contact = [item["two_sided_contact_fraction"] * 100 for item in report["trials"]]
    ax_contact.plot(forces, contact, marker="o", color="#38bdf8", linewidth=3)
    ax_contact.set_ylim(0, 105)
    ax_contact.set_title("双侧接触保持率", fontsize=14)
    ax_contact.set_xlabel("目标力 / N per jaw")
    ax_contact.set_ylabel("contact / %")
    ax_contact.grid(alpha=0.2)
    ax_text = figure.add_subplot(grid[2, 4:6])
    ax_text.axis("off")
    text = ("急停开关模拟夹取\n\n"
            "• 两个对称摩擦方块\n"
            "• 每块质量：0.025 kg\n"
            "• 夹块摩擦：1.0\n"
            "• 位置伺服：200 gain\n"
            "• 每指上限：10 N\n\n"
            "1 N：未保持\n"
            "3 N：成功抬升，双侧接触 100%\n"
            "10 N：成功抬升")
    ax_text.text(0, 0.98, text, va="top", fontsize=15, color="#e2e8f0",
                 bbox={"boxstyle": "round,pad=0.8", "facecolor": "#111827", "edgecolor": "#334155"})
    figure.suptitle("Stage 3 物理探针：急停开关的双夹块夹取过程", fontsize=25, fontweight="bold", color="#f8fafc")
    figure.text(0.5, 0.01, "该实验用于检查场景中的接触、摩擦、质量和可抓取性，不等同于真实 Piper 夹爪标定。",
                ha="center", fontsize=13, color="#94a3b8")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return output


def _annotated_held_closeup(output: Path) -> Path:
    source = output / "frames/04_held.png"
    image = Image.open(source).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 72), "#0b1120")
    canvas.paste(image, (0, 72))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 15), "急停开关 · 3 N / 每指 · 左右对称夹取", font=_font(24, True), fill="#f8fafc")
    draw.text((112, 245), "左夹块", font=_font(18, True), fill="#38bdf8")
    draw.line((145, 268, 183, 205), fill="#38bdf8", width=3)
    draw.polygon([(183, 205), (174, 211), (185, 216)], fill="#38bdf8")
    draw.text((305, 245), "右夹块", font=_font(18, True), fill="#38bdf8")
    draw.line((337, 268, 304, 205), fill="#38bdf8", width=3)
    draw.polygon([(304, 205), (301, 216), (312, 211)], fill="#38bdf8")
    draw.text((202, 318), "急停开关", font=_font(18, True), fill="#fbbf24")
    draw.line((257, 318, 257, 274), fill="#fbbf24", width=3)
    draw.polygon([(257, 274), (251, 285), (263, 285)], fill="#fbbf24")
    path = output / "emergency_stop_grasp_held_closeup.png"
    canvas.save(path)
    return path


def _render_single_front_back(scene: Path, body: str, output: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="real2sim_front_back_grasp_") as directory:
        augmented = Path(directory) / "grasp.xml"
        root = _grasp_geometry(scene, body, 0.1, 1.0, 0.006, "auto", "y", 2.0)
        _rebase_scene_assets(root, scene, augmented)
        ET.ElementTree(root).write(augmented, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(augmented))
        model.vis.global_.offwidth = 1920
        model.vis.global_.offheight = 1080
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stage2_initial")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        close_motors = [model.actuator(f"stage3_probe_close_motor_{side}").id for side in ("left", "right")]
        raise_motors = [model.actuator(f"stage3_probe_raise_motor_{side}").id for side in ("left", "right")]
        target_id = model.body(body).id
        for motor in close_motors:
            model.actuator_forcerange[motor] = [-3.0, 3.0]
        total_steps = int(round(0.65 / model.opt.timestep))
        for step in range(total_steps):
            time = step * model.opt.timestep
            for motor in close_motors:
                data.ctrl[motor] = 0.05
            lift = 0.0 if time < 0.25 else min(0.08, 0.08 * (time - 0.25) / 0.4)
            for motor in raise_motors:
                data.ctrl[motor] = lift
            mujoco.mj_step(model, data)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 0.19
        camera.azimuth = 45
        camera.elevation = -12
        camera.lookat[:] = data.xpos[target_id]
        camera.lookat[2] += 0.01
        with mujoco.Renderer(model, height=1080, width=1920) as renderer:
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not write {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--single-front-back", action="store_true",
                        help="write only an unannotated 1920x1080 front/back grasp render")
    args = parser.parse_args()
    task_dir = args.task_dir.expanduser().resolve(strict=True)
    scene = Path(json.loads((task_dir / "final_manifest.json").read_text(encoding="utf-8"))["scene"]).resolve(strict=True)
    output = (args.output or task_dir / "visualization").expanduser().resolve()
    if args.single_front_back:
        path = _render_single_front_back(scene, "Stage2_EStopHousing",
                                          output / "emergency_stop_grasp_front_back_1080p.png")
        print(json.dumps({"render": str(path), "resolution": "1920x1080", "force_per_jaw_n": 3.0},
                         ensure_ascii=False, indent=2))
        return 0
    workflow = _workflow(task_dir, output / "stage3_workflow_overview.png")
    montage = _grasp_montage(scene, "Stage2_EStopHousing", task_dir, output / "emergency_stop_grasp_visualization.png")
    closeup = _annotated_held_closeup(output)
    print(json.dumps({"workflow": str(workflow), "grasp_visualization": str(montage),
                      "held_closeup": str(closeup), "frames": str(output / "frames")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
