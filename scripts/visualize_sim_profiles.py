"""Compare active SIM exact/randomized wrist observations and robot arrival poses."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sim.episode import SimEpisode
from sim.harness import EXACT_CONFIG, RANDOMIZED_CONFIG
from sim.randomization import Pose3D
from sim.scene_adapter import DOWN_QUATERNION, PiperTabletopScene


names = ("blue block", "yellow bowl")
configs = {"exact": EXACT_CONFIG, "randomized": RANDOMIZED_CONFIG}
target = Pose3D((-0.08, -0.08, 0.97), DOWN_QUATERNION)


def render_initial(profile, scene_path, seed, output):
    scene = PiperTabletopScene(scene_path, names)
    try:
        episode = SimEpisode(scene, scene_path, seed, configs[profile])
        episode.reset(scene.nominal_objects(), scene.nominal_cameras())
        frame = np.asarray(scene.read_rgbd().color).copy()
        Image.fromarray(frame).save(output / f"{profile}_wrist.png")
        return frame, episode.to_manifest()
    finally:
        scene.close()


def measure_arrival(profile, scene_path, seed):
    scene = PiperTabletopScene(scene_path, names)
    try:
        episode = SimEpisode(scene, scene_path, seed, configs[profile])
        episode.reset(scene.nominal_objects(), scene.nominal_cameras())
        execution = episode.move_pose(target, "arrival_probe")
        measured = [float(value) for value in scene.robot.state().pose.position]
        requested = np.asarray(target.position)
        return {
            "requested_position_m": list(target.position),
            "arrived_command_position_m": list(execution.arrived_pose.position),
            "measured_position_m": measured,
            "requested_to_measured_error_mm": ((np.asarray(measured) - requested) * 1000).tolist(),
            "requested_to_measured_norm_mm": float(np.linalg.norm(np.asarray(measured) - requested) * 1000),
            "control_error": execution.error.to_manifest(),
        }
    finally:
        scene.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path,
                        default=Path(__file__).resolve().parents[1] / "tests/fixtures/stage3/code_as_policy_tabletop.xml")
    parser.add_argument("--output", type=Path, default=Path("runs/sim_profiles"))
    parser.add_argument("--seed", type=int, default=202)
    args = parser.parse_args()
    scene_path = args.scene.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = {}
    manifests = {}
    for profile in configs:
        frames[profile], manifests[profile] = render_initial(profile, scene_path, args.seed, output)
    difference = np.abs(frames["exact"].astype(float) - frames["randomized"].astype(float))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, title, image in zip(axes, ("exact wrist RGB", "randomized wrist RGB", "absolute RGB difference"),
                                  (frames["exact"], frames["randomized"], difference.mean(axis=2))):
        axis.imshow(image, cmap="magma" if image.ndim == 2 else None,
                    vmin=0 if image.ndim == 2 else None,
                    vmax=80 if image.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "wrist_image_difference.png", dpi=150)
    plt.close(figure)

    arrivals = {profile: measure_arrival(profile, scene_path, args.seed) for profile in configs}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    positions = np.arange(3)
    for index, profile in enumerate(configs):
        axes[0].bar(positions + (index - 0.5) * 0.3,
                    arrivals[profile]["requested_to_measured_error_mm"],
                    width=0.3, label=profile)
        measured = arrivals[profile]["measured_position_m"]
        axes[1].scatter(measured[0], measured[1], label=f"{profile} measured", s=90)
    axes[0].set_xticks(positions, ("x", "y", "z"))
    axes[0].set_ylabel("measured minus requested (mm)")
    axes[0].set_title("end-effector arrival error")
    axes[0].legend()
    axes[1].scatter(target.position[0], target.position[1], marker="x", s=100,
                    color="black", label="requested")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_title("requested vs measured XY")
    axes[1].axis("equal")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "control_arrival_difference.png", dpi=150)
    plt.close(figure)

    report = {
        "scene": str(scene_path), "seed": args.seed,
        "profiles": {profile: config.to_manifest() for profile, config in configs.items()},
        "episode_manifests": manifests,
        "image_mean_abs_rgb": float(difference.mean()),
        "image_changed_fraction_mean_abs_gt_10": float(np.mean(difference.mean(axis=2) > 10)),
        "arrivals": arrivals,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"image_mean_abs_rgb": report["image_mean_abs_rgb"],
                      "image_changed_fraction_mean_abs_gt_10": report["image_changed_fraction_mean_abs_gt_10"],
                      "arrival_norm_mm": {profile: entry["requested_to_measured_norm_mm"]
                                          for profile, entry in arrivals.items()}}, indent=2))


if __name__ == "__main__":
    main()
