"""Remove the supplied HCL logo's near-black JPEG background."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "docs/assets/hcl-lab-logo.jpg"
    destination = root / "docs/assets/hcl-lab-logo.png"
    rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.float32)
    background = rgb[0, 0]

    # Feather compressed dark edges and undo their black-background blending.
    signal = np.maximum(rgb - background, 0).max(axis=2)
    signal = np.asarray(
        Image.fromarray(signal.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
        dtype=np.float32,
    )
    alpha = np.clip((signal - 14) / 40, 0, 1)
    foreground = background + (rgb - background) / np.maximum(alpha[..., None], 0.25)
    rgba = np.concatenate(
        [np.clip(foreground, 0, 255), alpha[..., None] * 255], axis=2
    ).round().astype(np.uint8)
    rgba[rgba[..., 3] == 0, :3] = 0
    image = Image.fromarray(rgba)
    image.save(destination, optimize=True)
    image.crop(image.getbbox()).save(destination.with_stem("hcl-lab-logo-trimmed"), optimize=True)
    print(destination)


if __name__ == "__main__":
    main()
