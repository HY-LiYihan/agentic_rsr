"""Run an isolated SAM3 prompt trial and build one 32-frame comparison per scene."""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from perception.sam3 import DEFAULT_PROMPT
from perception.config import create_sam3, release_backend

from .pipeline import sha256
from .sim3_registration import ROOT


REQUESTED_PROMPT = "带底座机械臂加相机"
PROMPT = DEFAULT_PROMPT
FRAMES = range(32)
THUMBNAIL = (192, 108)


def segment_with_retry(service, frame: Path, prompt: str) -> np.ndarray:
    for attempt in range(3):
        try:
            return service.segment(frame, prompt=prompt)
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def segment_scene(source: Path, output: Path, service, prompt: str = PROMPT) -> dict:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    video = Path(manifest["source_video"])
    if (manifest.get("robot") != "piper" or manifest.get("status") != "inference_complete"
            or not video.is_file() or sha256(video) != manifest.get("video_sha256")):
        raise ValueError(f"Piper video/inference is missing or stale: {source.name}")
    output.mkdir(parents=True, exist_ok=True)
    mask_folder = output / "masks"
    mask_folder.mkdir(exist_ok=True)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if report and (report.get("prompt") != prompt or report.get("video_sha256") != manifest["video_sha256"]):
        raise ValueError(f"existing trial used a different prompt or video: {output}")
    report.setdefault("prompt", prompt)
    report.setdefault("video_sha256", manifest["video_sha256"])
    report["status"] = "running"
    report.setdefault("per_frame", {})
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for index in FRAMES:
        frame_path = source / "frames" / f"{index:02d}.jpg"
        cached_path = mask_folder / f"mask_{index:02d}.npy"
        key = f"{index:02d}"
        if cached_path.exists() and key in report["per_frame"]:
            cached = np.load(cached_path, allow_pickle=False)
            if cached.shape == (288, 512) and cached.dtype == np.bool_:
                continue
        if not frame_path.is_file():
            raise FileNotFoundError(frame_path)
        candidate = np.asarray(segment_with_retry(service, frame_path, prompt), dtype=bool)
        if candidate.shape != (288, 512):
            raise ValueError(f"SAM3 mask is not 288x512 for {source.name}/{index:02d}")
        np.save(cached_path, candidate)
        response = getattr(service, "metadata", {})
        report["per_frame"][key] = {"coverage_percent": float(candidate.mean() * 100),
                                     "instance_count": response.get("k"),
                                     "conf_thresh": response.get("conf_thresh")}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    tiles = []
    old_areas = []
    new_areas = []
    for index in FRAMES:
        image = cv2.imread(str(source / "frames" / f"{index:02d}.jpg"))
        old_mask = np.load(source / "sam3_masks" / f"mask_{index:02d}.npy", allow_pickle=False)
        new_mask = np.load(mask_folder / f"mask_{index:02d}.npy", allow_pickle=False)
        if image is None or image.shape[:2] != old_mask.shape or old_mask.shape != new_mask.shape:
            raise ValueError(f"mask/RGB alignment failed for {source.name}/{index:02d}")
        old_area, new_area = float(old_mask.mean() * 100), float(new_mask.mean() * 100)
        old_areas.append(old_area)
        new_areas.append(new_area)
        old_overlay = image.copy()
        new_overlay = image.copy()
        old_overlay[old_mask] = (0.45 * image[old_mask] + 0.55 * np.array([0, 215, 0])).astype(np.uint8)
        new_overlay[new_mask] = (0.45 * image[new_mask] + 0.55 * np.array([200, 25, 210])).astype(np.uint8)
        for mask, overlay, color in ((old_mask, old_overlay, (0, 255, 0)),
                                     (new_mask, new_overlay, (255, 0, 255))):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 1)
        body = np.concatenate([cv2.resize(part, THUMBNAIL, interpolation=cv2.INTER_AREA)
                               for part in (image, old_overlay, new_overlay)], axis=1)
        header = np.full((29, body.shape[1], 3), 24, dtype=np.uint8)
        label = (f"{index:02d}  RGB   old {old_area:.1f}%   new {new_area:.1f}% "
                 f"K={report['per_frame'][f'{index:02d}']['instance_count']}")
        cv2.putText(header, label, (5, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    .52, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.concatenate((header, body), axis=0))
    contact = np.concatenate([np.concatenate(tiles[start:start + 4], axis=1)
                              for start in range(0, 32, 4)], axis=0)
    image_path = output / "comparison_32.jpg"
    if not cv2.imwrite(str(image_path), contact, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(image_path)
    report.update({"status": "complete", "frames": list(FRAMES), "reference_prompt": manifest["sam3_prompt"],
                   "mean_old_coverage_percent": float(np.mean(old_areas)),
                   "mean_new_coverage_percent": float(np.mean(new_areas)),
                   "empty_new_frames": [index for index, value in enumerate(new_areas) if value == 0],
                   "comparison_image": image_path.name})
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"scene": source.name, "old_percent": report["mean_old_coverage_percent"],
            "new_percent": report["mean_new_coverage_percent"],
            "empty_new_frames": len(report["empty_new_frames"])}


def run_all(processed: Path, raw: Path, output: Path, service, prompt: str = PROMPT) -> list[dict]:
    videos = sorted(raw.glob("*.mp4"))
    if not videos:
        raise ValueError(f"no Piper videos in {raw}")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for position, video in enumerate(videos, 1):
        print(f"[{position}/{len(videos)}] {video.stem}: SAM3 text prompt", flush=True)
        result = segment_scene(processed / video.stem, output / video.stem, service, prompt)
        results.append(result)
        print(f"    coverage {result['old_percent']:.1f}% → {result['new_percent']:.1f}%, "
              f"empty frames {result['empty_new_frames']}", flush=True)
    (output / "summary.json").write_text(json.dumps({"requested_prompt": REQUESTED_PROMPT,
                                                      "actual_prompt": prompt, "results": results},
                                                     indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cards = ''.join(
        f'<section><h2>{html.escape(item["scene"])} · 旧 {item["old_percent"]:.1f}% '
        f'→ 新 {item["new_percent"]:.1f}% · 空帧 {item["empty_new_frames"]}</h2>'
        f'<a href="{quote(item["scene"], safe="._-")}/comparison_32.jpg">'
        f'<img loading="lazy" src="{quote(item["scene"], safe="._-")}/comparison_32.jpg" '
        f'alt="{html.escape(item["scene"])} 32 帧新旧对照"></a></section>' for item in results)
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Piper SAM3 提示词对照</title>'
        '<style>body{max-width:2304px;margin:24px auto;padding:0 12px;font-family:system-ui}'
        'section{border-bottom:1px solid #bbb;padding:10px}img{width:100%;height:auto}</style>'
        '<h1>Piper SAM3：每场景一张 32 帧对照图</h1>'
        f'<p>用户原句：{html.escape(REQUESTED_PROMPT)}。'
        '中文原句的首帧试跑未检出实例；为完成全视频试验，实际发送同义英文：'
        f'<code>{html.escape(prompt)}</code>，conf=0.25。每格依次是原图、旧提示词绿色 mask、'
        '新提示词紫色 mask。仅评估分割，不修改现有 mask 或配准结果。</p>' + cards + '</html>\n',
        encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Trial a SAM3 prompt on all 32 frames of current Piper videos")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed/piper/stage1")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/piper")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sam3_prompt_trials/piper/base_camera")
    args = parser.parse_args()
    service = create_sam3()
    try:
        run_all(args.processed, args.raw, args.output, service)
    finally:
        release_backend(service)


if __name__ == "__main__":
    main()
