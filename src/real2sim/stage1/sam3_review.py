"""Visualize cached SAM3 union masks without running inference or registration."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from .pipeline import sha256
from .sim3_registration import ROOT


FRAMES = range(32)
THUMBNAIL = (256, 144)


def review_scene(source: Path, output: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "inference_complete" or manifest.get("robot") != "piper":
        raise ValueError(f"incomplete Piper inference: {source}")
    video = Path(manifest["source_video"])
    if not video.is_file() or sha256(video) != manifest["video_sha256"]:
        raise ValueError(f"source video changed since SAM3 inference: {source.name}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    tiles = []
    percentages = []
    source_indices = manifest.get("source_frame_indices", list(FRAMES))
    for index in FRAMES:
        frame = source / "frames" / f"{index:02d}.jpg"
        image = cv2.imread(str(frame))
        mask = np.load(source / "sam3_masks" / f"mask_{index:02d}.npy", allow_pickle=False)
        if image is None or image.shape[:2] != mask.shape or mask.dtype != np.bool_:
            raise ValueError(f"frame/mask mismatch in {source.name}, frame {index}")
        area = float(mask.mean() * 100)
        percentages.append(area)
        overlay = image.copy()
        overlay[mask] = (0.5 * image[mask] + 0.5 * np.array([0, 220, 0])).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
        mask_image = (mask.astype(np.uint8) * 255)
        overlay_path = output / f"overlay_{index:02d}.png"
        mask_path = output / f"mask_{index:02d}.png"
        if not cv2.imwrite(str(overlay_path), overlay) or not cv2.imwrite(str(mask_path), mask_image):
            raise OSError(f"cannot write SAM3 previews for {source.name} frame {index}")

        original = quote(os.path.relpath(frame, output), safe="/._-")
        tile = np.concatenate((cv2.resize(image, THUMBNAIL),
                               cv2.resize(overlay, THUMBNAIL)), axis=1)
        banner = np.full((25, tile.shape[1], 3), 24, dtype=np.uint8)
        cv2.putText(banner, f"frame {index:02d}   mask {area:.1f}%", (7, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.concatenate((banner, tile), axis=0))
        rows.append('<article><h3>抽帧 {frame:02d} · 原视频帧 {original_index} · '
                    'mask {area:.1f}%</h3><div class="views">'
                    '<figure><img loading="lazy" src="{original}" alt="原图"><figcaption>原图</figcaption></figure>'
                    '<figure><img loading="lazy" src="overlay_{frame:02d}.png" alt="SAM3 叠图">'
                    '<figcaption>绿色：SAM3 联合 mask</figcaption></figure>'
                    '<figure><img loading="lazy" src="mask_{frame:02d}.png" alt="纯 mask">'
                    '<figcaption>白色：mask；黑色：背景</figcaption></figure>'
                    '</div></article>'.format(frame=index, original_index=source_indices[index],
                                              area=area, original=original))
    overview = np.concatenate([np.concatenate(tiles[start:start + 2], axis=1)
                               for start in range(0, 32, 2)], axis=0)
    if not cv2.imwrite(str(output / "overview.jpg"), overview,
                       [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise OSError("cannot write SAM3 overview")
    response = manifest.get("models", {}).get("sam3", {}).get("response", {})
    title = html.escape(source.name)
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>SAM3 | '
        f'{title}</title><style>body{{font-family:system-ui;background:#17191d;color:#eee;'
        'max-width:1700px;margin:auto;padding:24px}}a{color:#90dbaa}'
        'article{border-top:1px solid #555;margin:24px 0;padding:8px 0}'
        '.views{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0}'
        'figure img{width: min(100%,512px);height:auto}figure figcaption{padding:5px}'
        '</style><h1>SAM3 单场景核查：' + title + '</h1>'
        '<p>32 帧；绿色只代表 SAM3 缓存的机器人 mask，不含深度或 URDF 投影。'
        '逐帧以 robotic arm、conf=0.25 文本提示推理；多个实例直接合并。'
        '已缓存结果不能还原逐实例分数与框。</p>'
        f'<p>覆盖率均值 {np.mean(percentages):.1f}%；最后一帧服务返回实例数 '
        f'{html.escape(str(response.get("k", "未知")))}（其他帧的实例数未记录）。'
        '<a href="overview.jpg">查看 32 帧汇总图</a></p>' + ''.join(rows) + '</html>\n',
        encoding="utf-8")
    return {"scene": source.name, "mask_mean_percent": float(np.mean(percentages)),
            "mask_min_percent": float(min(percentages)), "mask_max_percent": float(max(percentages)),
            "empty_frames": [index for index, area in enumerate(percentages) if area == 0],
            "last_response_k": response.get("k"), "video_sha256": manifest["video_sha256"]}


def review_all(processed: Path, output: Path, raw: Path, only: str | None = None) -> list[dict]:
    videos = [raw / f"{only}.mp4"] if only else sorted(raw.glob("*.mp4"))
    if not videos or any(not video.is_file() for video in videos):
        raise ValueError("no matching Piper videos")
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for video in videos:
        source = processed / video.stem
        summary = review_scene(source, output / video.stem)
        summaries.append(summary)
        print(f"{video.stem}: 32 masks, average coverage {summary['mask_mean_percent']:.1f}%", flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    cards = ''.join(
        f'<li><h2><a href="{quote(item["scene"], safe="._-")}/index.html">'
        f'{html.escape(item["scene"])}</a> · coverage {item["mask_mean_percent"]:.1f}%</h2>'
        f'<a href="{quote(item["scene"], safe="._-")}/index.html">'
        f'<img loading="lazy" src="{quote(item["scene"], safe="._-")}/overlay_00.png" '
        f'alt="{html.escape(item["scene"])} 第 0 帧 mask 叠图"></a> '
        f'<a href="{quote(item["scene"], safe="._-")}/overview.jpg">32 帧总览</a></li>'
        for item in summaries)
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Piper SAM3 核查</title>'
        '<style>body{font-family:system-ui;margin:24px auto;max-width:1550px;padding:0 12px}'
        'ol{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:18px;'
        'list-style:none;padding:0}li{padding:12px;border:1px solid #ccc}li img{width:100%}</style>'
        f'<h1>Piper：{len(summaries)} 段视频的 SAM3 诊断</h1>'
        '<p>左侧原图、右侧绿色 SAM3 mask；不是机械臂配准投影。'
        '点任务名查看全部 32 帧的原图、叠图与纯 mask；首页缩略图为第 0 帧。</p><ol>' + cards + '</ol></html>\n',
        encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize existing Piper SAM3 masks without remote calls")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed/piper/stage1")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/piper")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sam3_review/piper")
    parser.add_argument("--only", help="Show only one current Piper video by stem")
    args = parser.parse_args()
    review_all(args.processed, args.output, args.raw, args.only)


if __name__ == "__main__":
    main()
