# Publication Media Sources

Prepared on 2026-10-14 from the author's supplied `Agentic_RSR_paper` folder.
The author list, title, affiliations and reported numbers follow `main.tex`,
`section/04_experiments.tex` and `tab/task_comparison.tex` in that source.
This is a preprint, not a claim of venue acceptance or a new repository validation run.

| Published file | Supplied paper source | Transformation |
| --- | --- | --- |
| `assets/paper.pdf` | `main.pdf` | Unmodified copy |
| `assets/overview.png` | `img/asset/fig_teaser.pdf` | 144 dpi rasterization, outer white margin trim |
| `assets/metric-alignment.png` | `img/asset/agentic_rsr.pdf` | Same |
| `assets/real-sim.png` | `img/asset/sim_real.pdf` | Same |
| `assets/robot-execution.png` | `img/asset/real_robot_execution.pdf` | Same |
| `assets/policy-loop.png` | `img/asset/policy_iteration_loop.pdf` | Same |

No panel or failure marker was removed. The previous illustrative README teaser
is no longer used. Paper figures are published at the author's request; no separate
permissive reuse license for figures or video has been selected. Retain attribution
and obtain permission for uses beyond applicable rights.

On 2026-10-14, the complete PDF served by the project page matched both the
checked-in `assets/paper.pdf` and the supplied `main.pdf`, SHA256
`0f60edefef3f67bbe02bb3c8435e755d695ca70277bd3b21858757a25edeaabe`.

## Affiliation Marks

- Sun Yat-sen University: [official homepage](https://www.sysu.edu.cn/),
  [header image](https://www.sysu.edu.cn/images/zhuzhanLOGO.png).
- HKUST (Guangzhou): [official homepage](https://www.hkust-gz.edu.cn/),
  [header image](https://www.hkust-gz.edu.cn/wp-content/themes/hkust-gz-official-0827/images/logo-e-white-2x.png).

The official white header marks were recolored to SYSU green and HKUST dark blue,
preserving shape and aspect ratio, then placed on white backgrounds for readability.
These marks identify the paper authors' affiliations. They remain the property
of their institutions; they are not included under the project's Apache-2.0
license and do not imply institutional endorsement. Institutional branding policies
continue to apply.

## HCL Lab Mark

The author supplied two Humanoid Computing Lab (HCL Lab) logo images on
2026-10-14 and requested inclusion on the project page. The clearer square
version is retained without cropping, recoloring or alteration as
`assets/hcl-lab-logo.jpg` (1024x1024 JPEG), SHA256
`3eab854a2a28c826ad13a95ed56af738f8a598fba361af6df717d8140f6e3e76`.
The mark remains subject to its owner's rights; the project's Apache-2.0
software license does not grant trademark or logo reuse rights.

On 2026-10-15, the author requested a transparent background. The displayed
`assets/hcl-lab-logo.png` preserves the 1024x1024 grid, removes the near-black
background and feathers/unmattes dark JPEG edges. The original JPEG is retained.
Regenerate with `python scripts/prepare_hcl_logo.py` using NumPy and Pillow.
The compact masthead uses `assets/hcl-lab-logo-trimmed.png`, which only removes
fully transparent outer margins (558x767); all visible logo pixels are retained.
The author's 2026-10-15 correction associates first author Yihan Li with both
Sun Yat-sen University and HKUST (Guangzhou).

## White Drawer Video Comparison

The local comparison below the pipeline figure uses `data/raw/frank/close_drawer.mp4`
and the author's downloaded
`Agentic_RSR_all_models_18_scene_videos/models/gpt-6-astra/dsh/franka/close_drawer/aligned.mp4`.
The render's `video_validation.json` binds it to the original video SHA256
`7045f7466752965d81d933c8f849b8133f5b06c30cbf460b54b224afbd4174e4`,
with 198 frames and zero recorded source/render timestamp error. The SIM video
shows static Stage 2 geometry from aligned moving cameras, not reconstructed
robot/object motion or a policy rollout.

`scripts/prepare_comparison_videos.py` creates silent 1280x720 H.264 animations
that play forward then reverse at the original 30000/1001 frame rate, and JPEG
posters. Each loop contains 394 frames; repeated turnaround endpoints are omitted.
They autoplay like GIFs while keeping both views synchronized. The slider only
controls the divider and defaults to the midpoint. The local
`assets/comparison/manifest.json` records source and output hashes. Videos remain
Git-ignored; this local preview does not upload or publish experiment media.

## Resource Links

The masthead resource links follow the icon-and-label convention seen on
[Nerfies](https://nerfies.github.io/) and [OpenVLA](https://openvla.github.io/).
The styling is implemented locally, without copying their site code.
The arXiv icon comes from Simple Icons (CC0; `assets/icons/SIMPLE_ICONS_LICENSE.md`),
the GitHub mark from Primer Octicons (MIT; `assets/icons/OCTICONS_LICENSE`),
and the citation icon from Lucide (ISC; `assets/icons/LICENSE`). The Hugging Face
logo is the copy served by [OpenVLA](https://openvla.github.io/static/images/hf_icon.svg).
Brand marks remain subject to their owners' trademark rights.

## Regeneration

Install Node.js 22 and `sharp` in a separate tool environment. Use `pdftoppm`
to render each supplied figure as `<source_name>.png` in a temporary directory:

```bash
pdftoppm -png -r 144 -singlefile /path/to/paper/img/asset/fig_teaser.pdf /path/to/renders/fig_teaser
# Repeat for agentic_rsr, sim_real, real_robot_execution and policy_iteration_loop.
NODE_PATH=/path/to/tool/node_modules node scripts/prepare_project_media.mjs \
  /path/to/renders /path/to/sysu-header.png /path/to/hkust-header.png
```

This tool is optional and is not an inference dependency. The original paper folder
and the university image downloads are not required to view the published site.
