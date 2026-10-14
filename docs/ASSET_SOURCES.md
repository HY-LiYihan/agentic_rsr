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

On 2026-10-14, the complete PDF served by GitHub Pages matched both the
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
