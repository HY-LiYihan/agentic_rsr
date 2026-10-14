# Source-Specific Notices

The documents below are unmodified copies from the official RealSense ROS
repository at commit `7a87097c499e3b0b69518d54a8e7a9ac1924e87c`.
They accompany the exact matching D435 mesh content already present in the
fixed robot-control sources:

- `external/robot_control/vendor/fr3_d435i/realsense2_description/meshes/d435.dae`
- `external/robot_control/vendor/piper_isaac_sim/realsense2_description/meshes/d435.dae`

Both files match upstream `realsense2_description/meshes/d435.dae`, Git blob
`0ed32368fc5ba6912be530395a72553de9f21d9e`, byte for byte. The official
`realsense2_description/package.xml` declares Apache License 2.0; there is no
closer LICENSE, NOTICE, COPYING or README in that package's tree.

| Retained File | Official Source | Git Blob SHA |
| --- | --- | --- |
| [realsense-ros-LICENSE](realsense-ros-LICENSE) | [LICENSE](https://github.com/realsenseai/realsense-ros/blob/7a87097c499e3b0b69518d54a8e7a9ac1924e87c/LICENSE) | `d639b1e2457de50acbd2d943a593b701331a2824` |
| [realsense-ros-NOTICE.md](realsense-ros-NOTICE.md) | [NOTICE.md](https://github.com/realsenseai/realsense-ros/blob/7a87097c499e3b0b69518d54a8e7a9ac1924e87c/NOTICE.md) | `01c648c4e7eaad5ae4e84047af4ec5c13e2cb396` |

NOTICE is the upstream repository-wide software-component inventory. Keeping
it does not imply that every listed component is distributed with a standalone
mesh. These documents do not license the main project, custom mounting parts,
Piper stand, unmatched converted meshes, or all generated robot surfaces.
The segmented D435i model has a separate retained upstream license in the
control submodule; it is not interchangeable with these original DAE notices.

Per-file evidence is in [urdf-assets.json](../provenance/urdf-assets.json).
See [THIRD_PARTY.md](../THIRD_PARTY.md) for remaining source and rights questions.
