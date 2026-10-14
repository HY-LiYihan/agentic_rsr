# Robot Control Integration

## 现有仓库

`external/robot_control` 是独立的机器人控制仓库，通过 Git submodule 接入主项目。它目前已经覆盖：

- Piper 的 MuJoCo 仿真、Piper 真机接口和 twin 模式；
- Franka FR3 的 MuJoCo 仿真、基于本地 `pylibfranka`/`libfranka` 的真机运动后端及 twin 接口；
- 统一的 `robot_control` CLI 与 Python API；
- 关节、末端位姿、夹爪和场景控制；
- Piper 腕部 D435i RGB-D 相机，以及 MuJoCo 中对应的腕部相机；
- Piper/FR3 的 URDF、场景和必要的模型转换逻辑。

当前仓库 README 中的 `pepper` 是 Piper 的兼容别名，不是另一个 Pepper 机器人型号；主项目文档统一使用 **Piper**，避免和机器人型号混淆。

这个仓库应保持为独立项目，不把它的源码复制到 `agentic_rsr`。主项目只通过安装包、CLI 或一个薄适配层调用它。

## 在主项目中的位置

推荐的依赖关系如下：

```text
agentic_rsr/
├── src/
│   ├── perception/                 # VGGT-Ω、SAM3
│   ├── real2sim/                   # 三个 Real2Sim Stage
│   ├── sim/                        # 场景、任务、随机化、策略
│   ├── sim2real/                   # 真机闭环、repair、memory
│   └── robot_control_adapter/      # 主项目对控制仓库的薄适配层
│
├── external/                      # 外部依赖
│   └── robot_control/              # 必选 Git submodule
│
└── configs/robots/
    ├── piper.yaml                  # Piper 的控制和相机配置
    └── fr3.yaml                    # FR3 的控制和相机配置
```

上游仓库可以在主项目之外单独维护，但主项目使用固定的 submodule 路径：

```text
~/agentic_rsr/external/robot_control/
```

本项目正式将它作为 Git submodule 放到 `external/robot_control/`，不把外部仓库源码复制进主项目。主项目提交记录中的 submodule commit 即为本次 Sim/Real 控制接口使用的版本。

首次取得主项目后，需要递归初始化子模块：

```bash
git submodule update --init --recursive
```

更新控制仓库时，应显式更新并记录 submodule commit，不能让主项目依赖工作区中未提交的外部改动：

```bash
git -C external/robot_control status
git -C external/robot_control pull
git add external/robot_control
```

## 职责边界

| 能力 | `agentic_rsr` | `robot_control` |
| --- | --- | --- |
| VGGT-Ω / SAM3 感知 | 负责 | 不负责 |
| Real2Sim 场景重建与 Blender 产物 | 负责 | 不负责 |
| MuJoCo 场景中的任务物体与物理调优 | 负责 | 提供机器人底座和控制接口 |
| 机器人 URDF、关节、夹爪、腕部相机 | 读取和校验 | 负责维护 |
| MuJoCo / 真机 / twin 后端 | 通过适配层调用 | 负责实现 |
| 策略、任务程序、Agent memory 和 repair | 负责 | 提供动作执行能力 |
| 真机安全限制和硬件通信 | 通过接口使用 | 负责底层实现，主项目负责上层门控 |

主项目不能重新定义一套和控制仓库不一致的机器人模型。Stage 1 生成的机械臂场景、Sim 使用的 URDF 和 Real 使用的控制接口必须来自同一版本的机器人资产；腕部相机也必须使用同一安装和坐标定义。

## 接入方式

适配层优先使用控制仓库已有的统一接口，而不是直接访问其内部模块。目标接口包括：

```text
connect(backend="mujoco" | "real" | "twin", robot="piper" | "franka_fr3")
state()
pose()
move_joints(...)
move_p(...)
gripper(...)
observe_wrist_rgbd()
stop()
```

上述是**主项目适配层的目标接口**，不是 `robot_control.api.Robot` 的逐项方法列表。其中 `pose()` 由 `Robot.state().pose` 提供；`observe_wrist_rgbd()` 复用 `Robot.camera(width=..., height=...)` 采集的 RGB-D 帧。适配层只负责组合接口和保持 Sim/Real 的观测格式一致。

主项目的 Sim 和 Sim2Real 程序应通过同一套高层动作接口运行；只切换 backend 和 robot 配置，不重写任务逻辑。真机调用必须额外经过主项目的安全门控，不能因为仿真接口和真机接口形式相同就自动放行。

当前能力边界需要记录在配置和文档中：Piper 支持 MuJoCo、真机和 twin；FR3 也已有 MuJoCo、真机运动及 twin 的代码接口，但依赖本机安装的 `pylibfranka`/`libfranka`、实际机器人连接和安全配置。FR3 真机腕部相机外参尚未提供；本项目尚未验证 FR3 真机闭环，不能因上游实现了后端就自动放行真实动作。`Robot.move_joints(...)`、`Robot.move_p(...)` 可选 `duration_s` 控制轨迹时长（FR3 默认 4 秒，Piper 仿真默认 2 秒）；Piper 真机不支持该参数。

子模块远端和固定路径记录在 `.gitmodules` 中。当前 Python 包名为 `robot_control`（旧 `piper_control` 导入路径兼容），主项目以 submodule commit 固定版本；FR3 真机后端已实现但未经本项目硬件验收。
