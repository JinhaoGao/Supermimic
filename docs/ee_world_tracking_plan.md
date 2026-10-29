# 末端世界位姿跟踪（Workhorse 式）修改方案

分支：`feat/ee-world-tracking`（从 `dev_gjh` @ bf4cc5b 切出）。H4 的代码路径保持不变：新功能默认关闭时，H4 配置下的训练、评测和检查点逐位不变。

> **已定（10-29）**：末端位置**不缩放**；部署时**没有外部定位**；权重和核宽通过消融决定（bj9 八卡，见 §10）。§10 记录实际实现与本文初稿的差异，以 §10 为准。

## 0. 目标与总体思路

- **现状（H4）**：`--reference human`，配置为 human2 加 `term_foot_lift=0.25`。奖励是尺度无关的骨方向，外加根速度、航向、高度和抬脚高度。这些都是航向系或归一化的量，与机器人在世界里的 xy 位置无关，所以不会纠正累积漂移，也不约束手和脚在世界里的绝对位置。
- **目标**：跟踪人体动作中 5 个任务连杆（torso、左腕、右腕、左脚、右脚）的世界位姿，与 Workhorse 的 tracker 相同。
- **一个连续条件量 α ∈ [0, 1]**：α = 0 时等同 H4，α = 1 时偏向末端世界位姿。α 作为 goal 的一部分输入策略，部署时可在线调节。训练时按 episode 随机采样 α，奖励和终止按 α 插值。
- **复用的现有基础设施**：
  - `bone_map.py` 已定义这 5 个 `TaskLink`，标定文件里已有 `task_rot_offset`。
  - `human_task_poses` / `robot_task_poses` 可以直接调用。
  - 世界模型会积分根位置 `p`，所以世界系奖励在想象 rollout 里同样可以反传。
  - 已有 `goal_layout` graft（`trainer._graft_inputs`），可以从 H4 检查点热启动：新增的 goal 段零初始化，起步时 α = 0 的行为就是 H4。

## 1. Workhorse tracker 要点，以及本方案的取舍

| Workhorse（arXiv 2610.09117 §IV） | 本方案 |
|---|---|
| 5 个连杆的完整位姿，对齐点分别是相机、掌心、脚底中心 | 同样 5 个连杆，用现有 TaskLink 的点和标定旋转偏置 |
| **不缩放**：示教者录制时有实时 IK 预览，够不着的姿势当场修正 | AMASS 没有这一步，所以加一个可配置缩放 `ee_scale`（默认 k，见 §2.2），再用 IK 可达性重新筛选动作（§6） |
| 命令在机器人**当前 torso 系**下给出：6 个样本，间隔 0.04 s，覆盖 0.2 s | goal 中新增 EE 块，放在当前 torso_link 系下，沿用 K = 10 帧（间隔 0.02 s，同样覆盖 0.2 s） |
| 奖励 = 单位权重的高斯核之和。style 项是 14 个连杆在 IK 全身姿态上的宽核；task 项见右列 | style 项改为 H4 骨方向项（权重降低、核放宽）。task 项：torso 宽核 + 窄核（世界系），wrists 窄核（世界系），feet（torso 系），抬脚净空 |
| 命令增强：重规划间隔随机 0.2–1 s；两次重规划之间，刚性漂移从 0 线性增长，最多水平 10 cm、偏航 10° | 照搬，作为 `ee_drift` 增强。论文消融：有推力时成功率 64% vs 52% |
| 部署时没有全局定位：腿运动学加 IMU 的 KF，按 5 Hz 重新锚定 | 部署时做周期性重锚定，原理同上（§5）；有外部定位时可以不重锚 |
| 论文没给终止条件 | 沿用 H4 的终止判据，另加随 α 变化的 EE 世界误差终止（§3.3） |

## 2. 参考数据与对齐

### 2.1 缓存新增字段（`human/motions.py`，`CACHE_VERSION = "human-v3"`）

| 字段 | 形状 | 含义 |
|---|---|---|
| `h_task_pos` | [T, 5, 3] | 人体任务点的世界位置（未缩放），由 `human_task_poses` 算出 |
| `h_task_rot6d` | [T, 5, 6] | 任务连杆的旋转，已乘 `task_rot_offset`，即"机器人连杆应有的世界旋转"，存成 rot6d |
| `h_pelvis_xy` | [T, 2] | 人体骨盆世界 xy，作为对齐锚点 |
| `h_swing` | [T, 2] | 摆动脚标记：`1 − h_contact`，再加一点平滑 |
| `init_ee_*` | – | 以 5 连杆为目标的 IK 重置姿态（§6.1），供 α 偏高的 episode 重置用 |

### 2.2 世界对齐

episode 从帧 f₀ 开始时，机器人放在 env 原点，航向取人体在 f₀ 的航向（现有 reset 已经这样做）。由此定义从人体世界到机器人世界的映射：

```
p_target(t) = R_z(δψ(t)) · S · (p_h(t) − [a_x, a_y, 0]) + [δx(t), δy(t), 0]
R_target(t) = R_z(δψ(t)) · R_h(t) · C          (C = task_rot_offset，已存入缓存)
```

- **锚点**：a = `h_pelvis_xy[f₀]`，每个 episode 一次。
- **缩放 S**：
  - `ee_scale=k`（默认）：S = diag(k, k, k)，k = 0.792 / `h_rest`。这和 H4 中根速度、抬脚的 k 缩放一致。地面 z = 0 在缩放下不变。
  - `ee_scale=1`：不缩放，等同 Workhorse。用于以后在 G1 尺度下录制的任务数据。
- **漂移 (δx, δy, δψ)**：§4 的命令增强。

### 2.3 每个 env 的条件状态（新增 replay 字段 `cond`）

想象 rollout 从 replay 里的状态出发，所以每一步都要存下这一步的条件量：

```
cond = [α, ee_valid, a_x, a_y, δx, δy, δψ, δ̇x, δ̇y, δ̇ψ, steps_to_replan]   (11 维)
```

- 写入方式：`ReplayBuffer` 加一个张量，`add(cond=...)`；`sample_starts` 和 `sample_windows` 一并返回它。
- H4 配置下这个字段不分配，所以 H4 的 replay 逐位不变。

## 3. 策略输入、奖励与终止

### 3.1 goal（每帧）

| 段 | 维度 | 备注 |
|---|---|---|
| H4 原有：`h_feat` + rel_yaw | 140 | 不变 |
| EE 块：5 连杆在当前机器人 torso_link 系下，pos 3 + rot6d 6 | 45 | robot-relative 项。第一帧就是当前 torso 跟踪误差（同 Workhorse） |
| α | 1 | |
| `ee_valid` | 1 | 0 表示 EE 块被置零。部署时没有可用里程计的 α = 0 模式，可以直接输入全零 |

- 每帧共 187 维，即 `goal_layout` 追加 `("ee", 45), ("alpha", 1), ("ee_valid", 1)`。
- 从 H4 检查点 graft 时，新列零初始化。running mean/std 中新段的统计量从头开始。
- torso 系用完整 6 自由度（论文里明确说航向系看不到前倾和高度）。所需的机器人量：root 位置（训练时就是状态 `p`）、root 姿态（IMU）、关节角，然后做 FK。

### 3.2 奖励：R = (1 − α) · R_H4 + α · R_EE

- **满分一致**：R_H4 和 R_EE 的正权重之和都设为 10.0，这正是 H4 现有的总和。因此对任何 α，`reward_max_per_step` 都是 0.2，看门狗、采样器和 critic 的值域不受 α 影响。
- **R_H4**：完全复用现有 `HumanHeads.reward` 和 H4 配置。
- **R_EE**（新增 `w_ee_*` / `std_ee_*` 字段；括号内为初始值）：

| 项 | 计算 | 权重 | 核宽 |
|---|---|---|---|
| style：H4 的骨方向项（全局 + 局部） | 各项权重 × 0.3，骨方向核放宽到 0.6 | 共约 2.5 | 0.6 |
| `ee_torso_wide` | torso 世界位置 + 旋转 | 1.0 | 0.3 m / 0.4 rad |
| `ee_torso_narrow` | 同上 | 1.0 | 0.06 m / 0.15 rad |
| `ee_wrist_wide` | 两腕世界位置 | 1.0 | 0.3 m |
| `ee_wrist_pos` | 两腕世界位置 | 1.5 | 0.06 m |
| `ee_wrist_rot` | 两腕世界旋转 | 0.5 | 0.15 rad |
| `ee_foot` | 两脚在 torso 系下的位姿 | 1.0 | 0.06 m / 0.1 rad |
| `ee_clearance` | 人体脚处于摆动时，踝离地 ≥ 10 cm | 0.5 | 0.035 m |
| `root_vel` / `heading` / `height` | 同 H4 | 合计 1.0 | |

- **为什么腕部也加宽核**：论文的 wrists 只有窄核。但 SuperMimic 靠 BPTT 对奖励求梯度，窄核离目标远时梯度几乎为 0，所以腕部也加一个宽核（torso 的宽窄双核同理）。实际上这是 H4 里"腿核 0.35 → 0.6"那条教训的推广。
- **误差形式**：
  - 位置误差用 ‖·‖²。
  - 旋转误差用 `2 − 2cos` 形式的弦长平方，例如 ‖R_aᵀR_b − I‖²_F / 4。不要用 `acos`，它在 0 处梯度是无穷大，和 `human_input_plan.md` §4 里骨方向误差不用 atan2 角度是同一个原因。

### 3.3 终止

- H4 的终止判据（躯干、高度、`term_foot_lift`）一直保留。
- 新增 EE 世界误差终止：torso 误差 > `term_ee_torso`（0.5 m），或任一腕误差 > `term_ee_wrist`（0.4 m）。
  - 阈值随 α 线性放松：`th(α) = th / max(α, ε)`，α = 0 时等于关闭。
  - 软终止沿用 sigmoid 加 noisy-OR，仿真和想象两端调用同一个函数，与现有做法一致。

## 4. α 采样与命令增强

- **α 采样**（每次重置）：25% 取 0，25% 取 1，50% 取 U(0, 1)。另外以 p = 0.3 的概率在 episode 中途切换到新的 α，切换时线性过渡 0.5 s，这样部署时可以在运行中调 α。
- **`ee_valid`**：α = 0 的 episode 中，50% 的概率把 EE 块置零、`ee_valid` 设为 0。这样 H4 模式部署时不需要任何里程计。
- **漂移**（同 Workhorse）：
  - 重规划间隔 ~ U(0.2, 1.0) s。每次重规划时，漂移归零，并重新采样漂移速率，使到下次重规划时漂移达到 |δxy| ≤ 10 cm、|δψ| ≤ 10°。
  - 每个 goal 样本带着外推到它自己时刻的漂移。
  - 奖励也对着漂移后的命令算（论文原话：tracker 分不出漂移和本意动作）。
  - 想象 rollout 里，漂移按 `cond` 里存的速率继续外推；在 `steps_to_replan` 到点时清零，之后按同样的分布重新采样。

## 5. 部署：α 作为运行时参数

目前仓库里**还没有 human 策略的部署导出**，`export_deploy.py` 只支持 robot 参考。这一阶段需要一起补上：

1. **`scripts/human/export_deploy_human.py`**：导出 ONNX，输入为 `obs_hist [1, 10, 93]` 和 `goal [1, 10, 187]`。同时导出参考侧量：离线算好的 `h_feat`、`h_task_pos`、`h_task_rot6d`，每个 clip 一个 npz。
2. **部署端 goal 构造**：每步只算机器人相关的量。
   - rel_yaw。
   - EE 块：用状态估计器给出的 root 位姿做 FK 得到 torso_link 位姿，再把目标变换进 torso 系。
   - α 来自手柄、键盘或 ROS param，限速 ≤ 1 /s，与训练时的过渡速度匹配。
3. **两种锚定方式**：
   - (a) 只用机载里程计（Unitree 的状态估计，或腿运动学加 IMU 的 KF），每 0.2–1 s 把参考重新锚定到当前 torso。这是 Workhorse 的做法，产生的漂移在训练增强覆盖范围内。
   - (b) 有外部定位（mocap、SLAM）时不重锚，跟踪真正固定的世界坐标。
4. **α = 0 且没有里程计时**：EE 块置零、`ee_valid = 0`，部署方式等同 H4。

### 5.1 实现与用法（10-29）

H4 和 ee_mode 检查点用同一条路径（goal 140 / 187 维由检查点的 `goal_layout` 决定）。

**导出**（需要 mjlab，建一个单 env 的 eval 场景读机器人模型、动作项和站立几何）：

```bash
cd ~/supermimic_ws/mjlab && PYTHONPATH=<repo> uv run python <repo>/scripts/human/export_deploy_human.py \
    --ckpt ~/supermimic_runs/logs/<run>/model_final.pt --motions <人体 npz 文件或目录> [--motion_list x.txt] --out <dir>
```

产出（`<dir>/`）：

| 文件 | 内容 |
|---|---|
| `policy.onnx` | 输入 `obs_hist [1,10,93]`（旧→新；每帧 q − default、dq、骨盆角速度、骨盆系重力、上一步动作）和 `goal [1,K,G]`，输出 `action [1,29]`（关节目标 = offset + scale × action）。归一化和 clamp 在图里 |
| `meta.json` | 关节顺序、默认姿态、action scale/offset、PD 增益、G1 SDK 索引映射、K / hist / 维度、`goal_layout`、归一化统计量、heads cfg；goal 构造所需的 FK 表（bone-map 连杆及其祖先）、heading/foot/task 连杆、踝高 |
| `clips/<name>.npz` | 每个 clip 的参考侧量：`h_feat`、`h_yaw`、`h_task_pos`、`h_task_rot6d`、`h_root_xy`、IK 重置姿态 `init_*`、fps。和训练缓存逐位相同（`derive_fields` / `derive_ee_fields`） |
| `deploy_goal.py` | `supermimic/human/deploy_goal.py` 的副本，只依赖 numpy |

**机载循环**（50 Hz；q 按 `meta["joint_names"]` 顺序，IMU 四元数 wxyz 为骨盆姿态）：

```python
from deploy_goal import HumanGoalBuilder
b = HumanGoalBuilder.from_files("meta.json", "clips/<name>.npz")   # alpha_rate=1.0 /s, realign_on_activate=True
b.start(f0, q, imu_quat, odom)            # odom = (x, y, yaw) 腿里程计骨盆位姿，或 None
for f in range(f0, T):
    goal = b.step(q, imu_quat, f, alpha_cmd, odom)                  # [1,K,G] float32
    action = sess.run(None, {"obs_hist": obs_hist, "goal": goal})[0]
```

- **对齐**：`start()` 按"在帧 f0 重置一个 episode"的方式把参考对到机器人上：骨盆 xy 对人体骨盆 xy，骨盆航向对该帧 IK 重置姿态的航向。之后 rel_yaw 和 EE 块都在这个"训练世界"里计算，与训练逐位一致（无漂移）。
- **航向来源**：有里程计时用里程计 yaw（IMU 只给 roll/pitch），没有时用 IMU 自己的 yaw。来源切换时重选偏置，使 rel_yaw 不跳变。
- **α**：`alpha_cmd` 被限速到 ≤ `alpha_rate`/s（训练中的切换斜坡是 0.5 s）。`odom is None` 时实际 α 立即置 0、`ee_valid = 0`、EE 块清零（训练里从不出现 α > 0 而 `ee_valid = 0`）。
- **EE 段开始时重锚**（`realign_on_activate`，默认开）：α 从 0 升起的那一步只重锚 xy，让世界目标从机器人当前位置出发，而不是从 H4 跟踪期间累积漂移后的位置；关掉即保持起始锚点（真正的固定世界坐标）。`realign()` 可随时手动重锚 xy + 航向。
- **骨盆高度**：EE 块需要 torso_link 的完整 6 自由度位姿。不传 `base_z` 时，用较低踝连杆按站立踝高（0.035 m）反推（脚放平时误差约 1 cm）。
- **耗时**：goal 构造约 0.4 ms/步，加 ONNX 推理约 1 ms/步（单核 CPU）。

**检查**：`tests/test_human_deploy.py`（H4 与 ee_mode）—— clip npz 等于训练缓存；numpy FK 等于 TorchFK；`goal_world` 等于 `HumanHeads.goal`（随机根位姿、关节、α、`ee_valid`、锚点，最大误差 ~1e-6）；在旋转平移过的里程计坐标系里逐步构造等于训练端 goal；α 限速、里程计丢失、重锚；ONNX 等于 torch。`SUPERMIMIC_CKPT=<model.pt>` 时用该检查点的策略。

## 6. 动作重新筛选（让 G1 真正能完成）

在现有 1881 对、8.0 h 的配对集上再筛一遍，产出新列表 `ee_list.txt`，`human_list.txt` 保持不动。

### 6.1 运动学可达（新脚本 `scripts/human/prepare_ee_ik.py`）

- **求解**：逐帧解 root 6 自由度加 29 个关节，目标是 §2.2 缩放后的 5 连杆位姿。
  - 约束：着地脚放平、贴地，关节限位，正则项拉向上一帧的解（论文的 IK 没有时间连续性，会在相邻帧之间翻转，这里加正则来避免）。
  - 用法：复用 `prepare_init_states.py` 的求解器框架，换成 EE 目标即可。
- **逐 clip 指标**：腕位置残差的 p95、torso 残差、脚残差、IK 解的关节速度峰值、着地脚的滑移。
- **筛选阈值（初值）**：腕 p95 < 5 cm，torso < 5 cm，脚 < 3 cm，关节速度在限值的 1.5 倍以内。
- **副产物**：IK 解写入 `init_ee_*`。α ≥ 0.5 的 episode 从它重置，这和论文"从 IK 姿态加扰动起步"一致。

### 6.2 规则过滤

- 已剔除的 horizontal / elevated / low_pelvis 继续剔除。
- 新增剔除：缩放后骨盆水平速度 > 1.5 m/s 的片段（世界系 EE 跟踪在快跑时意义不大，也难以实现）、手掌接触地面的片段、手到达头顶以上超过 G1 臂展的片段。

### 6.3 任务导向的子集加权

- 从 AMASS（KIT、CMU、BMLmovi 等）中按动作标签挑出搬运、伸手取物、弯腰拾取、踢、推、开门之类与 loco-manipulation 接近的片段。
- 在采样器上提高这部分的权重：复用 sampler 的 clip 权重，或用一个 `--clip_weights` 文件。

### 6.4 动力学可行（训练后）

- 用第一版检查点在 α = 1 下跑 eval，按 clip 统计失败率和腕误差。
- 持续失败的 clip 进入黑名单，与现有的 hard-clip 和 whitelist 流程一样。

## 7. 评测

- 在 `eval_paired.py` 和训练中评测里新增指标：torso、腕、脚的世界位置误差（cm）和旋转误差（°）；"腕误差 < 5 cm 的步占比"。
- 在 α ∈ {0, 0.5, 1} 三档上各跑一遍，画"骨方向误差 – 腕世界误差"的折中曲线。
- **验收**：
  - α = 0 时，相对 H4 检查点各项骨方向误差、失败数无明显回退（≤ 1° / ≤ 2 条）。
  - α = 1 时，腕世界误差显著下降（目标 < 5 cm 中位数）。
  - 加漂移增强与推力评测：40 N、持续 1 s，每 5–10 s 一次，与论文一致。

## 8. 改动清单（按实施顺序）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `human/motions.py` | §2.1 新字段，cache v3 |
| 2 | `human/heads.py` | `HumanHeadsCfg` 加 `ee_mode`（默认 False）、`w_ee_*`、`std_ee_*`、`term_ee_*`、`ee_scale`、`alpha_*`、`drift_*`；`kin` 增加 task 位姿；`goal`、`reward`、`termination`、`metrics` 加 α 混合；新增 `STEP_FIELDS` 和 `PACK_FIELDS` 条目，仅在 ee_mode 下启用 |
| 3 | `human/command.py` | 每个 env 的 `cond` 状态：α 采样和切换、锚点、漂移、重规划；按 α 选择重置姿态 |
| 4 | `human/fns.py`、`human/sim_interface.py` | `_act`、`_post`、`_imagine_step` 带上 cond；漂移外推 |
| 5 | `replay.py`、`trainer.py` | `cond` 字段（可选分配）；`_pack_args`；从 H4 graft 时扩展 goal（已有机制） |
| 6 | `scripts/human/prepare_ee_ik.py`、`filter_ee.py` | §6 |
| 7 | `scripts/human/export_deploy_human.py` + 部署端 goal 构造 | §5 |
| 8 | `scripts/human/launch_variants.sh` | 新增 ee 臂：`ee1` 从零训练，`ee2` 从 H4 检查点热启动 |
| 9 | `tests/test_human_heads.py` | ee_mode 下 α = 0 时奖励严格等于 H4；梯度有限（贴合、侧躺、倒立）；仿真和想象两端终止一致；漂移外推与 replay 一致 |

## 9. 风险与待定

- **采样器**：bin 回报的 EMA 混合了不同的 α。如果 α = 1 明显更难，采样会被它主导，必要时按 α 档位分桶。
- **世界模型的 xy 积分误差**：H = 48 时，想象 rollout 末端的 xy 误差需要实测，可以在 `wm/` 的 k 步诊断里加 `root_xy_err`。如果误差和窄核 6 cm 处于同一量级，就把窄核放宽到 0.08–0.1 m。
- **缩放的取舍**：k 缩放之后，"世界坐标"指的是缩小后的人体世界，不是录制现场的绝对坐标。要在真实场景里精确完成任务，需要在 G1 尺度下录制数据，并用 `ee_scale=1`。
- 论文没给的超参（终止阈值、DR 范围、惩罚权重）沿用 H4。

## 10. 实现状态（10-29）与初稿的差异

**已实现**（代码在本分支，`ee_mode=0` 时 H4 逐位不变，`tests/test_human_ee.py` 覆盖 α = 0 等于 H4）：

| 项 | 实现 |
|---|---|
| 缩放 | 不缩放（`ee_scale` 未实现）。人体任务点直接用世界坐标，只减去 episode 起始帧的人体骨盆 xy |
| torso 目标 | 只跟踪 **xy + 旋转，不跟 z**：AMASS 人体两肩中点高度中位 1.30 m（p10 1.19 m），G1 双腿伸直时只有 1.08 m，够不着；骨盆高度仍由 H4 的归一化高度项约束 |
| 腕 | 世界位置（宽核 0.3 m + 窄核 0.06 m）+ 世界旋转（0.15 rad） |
| 脚 | 相对 torso 点、在航向系下比较的 xy + 旋转（同 Workhorse 的思路：落脚位置在世界里自由）；摆动脚净空 10 cm |
| 权重 | `w_ee_*` 是相对权重，自动缩放到 (1 − ee_style) × H4 总和（`ee_task_scale`），所以任何 α、任何消融配置下 `reward_max_per_step` 都等于 H4 的 0.2 |
| 每片段可行性 | `ee_clip_list`：只有通过 `prepare_ee_ik.py` 筛选的片段才采样 α > 0，其余片段固定 α = 0。训练数据仍是 H4 的全量 1564 条，H4 能力不丢 |
| 漂移 | 有界的分段线性随机游走（盒子 ±10 cm / ±10°，速率每 0.2–1 s 重采样一次），**不归零**：模拟的是机载里程计误差，不是 Workhorse 那种重规划跳变 |
| 热启动 | 人体 goal 的 graft 已打开：H4 检查点 140 → 187 维，新列零初始化，起步就是 H4 |
| 筛选 | `scripts/human/prepare_ee_ik.py`：逐 5 帧做任务空间 IK（GPU 上批量 Adam），按腕、torso、脚的残差 p95、关节速度和骨盆速度筛选。本机 40 条试跑：腕残差 p95 中位 2.4 cm，主要淘汰原因是骨盆速度过快（跑跳）和数据跳变 |

**关于"不需要估计器"**：参考动作是给定的，不是规划出来的，所以 Workhorse 的"按 5 Hz 重新锚定规划块"确实不需要。但 EE 块是"目标在机器人当前 torso 系下的位姿"，计算它需要机器人在世界里的 xy 位置和航向。IMU 只给姿态，xy 位置至少要靠腿运动学 + IMU 的里程计积分。训练里的漂移增强就是在模拟这个积分误差。α = 0 且没有里程计时，可以把 EE 块清零（`ee_valid = 0`），这种模式训练里覆盖了。

**消融**（`scripts/human/launch_ee.sh`，tencent_bj9 八卡，各 40k 轮，在 ee_list 的 512 条片段上做 α = 1 评测）：

| 组 | 改动 |
|---|---|
| e1 | 默认配置，从零训练 |
| e2 | e1 从 H4（`ma1_human4` final）热启动，再训 40k |
| e3 | 窄核放宽（0.1 m / 0.25 rad） |
| e4 | 去掉腕部宽核（只有窄核，同 Workhorse） |
| e5 | ee_style 0.5 |
| e6 | 不加漂移 |
| e7 | 不加 EE 终止 |
| e8 | 只用 ee_list 里的片段训练 |

α = 0 端（H4 回退）用 `scripts/eval_ckpt.py --human_cfg eval_alpha=0` 离线评测。

**部署导出**：已实现，见 §5.1。

**待做**：α 偏高时用 EE-IK 姿态重置（目前用骨方向 IK 姿态重置，腕部初始误差约 14 cm）、`eval_paired.py` 的 α 档位。
