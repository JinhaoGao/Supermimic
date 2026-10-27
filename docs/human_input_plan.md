# 人体动作作为策略输入：Supermimic 实现方案（dev_gjh）

目标：在 Supermimic 上验证"策略直接以人体动作（SMPL）为输入、奖励直接对人体动作构建、全程不经过逐帧重定向"是否可行，以及它对重定向歧义片段是否有缓解。不追求与 Noetix-RGMT 的 HIM-RGMT / P1–P3 一致，只借用其中已验证的骨方向对应（bone map + T-pose 标定）。

范围：第一期只做"风格跟踪"（对应 RGMT P1 的思路，不含任务链接 / 漂移 / alpha）；结构上给第二期（度量任务目标、alpha）留接口。部署导出不在本期。

---

## 1. 现状：参考动作进入 Supermimic 的 4 个入口

世界模型只学机器人动力学，从不接触参考。参考只出现在：

| 入口 | 代码 | 现在 | 本期改成 |
|---|---|---|---|
| goal（policy / critic 输入） | `heads.goal`、`fns.goal_from_pack`、`sim_interface.goal_refs` | 参考关节角 / 速度 + 锚点相对 rot6d | 人体逐帧特征 + 人体/机器人相对航向 |
| 奖励（仿真端 + 想象端） | `heads.reward`、`fns._post / _imagine_step` | body 位置 / 朝向 / 速度（米、四元数） | 骨方向（全局 / 局部）、缩放根速度、yaw rate、归一化骨盆高度、航向 |
| 终止 | `heads.termination` + `env_cfg` 的 mjlab 终止项 | 锚点高度 / 倾斜 / 末端高度 / 根朝向 | 躯干骨方向误差、归一化骨盆高度误差（仿真与想象同一判据） |
| 重置 | `MotionLibrary.root_state` → `LibraryMotionCommand._resample_command` | 参考帧的机器人关节与根状态 | 离线骨方向 IK 的机器人姿态（只用于重置） |

不改动：世界模型、replay、DR / ξ、policy / critic 网络（仅 `goal_frame_dim` 变）、BPTT + λ-return、优化器与学习率、按 bin 回报的重置采样、看门狗、DDP、`torch.compile`。

## 2. 设计决定（带默认值，均可由开关覆盖）

**D1 参考表示：离线预计算，库里存"已经算好的人体量"。**
库不存原始 SMPL 参数，而存每帧直接可用的张量，goal 和奖励都只做 gather + 少量在线运算：

| 字段 | 形状 | 用途 |
|---|---|---|
| `h_feat` | [T, 138] | goal：骨盆系重力 3、航向系根速度 / 静息骨盆高 3、骨盆系角速度 3、骨盆高 / 静息高 1、21 关节局部 rot6d 126、足接触 2 |
| `h_dir` | [T, S, 3] | 奖励 / 终止：人体骨方向（人体航向系） |
| `h_dir_local` | [T, S, 3] | 奖励：人体骨方向（父关节系） |
| `h_root_vel` | [T, 3] | 奖励：人体航向系根速度（未缩放） |
| `h_yaw`、`h_yaw_rate` | [T]、[T] | 奖励：人体航向角（髋连线）与 yaw rate |
| `h_height` | [T] | 奖励 / 终止：骨盆高 / 人体静息骨盆高 |
| `h_contact` | [T, 2] | goal（已在 h_feat 内）；接触近似奖励（可选） |
| `k` | [T] | 机器人静息骨盆高 / 人体静息骨盆高（按片段常数，逐帧存便于 gather） |
| `init_*` | 关节位置 / 速度 [T, J]、根四元数 [T, 4]、根高度 [T] | 仅重置 |

S = 20 段骨（G1 bone map：躯干 4 + 每侧腿 4 + 臂 4）。每帧约 138·2 + 20·3·2·4 + 若干 ≈ 1.3 KB（fp16 速度类字段可再省），2264 条 / 约 10 h ≈ 2.3 GB，内存放得下。

**D2 goal：参考侧特征 + 一个相对航向项。**
每帧 goal = `h_feat`（138）+ `[cos Δψ, sin Δψ]`（2），Δψ = 人体航向 − 机器人航向（机器人航向由左右髋连杆在线算，可微）。理由：现在的 goal 里锚点相对 rot6d 是策略纠正航向累积误差的唯一信号；只给参考侧量的话航向误差不可观测。开关 `--human_goal_rel_yaw`（默认开）。K=10 帧窗口与现在一致。

**D3 奖励：全部由状态 s 经 TorchFK 可微得到，没有接触传感器依赖。**
骨方向误差用弦长平方 ‖d_h − d_r‖² = 2 − 2cos θ（≈ θ²），不用 `atan2` 角度：后者在 θ→0 处梯度奇异，BPTT 会出 NaN。

| 项 | 公式 | 默认权重 / std（起点取 RGMT P1） |
|---|---|---|
| `bone_global_torso` | 躯干组骨（航向系）exp(−‖Δd‖²/σ²) 加权平均 | 1.0 / 0.35 |
| `bone_local_{torso,leg,arm}` | 各骨在父连杆系 vs 人体父关节系（经标定 Q 映射） | 0.5 / 1.5 / 1.0，σ 0.35 |
| `root_vel` | 机器人骨盆航向系线速度 vs k·`h_root_vel` | 1.0 / 0.5 |
| `yaw_rate` | 骨盆 ω_z vs `h_yaw_rate` | 0.5 / 1.0 |
| `heading` | Δψ（D2） | 0.5 / 0.4（开关 `--w_heading`，0 = RGMT P1 行为） |
| `height` | 骨盆高 / 机器人静息高 vs `h_height` | 0.5 / 0.1 |
| `action_rate`、`joint_limit` | 不变 | −0.1、−2.0 |
| `contact`（可选） | 足踝离地高度的 sigmoid 作为软接触概率，与 `h_contact` 比较；slip = 软接触 × 足水平速度² | 默认关（`--w_contact`） |

所有项乘 dt（与现在一致）；`reward_max_per_step` 改成"正权重之和 × dt"，由新 heads 提供，看门狗 / 采样器 / 评测归一化自动跟着变。RGMT 的 `undesired_contacts` 不迁移（想象端没有接触）。

**D4 终止：仿真端与想象端同一函数。**
- 躯干：`pelvis_up` / `torso_up` 两段骨的航向系弦长对应角 > 1.0 rad；
- 高度：|骨盆高 / 机器人静息高 − `h_height`| > 0.3；
- 片段结束：`clip_end`（不变，time-out）。

想象端软终止按现有 noisy-OR + sigmoid 温度（沿用 0.05 / 0.02）。仿真端的 mjlab 终止项直接调用同一个 torch 函数（输入机器人 `body_link_pose_w` 中的 bone-map 连杆），保证 `train/term_mismatch` ≈ 0（已有的日志项，可直接当检验）。

**D5 重置：离线骨方向 IK，只用于重置。**
移植 RGMT 的 `prepare_smpl_init_states.py`：逐帧求骨盆 roll / pitch + 关节角使骨方向贴合人体，航向取人体髋连线，骨盆高度让最低足底落地。重置时根 xy 放在 env 原点（所有奖励都是航向系 / 尺度无关量，xy 无需对齐），根线速度 = k·v_human、角速度 = 人体。IK 姿态不进入输入、奖励、评测。备选（开关）：只从第 0 帧默认站姿开始（最纯粹，但无法中途采样，作对照用）。

**D6 机器人参考路径保持原样（bitwise）。**
全局开关 `--reference {robot,human}`，默认 `robot`。现有 16 个测试必须在改动后原样通过；另加一个回归测试确认 robot 模式的 goal / reward / termination 与改动前逐位一致。

## 3. 标定资产：可直接复用 RGMT 的 G1 标定

已核对：mjlab 的 G1（`asset_zoo/robots/unitree_g1`）与 RGMT 的 `g1.xml` 在 bone map 涉及的 11 个连杆上零位 FK 的位置与旋转完全一致，关节顺序也一致。所以 `smpl_bone_calib_g1.json`（robot_axes / human_axes / parent_q）可以直接拿来用。只有两项和碰撞几何有关，要用 Supermimic 实际训练的模型（`eval+evalcontact` 变体）重算：`robot_rest_pelvis_height`（足底到骨盆高度）和 IK 重置时的足底高度。

## 4. 代码改动清单

新增 `supermimic/human/`（与机器人参考路径隔离）：

| 文件 | 内容 | 约行数 |
|---|---|---|
| `human/bone_map.py` | 移植 `smpl_bone_map.py`（纯 torch）；`BoneMapTensors` 加 TorchFK 连杆下标 | 250（大部分照搬） |
| `human/calib_g1.json` | RGMT 标定 + 重算的静息高度 / 踝高 | — |
| `human/heads.py` | `HumanHeads`：与 `AnalyticHeads` 同接口（`kin / proprio / goal / reward / termination / state_features / goal_layout / reward_max_per_step`），`proprio` 直接复用 | 300 |
| `human/command.py` | `HumanMotionCommand`（mjlab `CommandTerm`）：当前帧缓存、IK 重置、clip_end；回报采样复用 `LibraryMotionCommand` 的采样代码（先把采样部分抽成 mixin，robot 路径行为不变） | 200 |
| `human/terms.py` | 仿真端终止项（调用 `HumanHeads` 的同一函数） | 50 |

修改：

| 文件 | 改动 |
|---|---|
| `motions.py` | 字段表参数化：`FIELD_DTYPES / NPZ_KEYS` → 由调用方传入 schema；人体 schema 不做关节 / 刚体名映射，缓存键包含 schema 名 |
| `fns.py` | 6 个参考张量（`p_q … p_bav`）→ 一个参考 dict；`goal_from_pack / _imagine_step / _post / _act` 改签名，由 heads 声明自己需要的字段（`heads.ref_fields`） |
| `trainer.py` | `update_policy` 的 pack 用 `heads.ref_fields`；`collect_step` 走 dict；`root_ori` / `ee` / `FailureWatch` 在 human 模式关闭；`_graft_inputs / _calibrate_goal_norm` 在 human 模式不适用（报错而不是静默） |
| `sim_interface.py` | `goal_refs / current_ref` 返回 dict；human 模式建 `HumanHeads`、FK 的 needed_bodies = bone-map 连杆 |
| `env_cfg.py` | `make_env_cfg(reference="human")`：换 command、去掉 mjlab 跟踪奖励（训练器不用）、换终止项 |
| `evaluate.py` | human 指标：失败数、归一化回报、骨方向误差（全局躯干 / 局部腿 / 局部臂，度）、根速度误差、高度误差、航向误差 |
| `scripts/train.py` | `--reference human`、`--human_*` 奖励 / 终止开关；cfg 写进 ckpt，`_resume_diffs` 报告 |

新增脚本（离线，跑在 `noetix_gmt` conda 环境，或给 venv 装 `smplx`）：

| 脚本 | 内容 |
|---|---|
| `scripts/human/prepare_smpl_motion.py` | AMASS → 人体 npz（移植 RGMT，加上 D1 的预计算字段） |
| `scripts/human/prepare_init_states.py` | 骨方向 IK 重置状态（移植，改用 mjlab G1 模型） |
| `scripts/human/pair_with_retarget.py` | 人体 npz 与 `g1_amass_retarget_npz` 逐帧对齐（见 §5），输出配对清单 |
| `scripts/human/eval_human_metrics.py` | 对任一策略（robot 或 human 模式训练的）在配对片段上算人体侧指标，用于公平对比 |

合计新增 / 修改约 1000 行。

测试（`tests/`，CPU）：bone map 在 T-pose 上误差为 0；`HumanHeads` 的奖励 / 终止对 s 的梯度有限（含 θ=0、躺倒、髋连线竖直等边界）；仿真端终止项与想象端硬终止逐位一致；robot 模式回归（D6）。

## 5. 数据

- 人体：`~/AMASS`（ACCAD / CMU / SFU，2402 条 stageii），SMPL-X 模型 `~/models/smplx_v1_1`。
- 配对的重定向数据（基线用）：`~/datasets/smplkp_full/g1_amass_retarget_npz`（2264 条，50 Hz，G1 37 刚体）。
- 歧义子集（实施时修正）：质量报告里的 `kp_vs_rt` 标记几乎都和 `horizontal / elevated / low_pelvis` 同时出现，这些片段平地上做不了，两臂都剔除，所以这个标记用不上。改用连续列 `kp_vs_retarget_err_m`：配对片段里排前 10% 的作为歧义子集（`ambig_top.txt`）。
- 对齐（实施时修正）：重定向管线是"源 → 30 Hz → 重定向 → 50 Hz 插值"，人体 npz 是"源 → 50 Hz"，抽样里约一半片段相差 1–4 帧。训练不受影响，因为各臂只用自己的数据。`pair_with_retarget.py` 用骨盆水平速度的互相关求偏移，写进 `pairs.tsv`，`eval_paired.py` 按"人体帧 = 机器人帧 + lag"对齐。相关 < 0.8 的片段（多是手势 / 站姿这类几乎不走动的片段，速度信号只是噪声）照样进训练，但标为未对齐，配对评测只在已对齐的片段上打分。全量结果：1881 对（8.0 h），1586 条已对齐，歧义子集 204 条。
- 先用 RGMT 已有的 15 条 pilot（`~/datasets/smpl_motion_pilot_50hz`，已含 IK 重置状态）打通流程。

## 6. 里程碑

| 阶段 | 内容 | 验收 |
|---|---|---|
| M0 | 管线重构：参考 dict、motions schema、采样 mixin（仍只有 robot 模式） | 16 个旧测试 + robot 回归测试通过；LAFAN1 短训练（2k 轮）曲线与 main 一致 |
| M1 | `human/` 模块 + 离线脚本 + 单元测试 | T-pose 误差 0；梯度测试通过 |
| M2 | 15 条 pilot 上训练 human 模式 | `term_mismatch` ≈ 0；想象奖励与真实奖励同量级；能稳定站立并跟上走路类片段；肉眼回放动作像人体参考 |
| M3 | 全量配对数据：基线（robot 模式，重定向参考）vs human 模式，同片段、同轮数、同 H / 网络 | 两者都用 `eval_human_metrics.py` 评：失败率、骨方向误差、根速度误差；单列 `kp_vs_rt` 歧义子集 |
| M4（可选） | 消融：去掉相对航向 goal / 去掉 heading 奖励 / 开接触近似 / 第 0 帧重置 | 看各项对失败率与骨误差的影响 |

M0 是纯重构，可先合入 dev_gjh 并单独验收；M1 与 M0 互不依赖，可以并行写。

## 7. 主要风险

- **航向漂移**：RGMT P1 不奖励绝对航向，只奖励 yaw rate。Supermimic 想象 H=48 步，航向误差会在 rollout 内累积。D2 / D3 的航向项默认打开，消融时再关。
- **骨方向奖励的梯度质量**：弦长形式可微，但躺倒 / 倒立时髋连线接近竖直，航向帧会退化。航向帧加 eps，并在梯度测试里覆盖这种情况。
- **奖励尺度**：项数和权重都与 BeyondMimic 不同。看门狗按 `reward_max_per_step` 归一化，阈值（`wd_drop` 0.2）可能要重调。先用 bj27 单卡那套风格（`--wd_sustain 200 --wd_warmup 1500 --wd_signal imagined_reward_robust`）避免误回滚。
- **IK 重置质量**：IK 误差大的帧（RGMT 有 `init_g1_ik_err_deg`）可能导致重置后立刻摔倒，拖累采样器。重置时跳过误差超过阈值的帧（`bin` 级屏蔽）。
- **基线对比的公平性**：基线的奖励目标是重定向结果，用人体侧指标评它，对它略不利；所以 M3 同时报告机器人侧 MPJPE（只对基线有意义）作参考，结论以失败率 + 歧义子集为主。
