# SuperMimic 复现与扩展（mjlab / MuJoCo Warp）

论文 *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*（Liao et al., arXiv:2610.28480, CoRL 2026）的复现，基于 [mjlab](https://github.com/mujocolab/mjlab)（固定在 commit `8bb2bc9`，2026-10-18）。论文代码未公开，这里的实现来自论文正文 + mjlab 的 BeyondMimic 跟踪任务（论文附录 A 与它逐项对应）。

在复现之上做了团队需要的扩展：MotionAtlas 全量数据（G1 25.5 万条 / E2 37 万条）、G1 + E2 两个机器人、单机多卡（torchrun）、团队 MuJoCo 评测站对接、能上真机的纯本体感知策略与 unitree_mujoco 部署链路。

> 部署口径：上榜和上真机的策略只用关节编码器、IMU 和参考动作，**不需要里程计**。goal 里的 `ref_root` 段只取参考动作本身的量（参考躯干高度 / 速度 / 重力方向），符合这个口径；`--goal_anchor_pos`（参考锚点相对机器人位置）需要里程计，不能上榜。

## 当前结果（2026-10-26）

团队评测站 MotionAtlas v4 官方协议（每条从第 0 帧开始，失败 = 根高度 / 根全角 / 末端高度任一超阈值）：

| 机器人 | 策略 | v4 成功率 | progress | 备注 |
|---|---|---|---|---|
| G1 | `g1v2_floor10_r25` model_208000 | **91.70%** | 95.36% | 站点 run 669a45a8，与离线评测逐位一致；榜 #4/307（榜首 91.83%，见下） |
| E2 | `e2v2_rorootfloor_r25` model_156000 | **89.91%** | 94.35% | 站点 run 0b0ce387；E2 榜 #34/295（榜首团队 GMT 集群 90.95%） |
| E2 | `e2v2_roroot_r26` model_164000 | 90.17% | 94.49% | 离线（同协议），未上站 |
| G1 | v1（论文配方）model_32000 | 67.80% | — | 起点：论文去掉了末端终止，评测站按末端高度判失败 |

G1 榜首 91.83% 是 bj31 8 卡数据并行线（骨盆朝向终止，model_174000，本仓库合并进来的 ddp-bj31-v2 分支训练）；它还没有用上 `--goal_ref_root`，在单卡线上这一项在骨盆终止之上 +0.5pp。

论文对照实验（LAFAN1 G1 40 条，2.45 h，单卡 RTX PRO 5000，论文完整配置）：2.44 h / 16000 轮达到 benchmark 回报 0.92、40 条 0 失败；论文报告的是 SuperMimic 1.2 h（H100）0.82、PPO-Sim 18 h（8 卡）0.72。

## 目录

| 路径 | 内容 |
|---|---|
| `supermimic/trainer.py` | 算法 A.1：采集 → 拟合世界模型 → 想象 rollout 上 BPTT 更新策略 + λ-return critic；看门狗 / NaN 守卫 / 提前停止；输入嫁接 |
| `supermimic/world_model.py` | 动力学 MLP（3×2048，航向不变或全本体系 `--wm_body_frame`），半隐式积分 + 残差，k 步自回归损失 |
| `supermimic/heads.py` | 解析的观测 / goal / 奖励 / 终止头（torch FK，可微）；软终止 noisy-OR |
| `supermimic/policy.py` | goal 编码器（K=10 帧 → z∈R³²）+ 解码器（10 帧本体历史）+ critic |
| `supermimic/fns.py` | 每步函数（`torch.compile`）：采集动作、想象一步、critic 输入 |
| `supermimic/library_command.py` | 多片段参考命令、按回报的重置采样（附录 D.1 + 钳位 / 封顶 / 聚类先验）、终止项 |
| `supermimic/motions.py`、`bin_clusters.py` | 动作库（全部片段拼成一条时间线，内存映射缓存）；bin 运动学特征 + k-means |
| `supermimic/env_cfg.py`、`robots/` | mjlab 环境（论文的改动 + 我们的终止开关）；G1 `eval/evalcontact`、E2 `nodamp` 机器人变体 |
| `supermimic/distributed.py`、`watch.py` | torchrun 数据并行（梯度 / 归一化器 / 采样器同步、跨卡状态校验）；常开的失败监控 |
| `supermimic/evaluate.py` | 训练内 benchmark（附录 F 协议） |
| `scripts/train.py` | 训练入口（单卡直接跑，多卡用 torchrun） |
| `scripts/build_cache.py`、`extend_cache.py` | 预建 / 追加动作库缓存 |
| `scripts/export_bundle.py`、`export_eval_onnx.py`、`export_deploy.py`、`deploy_from_server.sh` | 导出：回放包（ONNX + metadata）、评测站 ONNX、unitree_mujoco 部署文件 |
| `scripts/play_mujoco.py`、`view_local.sh` | sim2sim：原版 CPU MuJoCo + onnxruntime 回放 / 录视频 |
| `scripts/eval_ckpt.py`、`check_fk.py`、`wm_diag.py`、`avg_ckpt.py` | 离线评测、FK 校验、世界模型诊断、权重平均 |
| `scripts/launch_*.sh` | 集群启动脚本（bj27 单卡、bj31 多卡；路径用环境变量覆盖） |
| `eval_site/` | 团队评测站插件 `eval_supermimic.py`、站点补丁、离线 v4 评测器 `eval_r18.sh`、提交脚本 `submit_v4.sh`、失败分析工具 |
| `tests/` | CPU 单元测试（14 个文件，见"测试"） |
| `docs/README_legacy_en.md` | 之前的英文 README（逐轮日志、数据并行实现细节、看门狗事故记录） |

## 安装

```bash
bash scripts/setup_env.sh ~/supermimic        # 克隆 mjlab 到固定 commit，uv 建 venv（torch cu130 + mujoco-warp），以 editable 方式注册本仓库（~/supermimic/supermimic）
cd ~/supermimic/mjlab                           # 后续命令都在 mjlab 的 uv 工程里跑
```

同一台机器上并存多份代码时，用 `PYTHONPATH=<仓库目录>` 覆盖 editable 安装（集群上就是这么做的，比如 `PYTHONPATH=~/supermimic/supermimic_repro_c1`）。setup 脚本里把 PyPI 换成腾讯镜像是为集群网络准备的，其他机器可去掉。

## 数据

- 动作文件：50 Hz 的 npz（GMT / BeyondMimic 格式）：`fps, joint_names, joint_pos, joint_vel, body_names, body_pos_w, body_quat_w (wxyz), body_lin_vel_w, body_ang_vel_w`。关节 / 刚体按名字映射到机器人顺序。
- 动作库：第一次用某个文件列表时建缓存（每个字段一个 `.npy` + `clips.json` / `meta.json`，内存映射，多进程共享一份页缓存）。全量库很大（G1 1477 h ≈ 195 GB），建议先 `scripts/build_cache.py` 预建；追加片段用 `scripts/extend_cache.py`（旧 bin 下标不变，采样器统计可续用）。
- 选数据：`--motion_dirs <目录或 npz ...> [--blacklist bad.json] [--name_exclude ...]`，或 `--motion_list <每行一个 npz 路径>`。我们的全量列表 = MotionAtlas 去掉 GMT 黑名单（G1 另加 phc 层功夫片段，255,008 条）；列表文件在训练机上，不入库。
- 采样 bin：每条片段按 1 s 切 bin，重置时按 bin 回报采样（见"主要开关 · 采样器"）。

## 训练

单卡（论文对照，LAFAN1）：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python ~/supermimic/supermimic/scripts/train.py \
  --motion_dirs <lafan1 npz 目录> --run_name lafan1 --log_root ~/supermimic/logs --cache_dir ~/supermimic/cache
```

G1 全量，当前最好那条线的参数（单卡 RTX PRO 5000，约 0.7 s/轮）：

```bash
SW=noetix_kungfu=10,noetix_kungfu_dance_show=15,sportsmeeting=10,lafan1=15,g1_all=10,mixamo_cn=2,virtual_point=3
NW=tengkong=200,kongfan=200,shoufan=200,erqijiao=200,danpaijiao=200,xuanfeng=200,waibailian=100,lihetui=100,saotang=100,ceti=100,houti=100,zhengti=100
FW=getup=20,get_up=20,stand_up=20,ground_recovery=20,prone=20,lying=20,lie_getup=20,fallandgetup=20,breaking=20,crawl=5,kneel=5
uv run python ~/supermimic/supermimic/scripts/train.py --motion_list g1_list.txt --robot g1 --robot_variant eval+evalcontact \
  --source_weights $SW --name_weights $NW,$FW --sampler_clamp_returns \
  --ee_term_z 0.25 --root_ori_term 1.0 --root_ori_term_temp 0.05 --goal_ref_root --H 48 --fixes --num_workers 64 \
  --wd_sustain 200 --wd_warmup 1500 --wd_baseline_after_warmup --wd_signal imagined_reward_robust \
  --run_name g1_full --log_root logs --cache_dir cache [--resume model_N.pt --resume_lr_scale 0.25 --lr 5e-5]
```

这条线不是一次从零训出来的：v1（论文配方）→ 加末端终止 / 评测站模型 / 来源先验 → 难片段课程 → 采样器钳位 → 骨盆朝向终止 → goal 根部运动 → 地面动作加权，逐轮续训到 208000 轮（后期 `--lr 5e-5`，运行时 lr_scale 0.25）。完整参数的从零训练还没整段跑过。

E2 全量（`e2v2_rorootfloor_r25` 那条线）：`--robot e2 --robot_variant nodamp+evalcontact --motion_list e2_list.txt --source_weights $SW --name_weights $NW,$FW --sampler_clamp_returns --ee_term_z 0.25 --root_ori_term 1.0 --root_ori_term_temp 0.05 --goal_ref_root --H 48 --lr 2e-4 --fixes`（E2 根刚体是 `base_link`，末端是踝 + 肘）。

多卡（单机 torchrun；每卡保持论文的 4096 env / 4096 rollouts，全局 batch 随卡数放大）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29521 NPROC=4 ROOT=/data/supermimic VENV=~/supermimic/mjlab/.venv \
  bash scripts/launch_bj31_ddp.sh <run_name> <train.py 参数 ...>
bash scripts/stop_bj31_ddp.sh <run_name>
```

同一台机器跑两个多卡任务要各用一组 GPU 和各自的 `PORT`；控制台日志只有 rank 0，每个 rank 的完整输出在 `$ROOT/runs/<run>_ranks/`。

续训要点：
- `--resume` 恢复权重、优化器、采样器统计和运行时 lr_scale（看门狗每次回滚减半，会沿续训链累积）；`--resume_lr_scale X` 覆盖它。`--lr_scale` 是另一回事：基础学习率的倍数（多卡放大 batch 时用）。
- 新加 policy / critic 输入（`--goal_ref_root`、`--critic_errors`）续训时自动嫁接：旧列原位、新列零初始化、Adam 动量跟着搬，新 goal 段的归一化统计从 65k 库帧校准，嫁接后网络输出与原来一致。
- 世界模型表示换成本体系（`--wm_body_frame`）续训时也会嫁接，配合 `--policy_delay N --wm_delay_lr 6` 先只训世界模型。

## 主要开关

| 类别 | 开关 | 说明 / 结论 |
|---|---|---|
| 想象 | `--H 48`、`--rollouts`、`--env_steps_per_iter` | 想象步数（论文 32）；每次更新采样的仿真步数（2.5 试过：无收益且慢 15%） |
| 终止 | `--ee_term_z 0.25` 或 `--ee_termination/--no_ee_termination --ee_threshold` | 末端高度终止（论文去掉了，评测站要）；默认开 0.25 m |
| | `--root_ori_term 1.0 --root_ori_term_temp 0.05 [--root_ori_body]` | 根（G1 骨盆）全角含 yaw 终止 = 评测站 root_ori 判据；**有效**（bj31 线 +1.1pp，单卡线 +0.2~0.3） |
| | `--root_z_term 0.25` | 根高度终止 = 评测站 root_pos 判据（训练原来判躯干高度）；10-26 新加，待 r28 验证 |
| goal / critic | `--goal_ref_root` | goal 每帧加参考躯干高度 + 本体系线/角速度 + 重力（10 维，只用参考侧，可部署）；**有效**（骨盆终止之上 +0.5pp，E2 +0.4） |
| | `--critic_errors` | critic 加特权跟踪误差特征；无效 |
| 世界模型 | `--wm_body_frame`、`--policy_delay`、`--policy_freeze`、`--wm_delay_lr` | 本体系 WM（去掉俯仰 ±90° 的航向奇异）；无收益，翻腾失败阶段不变 |
| 采样器 | `--source_weights`、`--name_weights` | 来源 / 关键词先验（只作用在均匀份额上） |
| | `--sampler_clamp_returns` | bin 回报钳到 [0, g_max]；**有效**（+0.8，防片尾负奖励吸走全部份额） |
| | `--sampler_cap 50` | softmax 份额每 bin 封顶 50×均匀（allmimic）；G1 r25 时一半重置落在 0.03% 的 bin 上；待 r28 验证 |
| | `--sampler_clusters 256 --sampler_cluster_alpha 0.3 --sampler_cluster_mix 0.5` | 12 个运动学特征 k-means 聚类，簇份额 ∝ n^0.3，与来源先验混合；待 r28 验证 |
| | `--sampler_boot_mean` | 片尾超时用平均奖励自举（替代 r_last） |
| | `--hard_names --hard_term_schedule` | 难片段放宽终止的课程；短课程无收益 |
| 修复 | `--fixes` | 开源对照发现的六项：输入钳位 ±50σ、价值钳位、critic 按存活加权、想象片尾截断 + 自举、WM 前缀掩码、观测用干净动作；整体在噪声内，最好的模型都开着 |
| 看门狗 | `--wd_sustain`、`--wd_warmup`、`--wd_baseline_after_warmup`、`--wd_signal`、`--wd_reward_floor`、`--wd_nf_patience` | 想象奖励塌缩 / critic 损失爆炸 / 连续非有限时回滚到健康 checkpoint 并减半 LR；默认 = 多卡分支行为（立即回滚），单卡线用括号里那组 |
| 提前停止 | `--es_eval_fail_ratio 1.5 --es_patience 2` | 训练内评测失败数连续超中位数 1.5 倍即停，另存 `model_best_eval.pt` |
| 学习率 | `--lr`、`--lr_decay_iters`、`--lr_scale`、`--resume_lr_scale` | 余弦到 25 万轮剩 3%；后期调高 LR 有害，退火无收益 |

## 评测

- 训练内评测：`--eval_interval 2000 --eval_clip_names <名单>`（附录 F 协议，固定子集）。
- 离线评测某个 checkpoint：`uv run python scripts/eval_ckpt.py --ckpt model_N.pt ...`。
- 团队评测站（NOetix_GMT_eval，MotionAtlas v4 官方）：
  - 插件 `eval_site/eval_supermimic.py`（站点里按 sha256 钉住）；站点补丁 `eval_site/patch_site.py`（v1：goal 2nj+6）和 `patch_site_v2.py`（v2：goal 2nj+16，观测配置 `supermimic_g1_v2` / `e2_supermimic_v2`）。
  - 导出评测用 ONNX：`scripts/export_eval_onnx.py --bundle <bundle> --metric_bodies ... --damping actuator`。
  - 一条命令导出 + 上传 + 建官方评测：`bash eval_site/submit_v4.sh <run> <轮数> "<名字>"`（按 ONNX metadata 的 goal_layout 自动选 v1/v2 配置）；`OFFLINE=1` 改为在评测机上跑私有离线 v4（`eval_site/eval_r18.sh`，与站点逐条一致，不上榜）。
  - 分析工具：`whitelist_breakdown.py`（按来源 / 白名单拆分）、`flip_fail_phase.py`（翻腾失败发生在第几次倒立）、`cmp_by_source.py`、`kungfu_fail_modes.py` 等。

## 导出与部署

```bash
uv run python scripts/export_bundle.py --ckpt model_N.pt --motion <任一 npz> --out bundle_N      # 编译好的场景 + ONNX(metadata) + meta.json
python scripts/play_mujoco.py --bundle bundle_N --motion <clip.npz> --seconds 20 --viewer        # 原版 MuJoCo sim2sim（或 --video out.mp4）
SERVER=<训练机 ssh 名> REPRO=<训练机上的代码目录> bash scripts/deploy_from_server.sh <run> <轮数|best|latest>   # → unitree_mujoco 部署文件
```

unitree_mujoco（G1）里有两个 FSM 模式，按 ONNX 的 goal 格式区分，`export_deploy.py` 自动写到对应目录：
- `SuperMimic`（Velocity 里 RB + 左）：goal 每帧 2nj+6，`config/policy/gmt/supermimic/`。
- `SuperMimicV2`（RB + 右）：goal 带参考根部运动（2nj+16），`config/policy/gmt/supermimic_v2/`，链里带 `body_pos` 用来算参考躯干高度。

两种模式都支持 offline（播 npz）和 online（Redis 在线重定向流），启动时核对 ONNX 形状与 metadata。部署说明和仿真验证在 unitree_mujoco 仓库的 `deploy/robots/g1_29dof/SUPERMIMIC_DEPLOY_NOTES.md`。注意：参考躯干高度是绝对值，在线流的地面必须在 z = 0。

## 实验结论（G1 / E2 全量，v4 成功率）

有效：末端终止 + 评测站机器人模型（v1 67.8% → v2 85.7%）、难片段课程（腾空 1/13 → 13/13）、采样器钳位（+0.8）、骨盆朝向终止（+0.2~1.1）、goal 根部运动（+0.5）、地面动作加权（长程 +0.3~0.4）。

无效或噪声内：开源对照六项修复、critic 特权误差、本体系世界模型、UTD 2.5、骨盆终止收紧到 0.8、短难片段课程、checkpoint 权重平均、后期提高 / 降低学习率。

翻腾：失败基本都发生在参考第一次倒立后 0.6 s 内（18/21），换本体系世界模型不变；r28 用采样器封顶 + 聚类和"只训翻腾"的诊断实验区分是数据被稀释还是配方本身的问题。

单 checkpoint 的 v4 噪声约 ±0.3~0.5pp，比较配方要看多个点的均值。

## 与论文的差异 / 论文没说清的选择

- 世界模型状态 = 根位姿 + 根速度 + 关节位置 / 速度；奖励、观测、终止头都通过 torch FK 解析计算。
- goal 每个未来帧 = 参考关节角 / 速度 + 参考锚点（躯干）相对机器人锚点的姿态（rot6d），K = 10；可选 `ref_root`。
- 想象里的终止是硬阈值的软化版（sigmoid 的 noisy-OR，逐步 hazard：H = 48 步时恒定误差在"阈值 − 4.23×温度"处存活 50%）。
- "update-to-data ratio 0.4" 有歧义，`--env_steps_per_iter` 控制每次更新采集的仿真步数，默认 1。
- 采样器的片尾超时回报默认用 r_last/(1−γ) 自举（`--sampler_boot_mean` 改用平均奖励）。
- `torch.compile` 默认模式（`reduce-overhead` 的 CUDA graph 遇到输出别名会失败）。

## 测试

```bash
for t in tests/*.py; do PYTHONPATH=. python "$t" || echo "FAIL $t"; done      # CPU，torch（+ mujoco）即可，mjlab 缺失时自动打桩
```

覆盖：数据并行同步（3 个 gloo rank）、末端 / 骨盆朝向 / 骨盆高度终止、看门狗与 NaN 守卫、提前停止、采样器钳位 / 封顶 / 聚类、master 与多卡分支合并后的逐位一致性、输入嫁接与世界模型嫁接、旧代码能读新 checkpoint 的配置。

## 分支

- `master`：本仓库主线（单卡 + 多卡合一）。
- `terrain`：地形感知跟踪（高度图 / 深度相机 / 雷达学生、梅花桩等地形），在另一条线上持续开发，尚未合并。
- `vel-elegant`：速度指令模式（隐藏参考 + 摇杆指令），E2 速度控制用，尚未合并。
- `ddp-bj31`、`ddp-bj31-v2`：多卡分支，已全部并入 master。

## 已知问题

- 翻腾（空翻 / 手翻）仍是主要失败来源，见上。
- mujoco_warp 的求解器溢出警告：它在 CUDA graph 捕获时就编进内核，现在置 0 后重新捕获（`SimInterface`），日志不再刷屏。
- 集群脚本里的路径和机器名（bj27 / bj31、`/data2`、`/DATA/disk0`）是我们集群的；启动脚本的路径都能用环境变量覆盖。
