# SuperMimic 复现与扩展（mjlab / MuJoCo Warp）

论文 *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*（Liao et al., arXiv:2610.28480, CoRL 2026）的复现，基于 [mjlab](https://github.com/mujocolab/mjlab)（固定在 commit `8bb2bc9`，2026-10-18）。论文代码未公开，这里的实现来自论文正文 + mjlab 的 BeyondMimic 跟踪任务（论文附录 A 与它逐项对应）。

在复现之上做了团队需要的扩展：MotionAtlas 全量数据（G1 25.5 万条 / E2 37 万条）、G1 + E2 两个机器人、单机多卡（torchrun）、团队 MuJoCo 评测站对接、能上真机的纯本体感知策略与 unitree_mujoco 部署链路。

> 部署口径：上榜和上真机的策略只用关节编码器、IMU 和参考动作，**不需要里程计**。goal 里的 `ref_root` 段只取参考动作本身的量（参考躯干高度 / 速度 / 重力方向），符合这个口径；`--goal_anchor_pos`（参考锚点相对机器人位置）需要里程计，不能上榜。

## 当前结果（2026-10-26）

团队评测站 MotionAtlas v4 官方协议（每条从第 0 帧开始，失败 = 根高度 / 根全角 / 末端高度任一超阈值）：

| 机器人 | 策略 | v4 成功率 | progress | 备注 |
|---|---|---|---|---|
| G1 | `g1v2_ro10_8g_168k` model_174000（bj31 8 卡） | **91.83%** | — | 站点 run b43fdd83，**榜 #1**；骨盆朝向终止，goal 64 维（无 ref_root）；MPJPE 0.0851 |
| G1 | `g1v2_floor10_r25` model_208000（bj27 单卡） | **91.70%** | 95.36% | 站点 run 669a45a8，与离线评测逐位一致；骨盆终止 + ref_root + 地面加权 |
| E2 | `e2v2_rorootfloor_r25` model_156000 | **89.91%** | 94.35% | 站点 run 0b0ce387；E2 榜 #34/295（榜首团队 GMT 集群 90.95%） |
| E2 | `e2v2_roroot_r26` model_164000 | 90.17% | 94.49% | 离线（同协议），未上站 |
| G1 | v1（论文配方）model_32000 | 67.80% | — | 起点：论文去掉了末端终止，评测站按末端高度判失败 |

两条 G1 线的配方不同：8 卡线（ddp-bj31-v2 分支，已并入）只开了骨盆朝向终止；单卡线另外加了 `--goal_ref_root` 和地面动作加权（单卡线上在骨盆终止之上 +0.5 / +0.3pp）。两条线成功段的并集上限 94.67%，会的段不完全重合；8 卡线的移植估计认为这两项在它那条线上收益不大，所以要逐项做同起点对照，不要整套照搬。8 卡线的完整经验和对照数据见 `docs/bj31_8gpu_handoff_1026/`（交接总结 + 合并核对清单）。

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
| `tests/` | CPU 单元测试（16 个文件，见"测试"） |
| `tools/` | 两条训练线的诊断和 bj31 运维脚本（采样器探针、退化 / 骨盆 / root_pos 逐帧复跑、v4 差距分析、自动导出上站），见 `tools/README.md` |
| `docs/README_legacy_en.md` | 之前的英文 README（逐轮日志、数据并行实现细节、看门狗事故记录） |
| `docs/bj31_8gpu_handoff_1026/` | bj31 8 卡线交接：经验总结（配方、对照、退化 / 骨盆 / root_pos 诊断、运维坑）和合并核对清单 |

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

同一台机器跑两个多卡任务要各用一组 GPU 和各自的 `PORT`；控制台日志只有 rank 0，每个 rank 的完整输出在 `$ROOT/runs/<run>_ranks/`。8 卡 H48 约 0.89~0.95 s/轮（全局 36k env-step/s），每轮约相当于单卡的 1.6 倍样本效率（经验估计）；LR 不随卡数缩放。

bj31 8 卡线（G1 91.83%）的生产参数（从 model_174000 续训；`--iterations` 是绝对终点）：

```bash
NPROC=8 bash scripts/launch_bj31_ddp.sh <run> --num_threads 24 --motion_list <g1 列表> --robot g1 --robot_variant eval+evalcontact \
  --ee_term_z 0.25 --source_weights $SW --name_weights $NW --sampler_clamp_returns --num_envs 4096 --rollouts 4096 --H 48 \
  --wm_batches 10 --wm_windows 2048 --wm_k 8 --lr 8.9252e-4 --lr_decay_iters 250000 --iterations <终点> --seed 0 \
  --wd_drop 0.2 --wd_reward_floor -1.0 --wd_signal imagined_reward_alive --num_workers 64 \
  --eval_clip_names <1024 条评测名单> --eval_interval 2000 --eval_envs 1024 --es_eval_fail_ratio 1.5 --es_patience 2 \
  --ckpt_interval 2000 --log_interval 50 --param_check_interval 1000 --root_ori_term 1.0 --root_ori_term_temp 0.05 \
  --resume warmstart_ori8g_model_174000.pt
```

**两套看门狗风格**（合并后默认值随 8 卡分支：`imagined_reward_norm`、warmup 500、一触发就回滚；两种风格都要显式写全）：
- bj31 8 卡线：`--wd_signal imagined_reward_alive --wd_reward_floor -1.0`（不加 alive + floor 时，norm 会被死后发散的世界模型拖到 −17 / −35 造成误回滚）。
- bj27 单卡线：`--wd_sustain 200 --wd_warmup 1500 --wd_baseline_after_warmup --wd_signal imagined_reward_robust`。

**学习率**：实际 policy LR = `--lr` × ckpt 里继承的运行时 lr_scale × λ(it)，λ(t) = 0.03 + 0.97·½(1 + cos(π·t/250000))（λ(160k) = 0.309，λ(174k) = 0.235，λ(208k) = 0.096）。看门狗每回滚一次 lr_scale 减半，写进之后所有 ckpt：8 卡线的 ckpt 带 0.0625（8.9252e-4 × 0.0625 × 0.2349 = 1.31e-5），单卡 G1 线带 0.25。陷阱：在继承了 1/16 的 ckpt 上照旧写 `--lr 5e-5`，实际 LR 只有约 5e-7（bj27 r16–r18 掉进过）。每次续训后先看日志里的 `resumed … (policy lr X = lr x lr_scale Y x cosine Z)` 这一行。

续训要点：
- `--resume` 恢复权重、优化器、采样器统计和运行时 lr_scale；`--resume_lr_scale X` 覆盖运行时 lr_scale，`--base_lr_scale` 是基础学习率倍数。**`--lr_scale` 已删除**（两个分支里它的含义不同，旧脚本会静默多乘一次），用了会直接报错。
- 续训时 `_resume_diffs` 会对和 ckpt 不同的设置逐条打 WARNING（机器人 / 先验 / 钳位 / 课程 / H / heads / 片段表 / `input_clip` 与 `--fixes` 各项 / 执行噪声 / UTD / 世界模型坐标系 / 采样器封顶与聚类），续训后先看这些行。注意末端终止默认开（0.25），EE 关的旧 ckpt 不加旗标续训会被打开。
- 新加 policy / critic 输入（`--goal_ref_root`、`--critic_errors`）续训时自动嫁接：旧列原位、新列零初始化、Adam 动量跟着搬，新 goal 段的归一化统计从 65k 库帧校准，嫁接后网络输出与原来一致。
- 世界模型表示换成本体系（`--wm_body_frame`）续训时也会嫁接，配合 `--policy_delay N --wm_delay_lr 6` 先只训世界模型。

## 主要开关

| 类别 | 开关 | 说明 / 结论 |
|---|---|---|
| 想象 | `--H 48`、`--rollouts`、`--env_steps_per_iter` | 想象步数（论文 32）；每次更新采样的仿真步数（2.5 试过：无收益且慢 15%） |
| 终止 | `--ee_term_z 0.25` 或 `--ee_termination/--no_ee_termination --ee_threshold` | 末端高度终止（论文去掉了，评测站要）；默认开 0.25 m |
| | `--root_ori_term 1.0 --root_ori_term_temp 0.05 [--root_ori_body]` | 根（G1 骨盆）全角含 yaw 终止 = 评测站 root_ori 判据；**有效**（bj31 线 +1.1pp，单卡线 +0.2~0.3） |
| | `--root_z_term 0.25` | 根高度终止 = 评测站 root_pos 判据；8 卡线逐帧复跑 42 条 root_pos 失败：34 条躯干高度判据已先触发，40 条是跳不到参考高度（能力问题），预期无收益，r28 同起点对照中 |
| goal / critic | `--goal_ref_root` | goal 每帧加参考躯干高度 + 本体系线/角速度 + 重力（10 维，只用参考侧，可部署）；**有效**（骨盆终止之上 +0.5pp，E2 +0.4） |
| | `--critic_errors` | critic 加特权跟踪误差特征；无效 |
| 世界模型 | `--wm_body_frame`、`--policy_delay`、`--policy_freeze`、`--wm_delay_lr` | 本体系 WM（去掉俯仰 ±90° 的航向奇异）；无收益，翻腾失败阶段不变 |
| | `--wm_layernorm`、`--wm_input_clip`、`--wm_min_std` | 世界模型隐藏层加 LayerNorm、归一化输入钳位、归一化 std 下限（allmimic 的做法：翻腾时输入到 10~20σ）；续训无 LayerNorm 的 ckpt 会嫁接（LayerNorm 从恒等初始化开始，配 `--policy_delay`）；待验证 |
| 采样器 | `--source_weights`、`--name_weights` | 来源 / 关键词先验（只作用在均匀份额上） |
| | `--sampler_clamp_returns` | bin 回报钳到 [0, g_max]；**有效**（+0.8，防片尾负奖励吸走全部份额） |
| | `--sampler_cap 50` | softmax 份额每 bin 封顶 50×均匀（allmimic）；G1 r25 时一半重置落在 0.03% 的 bin 上；待 r28 验证 |
| | `--sampler_clusters 256 --sampler_cluster_alpha 0.3 --sampler_cluster_mix 0.5` | 12 个运动学特征 k-means 聚类，簇份额 ∝ n^0.3；与来源 / 关键词先验按 mix 混合而不是相乘（相乘时两者都给翻腾加权，翻腾 bin 占 0.08% 却拿到 G1 均匀份额的 34%；混合后约 4%）；待 r28 验证 |
| | `--sampler_boot_mean` | 片尾超时用平均奖励自举（替代 r_last） |
| | `--hard_names --hard_term_schedule` | 难片段放宽终止的课程；短课程无收益 |
| 修复 | `--fixes` | 开源对照发现的六项：输入钳位 ±50σ、价值钳位、critic 按存活加权、想象片尾截断 + 自举、WM 前缀掩码、观测用干净动作；整体在噪声内，最好的模型都开着 |
| 看门狗 | `--wd_sustain`、`--wd_warmup`、`--wd_baseline_after_warmup`、`--wd_signal`、`--wd_reward_floor`、`--wd_nf_patience` | 想象奖励塌缩 / critic 损失爆炸 / 连续非有限时回滚到健康 checkpoint 并减半 LR；默认 = 多卡分支行为（立即回滚），单卡线用括号里那组 |
| 提前停止 | `--es_eval_fail_ratio 1.5 --es_patience 2 [--es_median_floor 5]` | 训练内评测失败数连续超中位数 1.5 倍即停，另存 `model_best_eval.pt`；中位数下限默认 1，失败数个位数时过敏，建议 5；评测连续失败 3 次才关闭并明确提示 |
| 学习率 | `--lr`、`--lr_decay_iters`、`--base_lr_scale`、`--resume_lr_scale` | 余弦到 25 万轮剩 3%；后期调高 LR 有害，退火无收益；`--lr_scale` 已删除 |

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

有效（两条线合计）：末端终止（同起点 +8.8；v1 67.8% → v2 85.7%）、H32 → H48（+1.9）、来源加权采样（+3.7）、难片段课程（腾空 1/13 → 13/13，总分 −0.6）、采样器钳位（+0.8）、骨盆朝向终止（8 卡线 4+4 卡同起点六点全部领先，三点均值 +1.12；单卡线 +0.2~0.3，因为末端失败多了约 25 条）、goal 根部运动（单卡线骨盆终止之上 +0.5）、地面动作加权（单卡线长程 +0.3~0.4）、看门狗 alive 信号 + 奖励下限（8 卡线误回滚从 3 次到 0 次）。

无效或噪声内：开源对照六项修复、critic 特权误差、本体系世界模型、UTD 2.5、骨盆终止收紧到 0.8（两条线都不支持）、短难片段课程、checkpoint 权重平均、后期提高 / 降低学习率、8 卡从零训全量三层数据（H32、不加权，峰值 84.75 后退化）。

有害：H32 长续训（执行噪声 σ=0.1 下"峰后持续退化"，116k 88.51 → 212k 79.61；干预实验显示策略依赖训练时的执行噪声，确定性闭环会抖）——要看训练内评测 `eval/failed`（与 v4 相关 −0.98，比悬崖早约 2.8 万轮）并开早停。

翻腾：失败基本都发生在参考第一次倒立后 0.6 s 内（18/21），换本体系世界模型不变；r28 用采样器封顶 + 聚类和"只训翻腾"的诊断实验区分是数据被稀释还是配方本身的问题。

单 checkpoint 的 v4 噪声约 ±0.3~0.5pp，比较配方要看多个点的均值。

## 合并后的默认值变化（相对合并前的 master 60f6b6a）

| 项目 | 合并前 master | 合并后 | 想要原 master 行为时 |
|---|---|---|---|
| 末端终止 | 关（`--ee_term_z 0`） | 开，0.25 | `--no_ee_termination` 或 `--ee_term_z 0` |
| `--wd_signal` | `imagined_reward_robust` | `imagined_reward_norm` | `--wd_signal imagined_reward_robust` |
| 看门狗 warmup / 基线 / sustain | 1500 / warmup 后才计基线 / 200 | 500 / 从第 1 轮起 / 1 | `--wd_warmup 1500 --wd_baseline_after_warmup --wd_sustain 200` |
| 非有限的想象奖励 | 立即回滚 | 当暂态跳步，连续 `--wd_nf_patience 100` 轮才回滚 | — |
| `--lr_scale` | 续训覆盖运行时 lr_scale | 已删除（报错） | `--resume_lr_scale`；基础倍数是 `--base_lr_scale` |
| 簇先验 | 乘到来源先验上 | 混合（`--sampler_cluster_mix 0.5`） | `--sampler_cluster_mix 1` 只用簇先验（没有相乘模式） |

默认开关下，合并后的代码与 master 60f6b6a 逐位一致（CPU 6 轮，final ckpt 全部张量相同）；与 ddp-bj31-v2 42b211e 不逐位（e59e947 改了世界模型窗口损失的归一化），所以 8 卡线的数值轨迹只能统计意义上复现。

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
