# SuperMimic DDP 合并核对清单（bj31 八卡线 → master）

> 本文是 `README.md` §1 和 §2 的勾选速查版。细节、证据和出处都在 README。核对时间是 2026-10-26 17:1x。
> 合并状态：`ddp-bj31-v2`（42b211e）已在 **7d8d85b** 并入 master，之后是 e6ba84b（中文 README）和 **b0ddc5b**（测试修整），当前 HEAD 为 b0ddc5b。`ddp-bj31`（5bb8c0d）是 v2 的祖先，所以我方 15 个提交都已在 master 里。
> 标记含义：`[x]` 表示我已在 7d8d85b/e6ba84b 上读代码或复跑核到；`[ ]` 表示仍需处理或验证。

---

## A. 必须带过来的内容（P0）。缺任何一项，bj31 线都无法运行或复现

- [x] `supermimic/distributed.py` 整个文件（`DistCtx`），与 v2 相同。
- [x] trainer 里的全部 DDP 钩子：
  - `rms_update` 三处：obs/goal、WM 的 `stats_pairs`、`critic.err_norm`（trainer.py:660）；
  - `_apply_grads` 三处；
  - `_dist_start`、`_merge_sampler`、`_state_groups`、`_check_consistency`、`_dist_log_scalars`。
- [x] `scripts/train.py` 的三项：rank 0 解析文件表和 log_dir 后广播；`_die_with_torchrun_agent`（PDEATHSIG）；正常结束时才 `dctx.close()`。
- [x] `scripts/launch_bj31_ddp.sh` 和 `scripts/stop_bj31_ddp.sh`，与 v2 逐字节相同。
- [x] `motions.py` 的 body_names 回退（`G1_ISAAC_BODY_NAMES_37`、`npz_body_names`、`_worker_fill` 三元组）。bj31 缓存里 365855 条全部依赖这个回退。
- [x] `HeadsCfg.from_dict` 三种 cfg 都能读：
  - 我方：`ee_termination` / `ee_term_z_temp` / `root_ori_*`；
  - master v2：`ee_term_z` + `ee_term_temp` / `goal_ref_root`；
  - v1：没有 EE 键。
  - 核对方法：拿 model_174000 的 heads 调 from_dict，结果与生产 CLI 构造的 HeadsCfg 无差异。合并后 `scripts/eval_ckpt.py:50` 也改用了 `from_dict`。
- [x] 生产命令里每个开关都存在且语义不变：
  - 数据与采样：`--motion_list`（缺文件即报错）、`--robot_variant eval+evalcontact`、`--source_weights`、`--name_weights`、`--sampler_clamp_returns`；
  - 终止：`--ee_term_z 0.25`、`--root_ori_term 1.0 --root_ori_term_temp 0.05`；
  - 想象与看门狗：`--H 48`、`--wd_drop 0.2`、`--wd_reward_floor -1.0`、`--wd_signal imagined_reward_alive`；
  - 评测与早停：`--eval_clip_names`、`--es_eval_fail_ratio 1.5 --es_patience 2`；
  - 其它：`--param_check_interval 1000`、`--num_threads 24`。

## B. 应该带的内容（P1：修复与防护）

- [x] NaN 防护：`_apply_grads` 在规约后的梯度上判断是否全有限，各 rank 一致跳步；`--wd_nf_patience 100`；`--wd_early_ckpt 500`；`_learned_state_finite`；`guard/sim_nonfinite`。
- [x] 续训相关：`_resume_diffs`、`_load_sampler_state`（same / prefix / other）、`_resume_from`、`_es_rollback`。
- [x] 早停：`--es_eval_fail_ratio` / `--es_patience`、`model_best_eval.pt`、`best_eval.json`、`earlystop.json`。
- [x] 看门狗：`--wd_reward_floor`，以及 `--wd_signal imagined_reward_alive`。master e59e947 的 robust 方案同时保留，两者针对的失败模式不同。
- [x] 我方版本的 `extend_cache`：修了 `_worker_fill` 返回三元组时，`r is not None` 会把新增文件全判为失败的 bug。
- [x] 监控与日志（只记录，不影响训练）：`watch.py`、`eval/failed_ori`、`eval/failed_with_ori`、`pi/root_ori_*`、`wm/root_*`。
- [x] `cfg_dict` 在开关关闭时不写可选键，已扩展到 `goal_ref_root`、`critic_errors`、`root_z`。这样旧代码仍能读新 ckpt。
- [x] 骨盆终止只保留一套，用我方 ecc551f/f1f364b 的 `ref_rq` 签名，带 `root_ori_body`。master 60f6b6a 的 `root_z_term` 接在我方签名上。
- [x] 采样器钳位只保留一套（5bb8c0d 与 5f020a9 已在 decf018 合一）。

## C. 仍需处理的问题（按优先级）

- [ ] **高｜`--lr_scale` 改了语义。** 它现在是基础 LR 倍数，原来的运行时覆盖改名为 `--resume_lr_scale`。
  - bj27 上 16 个旧脚本（round19c…round27）仍按旧语义写 `--lr_scale 0.25/0.5/0.125/0.0625`，在新代码上 LR 会静默多乘一次。
  - 目前只有 README 第 101 行有说明，代码没有防护。
  - 建议：带 `--resume` 且 `--lr_scale ≠ 1` 时报错或要求确认；或把基础倍数改名为 `--base_lr_scale`，旧名直接报错。
- [ ] **中｜`_resume_diffs` 补比对项**：`input_clip`、`--fixes` 各项、`sampler_cap`、`sampler_clusters`、`sampler_cluster_mix`、`sampler_boot_mean`。
  - `input_clip` 不在 state_dict 里，却会改变前向。续训漏写 `--fixes`，它会从 50 静默变回 10。
  - 建议加一个单测：ckpt 带 `--fixes`，续训不带，应当打出 WARNING。
- [ ] **中｜看门狗默认值随我方**（norm、warmup 500、sustain 1、从第 1 轮起计基线），要写清楚两套风格各自的显式参数：
  - bj27 风格：`--wd_sustain 200 --wd_warmup 1500 --wd_baseline_after_warmup --wd_signal imagined_reward_robust`；
  - bj31 风格：`--wd_signal imagined_reward_alive --wd_reward_floor -1.0`。
- [ ] **中｜簇先验从"乘"改成了"混合"**（`--sampler_cluster_mix` 默认 0.5），60f6b6a 原来的语义已经不存在。我没查到这次改动的实验依据，要先问清楚。
- [ ] **中｜EE 默认开（0.25）。** EE 关的旧 ckpt 不加旗标续训会被打开；`_resume_diffs` 会打 WARNING，续训时要看。
- [ ] **中｜bj31 GPU 冒烟还没做**，见 E.2。
- [ ] **中｜bj31 上 `eval_export/scripts/submit.sh` 写死了 v1 profile。** ref_root 的 ckpt 交上去会被判不兼容，auto_v4 会无限重试。改用 master 的 `eval_site/submit_v4.sh`，它按 ONNX 的 `goal_layout` 选 profile。
- [ ] **低｜测试运行方式统一。** b0ddc5b 已让 master 带来的 4 个测试自己定位仓库、缺 mjlab/tensorboard 时打桩，我没复跑。另外 editable 安装指向 `/home/user/Workspace/supermimic`，所以不加 `PYTHONPATH=.` 时测的不是当前 worktree。
- [ ] **低｜审查报告提过、至今没修的问题：**
  - 评测一旦抛异常，`eval_enabled` 被永久置为 False，早停随之静默失效；
  - `eval_interval` 不是 sampler_sync / ckpt / 自检间隔的整数倍时，`model_best_eval.pt` 存的是未合并的 `bin_G`；
  - ckpt 的 lr_scale 等于 1.0 时，resumed 那一行打印的 LR 不对；
  - 3 rank gloo 测试在 barrier → destroy 之间偶发 SIGABRT（9 次里 1 次）；
  - `--wd_reward_floor` 不校验取值（nan、inf、正数都会出问题）；
  - `torch.clamp` 挡不住 NaN 的 `bin_G`；
  - 启动器日志里 PORT 显示重复；残留的 `MASTER_PORT` 会把默认启动变成 c10d；
  - `reset_prior` 在每个 rank 都打印；
  - 环境侧骨盆姿态滞后一个物理子步；
  - RSI 扰动让开局骨盆误差约 0.245 rad，A ≤ 0.25 时第一步就会被终止；
  - 早停判据用 `≥`，失败数很小时过敏，建议改成 `>` 或提高中位数下限。
- [ ] **低｜诊断脚本还在会话 scratch 里，要拷进仓库**，例如 `tools/diag/`。清单见 README §5.7。

## D. DDP 不变量（每次改 trainer 后逐条确认）

- [ ] **单卡路径不变。** `WORLD_SIZE ≤ 1` 时 `DistCtx` 处于禁用状态，所有 helper 退回单卡原代码。用 E.1 的逐位对照验证，基准是 master 60f6b6a。
- [ ] **设备绑定顺序。** 依次是 `torch.cuda.set_device(LOCAL_RANK)`、`torch.zeros(1, device)`（让 Warp 的默认设备落在本卡）、`init_process_group("nccl", device_id=…)`。另建一个 gloo 组，专门做 barrier 和对象广播。两个组的超时都是 `--dist_timeout_min`（120），必须长于 rank 0 的一次评测加一次建缓存。
- [ ] **种子。** 各 rank 的 torch、numpy、env 都用 `seed + rank`。评测子集用 `random.Random(cfg.seed)`，不加 rank。rank 0 的初始 policy 哈希等于单卡 seed0 的 `7af7000e5151`。
- [ ] **文件表和 log_dir 只由 rank 0 解析，再广播。** `--motion_list` 只要缺一个文件，就在广播前报错。
- [ ] **rank 0 先建 env 和缓存**（包括 `bin_clusters` 的特征和 k-means 结果），其余 rank 在 barrier 后再 mmap。
- [ ] **开训时 `broadcast_modules_(policy, critic, wm)`**，参数和 buffer 都要广播。
- [ ] **梯度同步。** 拼平后做 `all_reduce(SUM)`，除以 world，然后再裁剪、再 step。不要套 DDP 包装。是否全有限要在**规约后**的张量上判断，这样各 rank 会一致跳步。
- [ ] **新加的 RunningMeanStd 一律走 `dist.rms_update`。** 它做 all_gather (n, mean, var)，按 rank 顺序用 float64 Chan 公式合并。批为空的 rank 也要参与通信。
- [ ] **凡是会改变学到状态的随机数，各 rank 必须相同。** 嫁接时用 `torch.Generator(device).manual_seed(0)`，不能用全局 RNG。
- [ ] **采样器合并。** 每 `sampler_sync_interval` 轮（默认 50），以及每个 ckpt 槽和自检槽，做一次 `merge_sampler_bins`：
  - `bin_G` 按本期访问次数加权；
  - `bin_visits = V0 + ΣdV`；
  - `bin_weight_hist` 取各 rank 平均。
  - 开训、续训、回滚后，`_dist_start` 广播 rank 0 的三张表，并做一次全状态比对。
- [ ] **新增的学到状态要加进 `_state_groups`。** 现有的组：policy、critic、wm、三个 Adam、sampler、`sampler_prior`、counters、`library`、`curriculum`。用 blake2b 哈希经 gloo 比对，任何一组不同，所有 rank 都抛 RuntimeError。比对时机：开始、每 1000 轮、每次回滚后、结束时。
- [ ] **是否进 barrier，只能由 cfg 和迭代数决定。** 不能依赖 rank 0 运行时的状态（例如 `eval_enabled`）。TB、ckpt、评测、打印只在 rank 0 做。
- [ ] **日志聚合。** 先 crc 校验各 rank 的键集合，再取均值；回合统计和 watch 计数求和；`dist/*_max` 记最慢的 rank；`time/env_fps` 取合计。
- [ ] **早停。** rank 0 把评测摘要 `broadcast_object` 给所有 rank，各 rank 做同样的判决，在同一轮退出。
- [ ] **看门狗。** 先按 floor 钳，再 `reduce_scalars(mean)`，EMA、窗口、中位数、sustain 都作用在 rank 均值上。回滚时所有 rank 载入同一个文件，置 `buffer.size = 0`，LR 乘 `wd_lr_factor`，再调 `_dist_start`。
- [ ] **续训。** 所有 rank 读同一个 ckpt；`_resume_diffs` 只由 rank 0 打印；开训时 `_dist_start` 再比对一次。
- [ ] **进程生命周期。** rank 设 PDEATHSIG，停止一律用 `stop_bj31_ddp.sh`。原因是 elastic 让每个 rank 自成一个 session，只杀 pgid 会留下孤儿进程。
- [ ] **启动器。** 设了 PORT 时走 c10d，**必须带 `--rdzv-conf is_host=1`**，否则端口被占时会静默接入别人的 store。`CUDA_VISIBLE_DEVICES` 子集决定 NPROC。要清掉 `LD_PRELOAD` 和 `PYTHONPATH`，再设 `PYTHONPATH=仓库`。用 venv 自带的 NCCL 2.30.7+cuda13.3。

## E. 验证步骤

### E.1 CPU（本机）

- [x] **单进程逐位对照。** 合并后的代码与 master 60f6b6a 跑 6 轮，final ckpt 的全部张量逐位相同；25 个共有标量里只有 `guard/reward_ema` 不同，原因是看门狗默认信号不同。命令：
  ```
  PYTHONPATH=<code> ~/supermimic/mjlab/.venv/bin/python scripts/train.py --motion_dirs ~/supermimic/data/lafan1 \
    --device cpu --num_envs 16 --rollouts 16 --wm_windows 64 --wm_batches 2 --iterations 6 --log_interval 2 \
    --no_eval --ckpt_interval 0 --no_compile --ee_term_z 0.25 --seed 0
  ```
  比较工具是 `$S/a2test/cmp_ckpt.py` 和 `cmp_tb.py`，其中 `$S` = `/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad/supermimic`。
  - ⚠ 合并后的代码与 42b211e **不逐位**：6 轮后 obs_norm.mean 最大差 0.029，已知原因之一是 e59e947 改了 `window_loss`。今后"开关关闭即逐位"一律以 60f6b6a 为基准。
- [x] **2 rank gloo torchrun A/B/C 三组**，都 rc=0，每次 `[dist] … identical`。脚本 `$H/ddp/run_ddp.sh`，其中 `$H` = `…/scratchpad/handoff_tmp`。
  - A：生产开关加 `--wd_force_at 10`。两个 rank 同轮回滚到 model_8，回滚后哈希为 `917619962a63`。
  - B：master 侧功能全开，包括 `--critic_errors --goal_ref_root --fixes --sampler_cap 50 --sampler_clusters 4 --sampler_boot_mean --root_z_term 0.25` 和 robust 看门狗。
  - C：从 A 的 model_8 续训，带嫁接（goal 64→74，critic 误差特征 0→57），加 `--resume_lr_scale 0.25`。
- [x] **单测。** 我方 10 个文件共 95 个 ok。注意 gmr python 下 `test_ddp_sync` 会静默跳过 4 项，`test_nan_guard` 跳过 1 项。master 带来的 4 个测试要用 mjlab venv 才能通过。
- [ ] **在 b0ddc5b 上重跑全部单测**：`for t in tests/*.py; do PYTHONPATH=. ~/supermimic/mjlab/.venv/bin/python "$t"; done`。

### E.2 bj31 GPU（全部还没做）

- [ ] **部署代码。** 用 `git archive` 加 rsync 放到新目录 `~/supermimic/supermimic_repro_<tag>/`，写 `COMMIT` 文件，逐文件核对 md5。不要覆盖正在被 run 或 auto_v4 使用的目录。
- [ ] **在 bj31 上跑 14 个单测**，用真 mjlab、CPU。
- [ ] **8 rank 冒烟**，300 轮，参数为 `--eval_interval 200 --ckpt_interval 200 --param_check_interval 100`。逐项检查：
  - banner 是 `NCCL version 2.30.7+cuda13.3`；
  - 每张卡只有一个进程；
  - 每次都是 `[dist] … identical`；
  - ms/it 落在参考值附近：H48 8 卡 p50 约 890–950，H32 约 620；
  - rank 0 评测时其余 rank 在等，评测后继续；
  - rank 日志大小正常，以此验证 60f6b6a 的 warn_overflow 修复；
  - 用 model_200 `--resume` 再起一次。
- [ ] **从 174000 续训 200 轮。** ckpt 为 `/data2/supermimic/warmstart_ori8g_model_174000.pt`，md5 `2bd84993…`。日志里应满足：
  - 打印 `lr_scale 0.0625 restored from checkpoint`；
  - 打印 `policy lr 1.310e-05 = lr x lr_scale 0.0625 x cosine 0.2349`；
  - **没有** `WARNING resume from another setup`；
  - `255008 clips … 5428418 bins`；
  - EE on（4 个体）、root-orientation on；
  - **不触发嫁接**：goal 保持 64 维。
- [ ] **同机双任务。** 两个 4 卡任务，PORT 分别为 29521 和 29522，间隔 ≥ 300 s。再用同一个 PORT 起第三个任务，应当立即失败。用 stop 脚本停掉后，确认没有残留进程。
- [ ] **导出链。** 运行 `REPO=<新目录> XS=$REPO/scripts RUN_DIR=… TAG=… bash /data2/supermimic/eval_export/scripts/export.sh 174000`。期望 `check_onnx_vs_torch` 通过；ONNX sha256 与 `1aa8742b…291803c2c0` 一致，或至少数值等价；`eval_ckpt.py` 能读出 root_ori 是开的。

## F. 在合并后的代码上复现 bj31 生产 run 时必须显式写的 flag

```
--robot_variant eval+evalcontact --ee_term_z 0.25 --sampler_clamp_returns --H 48
--wd_drop 0.2 --wd_reward_floor -1.0 --wd_signal imagined_reward_alive
--es_eval_fail_ratio 1.5 --es_patience 2 --root_ori_term 1.0 --root_ori_term_temp 0.05
--lr 8.9252e-4   # 从 174k 起步，实际 LR = 8.9252e-4 × 0.0625（ckpt 里继承的 lr_scale）× 0.2349 = 1.310e-5
# 不要写 --lr_scale。要改运行时倍数，用 --resume_lr_scale
# 除非在做 T/C 对照，否则不要加 --fixes / --goal_ref_root / --critic_errors / --sampler_cap / --sampler_clusters / --sampler_boot_mean / --root_z_term
```

完整命令见 README §3.3。

## G. 默认值变化速查（相对合并前的 master 60f6b6a）

| 项目 | 合并前 master | 合并后（随我方） | 想要 master 原行为时加什么 |
|---|---|---|---|
| EE 终止 | 关（`--ee_term_z 0`） | 开，0.25 | `--no_ee_termination` 或 `--ee_term_z 0` |
| `--wd_signal` | `imagined_reward_robust` | `imagined_reward_norm` | `--wd_signal imagined_reward_robust` |
| `--wd_warmup` / 基线 / sustain | 1500 / warmup 之后才计基线 / 200 | 500 / 从第 1 轮起计 / 1 | `--wd_warmup 1500 --wd_baseline_after_warmup --wd_sustain 200` |
| 非有限的 r | 立即回滚 | 当作暂态，`--wd_nf_patience 100` | — |
| `--lr_scale` | 续训时覆盖运行时 lr_scale | 基础 LR 倍数 | 改用 `--resume_lr_scale` |
| 簇先验 | 乘到源先验上 | 混合，`--sampler_cluster_mix 0.5` | 只能改代码 |
