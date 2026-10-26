# SuperMimic G1 · bj31 八卡线交接（10-22 → 10-26）

> **读者**：在 `~/Workspace/supermimic` 整合代码、之后接手 G1 训练的 agent。
> **写于**：2026-10-26 17:1x（北京时间）。写作时只读核对过，master 当前 HEAD 是 **b0ddc5b**（16:57）。
> **依据**：四份分节材料，分别是代码、实验、运维、诊断。材料来源包括：读代码（`git show`、两个分支的 worktree、合并后的 master）；bj31 只读日志和评测站 API 的 GET 查询；记忆笔记；各工作流的结果。标〔复跑〕的数字是 10-26 在 scratch 克隆里实际跑出来的。分节材料之间有矛盾的地方集中列在 §0.5。
> **配套文件**：`MERGE_CHECKLIST.md`，是 §1 和 §2 的勾选版，放在同一目录。

**路径缩写**

| 缩写 | 实际路径 |
|---|---|
| `$SP` | `/tmp/claude-1000/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/scratchpad` |
| `$S` | `$SP/supermimic`。诊断脚本、守护脚本、两个分支的 worktree 都在这里 |
| `$H` | `$SP/handoff_tmp`。今天的复跑产物、合并后 master 的 scratch 克隆 `mt`、60f6b6a 的克隆 `m60f` 都在这里 |
| `$WF` | `/home/user/.claude/projects/-home-user-Workspace-Noetix-GMT/7380f59e-8ed0-40ea-9060-69c5f5a5a1ef/subagents/workflows` |
| `$MEM` | `/home/user/.claude/projects/-home-user-Workspace-Noetix-GMT/memory` |
| bj31 | ssh 别名 `tencent_bj31`，数据根目录 `/data2/supermimic` |
| bj27 | ssh 别名 `tencent_bj27`，另一会话的单卡配方探索线 |

⚠ `$SP` 是会话级 scratch，会话结束后可能被清理。要长期保留的脚本请拷进仓库，例如 `tools/diag/`；清单见 §5.7。

---

## 0. 结论与状态

### 0.1 一句话

bj31 上用 8 卡 torchrun 数据并行训练 SuperMimic G1，v4 最好做到 **91.83%，排行榜 #1**。增益基本都来自配方：EE 终止、H48、采样器钳位、骨盆全角朝向终止、从单卡最好链热启动。8 卡本身每轮只相当于单卡约 1.6 倍（经验估计），墙钟只快约 1.3 倍。代码已由另一个 agent 在 7d8d85b 并入 master，我逐项核对过，没有发现问题。还需要处理的有三类：几处 flag 改名和默认值变化留下的坑、续训时漏比的几个设置、bj31 GPU 上的冒烟测试还没做。

### 0.2 当前最好的 ckpt

| 项目 | 内容 |
|---|---|
| ckpt | 8 卡主线 `g1v2_ro10_8g_168k` 的 **model_174000** |
| v4 官方 | **91.83%**，run `b43fdd83`，10-26 16:2x 查询时排 #1 |
| 失败 256 条 | EE（body_pos）171，root_ori 43，root_pos 42；MPJPE 0.0851 |
| 对比原榜首 | GMT v2 集群 112500（run `130b4a73`）91.42%，失败 269 条 = EE 250 / root_pos 18 / root_ori 1 |
| 代码 | 42b211e（`ddp-bj31-v2`），bj31 目录 `~/supermimic/supermimic_repro_oriterm` |
| 位置 | bj31 `/data2/supermimic/logs/2026-10-26_03-10-22_g1v2_ro10_8g_168k/model_174000.pt`，同 md5 的副本是 `/data2/supermimic/warmstart_ori8g_model_174000.pt` |
| 校验 | ckpt md5 `2bd8499307d4f63adaca131aad9dd7be`；评测 ONNX sha256 `1aa8742b…291803c2c0` |
| 交付包 | `/home/user/Workspace/Noetix_GMT/deliverables/policies_1026/supermimic_g1_oriterm_8g_174000/` |

**其它值得保留的点**（都在 bj31 上）：

| ckpt | v4 | 说明 |
|---|---|---|
| T 臂 `g1v2_ro10_160k_T4` 168000 | 91.80 | 第一轮骨盆终止对照的赢家；交付包 `policies_1026/supermimic_g1_oriterm_T_168000/`，sha256 `c66915ae…` |
| C2 `g1v2_ro10_174k_C4` 182000 | 91.74 | 4 卡，从 174000 续训 |
| r15ws 160000 | 90.94 | H48 8 卡线骨盆终止之前的峰值；交付包在 `policies_1025/`，sha256 `a4d7df78…` |
| ws86k 92000 | 88.70 | H32 各线中最好的点 |
| bj27 `g1v2_floor10_r25` 208000 | 91.70 | 另一会话的单卡线。goal 是 2nj+16 的 ref_root 布局，需要评测站的 `supermimic_g1_v2` profile 和新的部署模式 |

### 0.3 状态：10-26 暂停

- 10-26 15:4x 用户指示暂停八卡 SuperMimic，等另一个 agent 合并代码。本会话之后转去做 E2V2 和 E2 高动态 C6/C7。
- 第二轮两臂的停止情况：
  - T2 `g1v2_ro08_174k_T4`：用 `stop_bj31_ddp.sh … || true` 停在约 185,850 轮，最后一个 ckpt 是 184000。
  - C2 `g1v2_ro10_174k_C4`：正常跑完到 186000，v4 91.58%（run `95653dbe`）。
  - 两路 auto_v4 分别在 15:44 和 15:54 打出 `training gone, nothing left -> exit`，自行退出。
- 16:05 和 17:0x 两次只读核对：8 张卡显存都是 0 MiB，没有我方的 train.py、torchrun 或 auto_v4 进程。评测站 `noetix-eval` 在运行，另一会话有一批私有离线评测进程在跑。
- bj31 的 8 张卡已转给 E2 HDC C7。C7 原来放在 `/dev/shm` 的 e2hdc8 缓存（267G）10-22 已删，回 bj31 要重建，估计约 1 h。SuperMimic 的缓存和 ckpt 都在 `/data2`，不受影响。

### 0.4 交给整合 agent 的五件要紧事

1. **⚠ `--lr_scale` 改了含义。** 合并后它表示"基础 LR 倍数"，原 master 7afc772 的"续训时覆盖运行时 lr_scale"改名为 **`--resume_lr_scale`**。
   - bj27 上 16 个旧启动脚本（round19c…round27）仍按旧语义写 `--lr_scale 0.25/0.5/0.125/0.0625`。拿到新代码上跑，LR 会**静默地再乘一次**。
   - 新 README 第 101 行已写明两者的区别，但代码里没有任何防护（§1.2 #1）。
2. **看门狗和 EE 的默认值改成了我方的。** 具体是 EE 开（阈值 0.25）、`wd_warmup 500`、`wd_sustain 1`、`wd_signal imagined_reward_norm`。
   - 要复现 bj27 风格，必须显式加 `--wd_sustain 200 --wd_warmup 1500 --wd_baseline_after_warmup --wd_signal imagined_reward_robust`。
   - 要复现 bj31 风格，必须显式加 `--wd_signal imagined_reward_alive --wd_reward_floor -1.0`。
3. **续训时 `_resume_diffs` 不比对**以下设置：`input_clip`、`--fixes` 各项、`sampler_cap`、`sampler_clusters`、`sampler_cluster_mix`、`sampler_boot_mean`。其中 `input_clip` 不在 state_dict 里，却会改变策略的前向。我 17:1x 用 grep 核对过，b0ddc5b 仍是这样。
4. **数值轨迹无法逐位复现 bj31 的 run。** 合并后的代码与 42b211e 不逐位一致，已确认的原因之一是 master e59e947 改了 `window_loss`。能复现的是配方，不是数值。合并后的代码在默认开关下与 **master 60f6b6a** 逐位一致，今后"开关关闭即逐位"的基准应改用 60f6b6a。
5. **合并后的代码还没上过 bj31 GPU。** 包括 8 rank NCCL 冒烟、从 174000 续训、导出链，都没做（§1.6）。

### 0.5 分节材料之间的分歧与更正

| # | 分歧 | 结论（以哪个为准） |
|---|---|---|
| 1 | 运维节写"master HEAD 60f6b6a，缺 `distributed.py` / `watch.py` / 启停脚本 / tests，合并时要带上"；诊断节写"任务描述里的 975decb 已过时" | **都已过时。** 合并已完成：7d8d85b（16:26）→ e6ba84b（16:34，中文 README）→ **b0ddc5b**（16:57，master 来源的 4 个测试改成自己定位仓库、缺 mjlab/tensorboard 时打桩）。运维节 §11 列的需要带上的文件，在 7d8d85b 里都核到了 |
| 2 | 诊断节写"master `eval_ckpt.py` 用 `HeadsCfg(**hk)`，读我方开了骨盆终止的 ckpt 会 TypeError" | **合并后已不成立**：`scripts/eval_ckpt.py:50` 改用 `HeadsCfg.from_dict(hk)`，`HeadsCfg` 也有 `root_ori_body` 字段（17:1x 用 `git show master:` 核对）。但用合并前的 master（≤60f6b6a）读我方 ckpt，仍会遇到这个问题 |
| 3 | 实验节 §7 写"用 master 7afc772 的 `--lr_scale` 覆盖" | 在合并后的代码里，这个覆盖叫 **`--resume_lr_scale`**；`--lr_scale` 已是基础倍数。见 §0.4 第 1 条 |
| 4 | 记忆笔记写 8 卡主线停在"~183k" | 以日志为准：最后一行 it 188500（10-26 11:00 停），最后 ckpt model_188000。184k/186k/188k 的 v4 为 91.13 / 91.16 / 91.23（run `ba79cdb0` / `813a7e01` / `be621069`） |
| 5 | 记忆笔记写 r15ws 最终到 162000 | 以日志为准：跑到 166000。162k/164k/166k 的 v4 为 90.68 / 90.59 / 90.52 |
| 6 | T2 的停止点：实验节和运维节写 185,850，诊断节写约 185.6k | 以日志的 185,850 为准。两种说法下最后一个 ckpt 都是 184000 |
| 7 | 254,518 条缓存的构建时间：实验节写约 75 min，运维节写实测约 26.5 min | 75 min 是按"349 s 建 2 万条"外推出来的估计。26.5 min 来自会话记录里的口头汇报，没有按日志时间戳逐秒核对 |
| 8 | `extend_cache` 用时：记忆和启动器注释写约 11 min，bj31 `prep_1025/prep.log` 记 `done in 166s` | 166 s 可能不含 dry_run 和校验。两个数字都留着 |
| 9 | 8 卡 H48 每轮耗时：代码节写 959 ms/it（c 64 / wm 251 / pi 644），fps 33956；实验节和运维节写 p50 892–895（8g）、951–952（r15ws） | 口径不同：前者取的是日志最后一行，后者是全程 p50。4 卡同理：代码节的"C2 780 ms/it"约等于 p10，p50 是 1125，被 CPU 争用拉高 |
| 10 | C2 五点均值：实验节写 91.47，诊断节写 91.48 | 实际计算为 91.476。本文写 91.48 |
| 11 | "我们败、榜首过"的 61 条按类拆分：记忆 M3 为起身 16 / 跳跃 17 / 其它 24；`gap_next` 为起身 15 / 跳跃 18 / 其它 21 | 两边用的关键词分类器不同，总数 61 都一样 |
| 12 | 执行动作加噪的代码位置：实验节写 `trainer.py:307`，诊断节写 repo_v2 `trainer.py` L421 | 是同一处代码（`collect_step` 里的 `a_exec = a + action_noise_std·randn`），只是版本不同、行号不同 |
| 13 | 单测：诊断节写"10 个文件全过"，代码节写"14 个文件" | 我方 10 个文件 + master 带来的 4 个 = 14 个。复跑时 master 那 4 个在 gmr python 下导入失败，换 mjlab venv 加 `PYTHONPATH=.` 才通过。b0ddc5b 声称已修好这一点，我**没有复跑** |
| 14 | "8 卡每轮约等于单卡 1.6 倍" | 出自记忆笔记 [M-bj31]，推导没有留档。我重建的一个可能依据是：8 卡 20k 为 83.18%，单卡 v2 从零训到 32k 为 82.99%，32/20 = 1.6。但两边配方不同，只能当量级参考 |

---

## 1. 合并清单（给整合 agent）

### 1.1 现状

- **分支拓扑**
  - merge-base：master 与 `ddp-bj31-v2` 是 5f020a9；master 与 `ddp-bj31` 是 3a29365。
  - `ddp-bj31`：c59c8ce → 62058a5 → b9415fc（并入 master 3a29365）→ 01c4d58 → bc9d507 → **5bb8c0d**。worktree 在 `$S/repo`。
  - `ddp-bj31-v2`：5bb8c0d → a4f8009（并入 master 18f528f）→ df670e6 → ca9266e → decf018（并入 master 5f020a9）→ 1662040 → 55e0eaf → ecc551f → f1f364b → **42b211e**。worktree 在 `$S/repo_v2`。
  - master 在 merge-base 之后：e59e947 → 052247b → 3396825 → 8453c3b → 7afc772 → f82e276（我方 ecc551f 的移植）→ 3ed4555 → 975decb → 60f6b6a → **7d8d85b**（合并）→ e6ba84b（文档）→ **b0ddc5b**（测试）。
  - 5bb8c0d 是 v2 的祖先，所以并入 42b211e 就带上了我方全部 15 个提交。
- **试合并的冲突规模**：我把 60f6b6a 并进 42b211e 试过一次，共 42 个冲突块，分布在 10 个文件：trainer 19、train.py 6、env_cfg 4、heads 4、library_command 3、sim_interface 2，eval_ckpt / evaluate / math_utils / world_model 各 1。**还有几处会导致 DDP 分叉，但 git 自动合并、不报冲突**，7d8d85b 都已改好：
  - `critic.err_norm.update()` 是直接调用，已改为 `dist.rms_update`；
  - `_graft_wm` / `_calibrate_goal_norm` 用的是全局 RNG，已改为固定种子的 Generator；
  - `fns.py` 和 `check_fk.py` 绑定的是我方的 `ref_rq` 签名。
- **我在 7d8d85b / e6ba84b 上做过的验证**〔复跑〕：
  1. **单进程逐位对照**。CPU eager 跑 6 轮，数据 lafan1 8 条，加 `--ee_term_z 0.25`。合并后的代码与 master 60f6b6a 的 final ckpt **全部张量逐位相同**。25 个共有的非计时 TB 标量中，只有 `guard/reward_ema` 不同，原因是看门狗信号的默认值不同。
  2. **2 rank gloo torchrun 端到端**，三组都 rc=0，每次 `[dist] … learned state identical`。三组的设置与结果见 §1.5。
  3. **CPU 单测**：14 个文件都能通过，但有运行条件，见 §1.5。
  4. **HeadsCfg 读取**：用 model_174000 的 heads 字典（取自 `train_cfg.json`）在合并后的代码上调 `HeadsCfg.from_dict`，结果与生产 CLI 构造出的 HeadsCfg 无差异。master 格式（`ee_term_temp` + `goal_ref_root`）也能读。

### 1.2 仍需处理的问题（按优先级）

| # | 优先级 | 问题 | b0ddc5b 现状 | 建议 | 怎么验证 |
|---|---|---|---|---|---|
| 1 | **高** | `--lr_scale` 改义：旧脚本的 LR 会被静默多乘一次。另外，ckpt 里继承的 lr_scale 照样生效，也不会有告警 | 只有 README 说明，代码没有防护。bj27 上 16 个脚本（round19c…round27）用的是旧语义，各值出现次数为 0.25×14、0.5×2、0.125×1、0.0625×2；round28 已改用 `--resume_lr_scale` | 在带 `--resume` 且 `--lr_scale ≠ 1` 时报错或要求确认；或把基础倍数改名为 `--base_lr_scale`，旧名直接报错 | 拿旧脚本的参数 dry-run，看 `resumed … (policy lr … = lr x lr_scale … x cosine …)` 这一行 |
| 2 | 中 | `_resume_diffs` 不比对 `input_clip`、`--fixes` 各项、`sampler_cap/clusters/cluster_mix/boot_mean` | 未改（`git show master:supermimic/trainer.py` 的 `_resume_diffs` 中 grep 不到这些键） | 把它们加进比对对。r17 之后 bj27 链都用了 `--fixes`（`input_clip` 50），续训漏写 `--fixes` 会静默变回 10 | 加单测：ckpt 用 `--fixes`，续训不带，应出现 WARNING |
| 3 | 中 | 看门狗默认值随我方（§0.4 第 2 条） | 默认 `imagined_reward_norm`。不加 alive + floor 时，norm 会被"死后"的 WM 发散拖到 −17 / −35，即 8 卡主线日志的 `rew -17.221` 和 T2 的 `rew -35.841` | 两套风格各写一个示例启动脚本；或者在 README 的推荐命令里写全 | 看日志开头的 `watchdog signal: …` 一行 |
| 4 | 中 | 簇先验的语义变了：60f6b6a 是**乘**到来源/关键词先验上，合并后是**混合**，`(1−mix)·源先验 + mix·簇先验`，`--sampler_cluster_mix` 默认 0.5 | 60f6b6a 的原语义已不存在；bj27 round28 的注释写的是 "mix 0.5" | 先问清另一会话为什么这么改。我没查到实验依据 | — |
| 5 | 中 | EE 默认开：EE 关的旧 ckpt 不加旗标续训会被打开 | `_resume_diffs` 对 `heads.ee_termination` 会打 WARNING | 续训后检查 WARNING | 日志 |
| 6 | 中 | 合并后的代码还没在 bj31 GPU 上跑过 | — | 按 §1.6 的清单做 | §1.6 |
| 7 | 中 | 我方的 `submit.sh`（在 bj31 `eval_export/scripts/`，不在仓库里）写死了 `observation_spec_id=supermimic_g1_v1`。ref_root 的 ckpt 交上去会被判不兼容，auto_v4 会每 10 分钟重试一次，没有上限 | 未改 | 改用 master 的 `eval_site/submit_v4.sh`：它按 ONNX 的 `goal_layout` 自动选 v1/v2，`OFFLINE=1` 时走私有离线评测、不上榜 | 交一个 ref_root 的 ONNX，看返回是否 `status=compatible` |
| 8 | 低 | 测试运行环境：本机 mjlab venv 的 `import supermimic` 是 editable 安装，指向 `/home/user/Workspace/supermimic`。在别的 worktree 里不加 `PYTHONPATH=.`，测的其实是 editable 指向的那份代码 | b0ddc5b 让 4 个 master 测试自己定位仓库，未复跑 | 统一写成 `PYTHONPATH=. ~/supermimic/mjlab/.venv/bin/python tests/<f>.py` | §1.5 |
| 9 | 低 | 审查报告提过、至今没修的问题，见本表下方列表 | 未修 | 有空再修 | — |
| 10 | 低 | 代码节称 `--wd_signal` 的 help 文本没提 robust | 纯文案，b0ddc5b 上没复核 | — | — |
| 11 | 低 | 诊断脚本都在 `$SP`，会话结束可能丢失 | — | 按 §5.7 拷进仓库，例如 `tools/diag/` | — |

**#9 的遗留问题清单**（来自 wf_6f8fe82a、wf_489d1978、wf_ae810087、wf_74aad1e4 的审查）：

- **评测异常让早停永久失效**：评测一旦抛异常，rank 0 就把 `cfg.eval_enabled` 永久置成 False，之后早停不再起作用，也不提示。
- **最佳 ckpt 可能存了未合并的采样器**：`eval_interval` 不是 sampler_sync / ckpt / 自检间隔的整数倍时，`model_best_eval.pt` 里存的是 rank 0 尚未合并的 `bin_G`。生产配置下不会触发。
- **resumed 行的 LR 打印不准**：ckpt 的 lr_scale 恰好是 1.0 时不会调用 `_set_lr_scale`，resumed 那一行打印的是 ckpt 里存的 LR。今天 C 组复现了这个问题。
- **gloo 退出竞态**：3 rank gloo 测试跑 9 次，有 1 次在 barrier 和 `destroy_process_group` 之间 SIGABRT，原因未查明。
- **`--wd_reward_floor` 不校验取值**：传 nan 等于没开；传 inf 会每 100 轮回滚一次，最后 RuntimeError；传正数会屏蔽塌缩检测。所有 rank 同时 < −1 的真塌缩会晚约 5 轮才发现。
- **`torch.clamp` 不处理 NaN**：载入含 NaN 的 `bin_G` 后，钳位挡不住，softmax 会失效。
- **启动器的两个小问题**：
  - 日志行 `rendezvous ${PORT:+localhost:$PORT}${PORT:-standalone}` 会打成 `localhost:2952229522`。只是显示错，实际命令是对的。
  - shell 里残留的 `MASTER_PORT` 会让默认启动从 standalone 变成 c10d。
- **`reset_prior` 在每个 rank 都打印**：各 rank 日志里都会有一份来源份额，纯外观问题。
- **环境侧骨盆姿态滞后一个物理子步**：`bad_root_ori_full` 读的是 `body_link_quat_w`，比 `sim.forward()` 晚一个子步，快速旋转时最多差约 0.03 rad，所以阈值附近 `term_mismatch` 不为 0。
- **RSI 扰动造成开局误差**：RSI 的 roll/pitch ±0.1、yaw ±0.2 会让开局骨盆误差约 0.245 rad，所以 A ≤ 0.25 时第一步就会被终止。
- **早停在失败数很小时过敏**：判据是 `≥` 中位数 × 1.5，中位数下限取 1。中位数为 2 时阈值只有 3.0。建议改成严格 `>`，或提高中位数下限。

### 1.3 逐提交清单（我方 15 个提交）

"逐位"一列的意思是：开关关闭或取默认值时，与改动前的代码相比是否逐位一致。"7d8d85b"一列是我读代码核对的结果。

| 提交 | 内容 | 关键 flag（默认值） | 逐位 | 合并前 master | 7d8d85b | 测试 |
|---|---|---|---|---|---|---|
| **c59c8ce** 10-22 | npz 缺 `body_names` 时回退到 `G1_ISAAC_BODY_NAMES_37`；按名字固定评测子集；设置线程数 | `--eval_clip_names <txt>`（""，会覆盖 `--eval_clips`，缺失项写进 `eval_clip_names_missing.txt`，一条都找不到就报错）；`--num_threads`（0）；`eval_ckpt.py --eval_clip_names` | 用回退顺序建的缓存与用原件建的缓存，6 个字段加 clips.json 逐位相同，缓存键不变 | 没有 | ✓ `motions.py` 与 v2 逐字节相同 | `test_motions_names`（3）。**P0**：bj31 缓存 `lib_365855_b608e70f1d` 中 `body_names_fallback_files` = 365855，全部依赖回退 |
| **62058a5** 10-22 | torchrun 数据并行（`supermimic/distributed.py` 的 `DistCtx`）、`launch_bj31_ddp.sh`、`stop_bj31_ddp.sh` | `--lr_scale`（1.0，基础倍数，记为 `TrainCfg.base_lr_scale`）；`--dist_timeout_min`（120）；`--sampler_sync_interval`（0，表示取 log_interval）；`--param_check_interval`（1000）；TrainCfg 新增 `world_size` | 不经 torchrun 时与 005e330 逐位一致：eager 和 compile 都比过，final ckpt 全部张量相同，22 个标量相同 | 没有 | ✓ `distributed.py` 与 v2 相同 | `test_ddp_sync`（2 进程和 3 进程 gloo）。**P0** |
| **b9415fc** 10-22 | 并入 master 3a29365，并让看门狗多卡一致：`reduce_scalars(check_keys=False)` 求 rank 均值；回滚后调 `_dist_start`；只有 rank 0 落盘 | — | 单进程与 3a29365 逐位一致（24 个标量含 guard/*）。2 进程 `--wd_force_at 6` 时两个 rank 同轮回滚，回滚后哈希都是 `6ae11a909a4f` | 3a29365 已在；多卡改造没有 | ✓ | — |
| **01c4d58** 10-22 | EE 高度终止：环境侧硬终止，想象侧用 σ((dz−0.25)/0.02) 的软项，与其它项做 noisy-OR。NaN 防护：规约后梯度非有限时各 rank 一致跳步、Adam 状态不动；NaN 期间存的 ckpt 不登记为健康；早期回滚点；仿真非有限时按失败结束 | HeadsCfg `ee_termination` True / `ee_term_z` 0.25 / `ee_term_z_temp` 0.02；`--ee_termination/--no_ee_termination/--ee_threshold`；`--wd_nf_patience`（100）；`--wd_early_ckpt`（500）；`eval_ckpt --ee_termination {ckpt,on,off}` | `--no_ee_termination` 在 CPU 上跑 40 轮逐位一致。**默认 EE 开，与改前不同，这是有意的**。梯度全有限时 NaN 防护逐位不变 | e57be5b 有 `--ee_term_z`（默认 0，即关），引擎逐位相同；没有 NaN 防护 | 用我方实现、我方默认；两种拼写都走 `heads.ee_flags` | `test_ee_termination`（7）、`test_nan_guard`（5）、`test_ddp_sync` 里的跨 rank NaN 用例 |
| **bc9d507** 10-22 | `_die_with_torchrun_agent`：在 torchrun 下设 `prctl(PR_SET_PDEATHSIG, SIGKILL)` | — | 非 torchrun 时不变 | 没有 | ✓ | — |
| **5bb8c0d** 10-23 | `--sampler_clamp_returns`：把 EMA 输入、采样用的 G、载入的 `bin_G` 都钳到 [0, g_max] | 默认关 | 关时与 bc9d507 逐位一致 | 5f020a9 是同名同修法 | 只保留一套（decf018 已合一，用 `clamp_bin_returns_`，只在真正恢复了 bin_G 时才钳并报告） | `test_sampler_clamp`（8） |
| **a4f8009** 10-23 | 并入 master 18f528f（18 个冲突块，10 个文件）。内容：EE 合成一套，`HeadsCfg.from_dict` 能读三种 cfg，`ee_flags` 遇到两种拼写不一致时报错；`library_command.reset_prior` 抽成函数，先验纳入 `sampler_prior` 校验组；`--motion_list` 由 rank 0 解析后广播；回滚时清空回放（`buffer.size = 0`）；`--robot_variant`；`eval_site/*` | `--wd_drop` 默认值 0.08 → **0.2** | `reset_prior` 与 master 的内联写法逐位一致 | — | ✓ | `test_v2_merge`（11） |
| **df670e6** 10-23 | `--motion_list` 只要缺一个文件，就在广播前抛 FileNotFoundError；`_resume_diffs` 在续训（非回滚）时逐条打 WARNING；采样器状态只在 bin 数和 basename 序列都相同时才载入 | — | — | 没有 | ✓ | — |
| **ca9266e** 10-23 | `--wd_reward_floor`：每个 rank 先把有限的 r 钳到 ≥ floor，再求 rank 均值和 EMA；记 `guard/reward_floored`；抽出 `_guard_scalars` | 默认 None | floor 为 None 时与 df670e6 逐位一致 | 没有；master 的思路是 e59e947 的 robust 指标 | 两者都保留 | `test_watchdog_floor`（9） |
| **decf018** 10-25 | 并入 master 5f020a9，包括 4314a50 难片段终止课程、c1c94be extend_cache、5f020a9 钳位。改动：`heads.termination(scale=None)`；`fns.post` / `imagine_step` 加 `term_scale=None`；新增 `hard_mult_at(it)` 和 `curriculum` 校验组；`motions.extend_cache` 与 `library_cache_dir`；`MotionLibrary.fingerprint` 纳入 `library` 组；`_resume_diffs` 返回 same / prefix / other | `--hard_term_schedule` 必须配 `--hard_names` | scale 为 None 时与课程前逐位一致，编译图也相同 | — | ✓。另外修了一个合并后才会出现的 bug：我方 `_worker_fill` 返回 `(path, err, fallback)` 三元组，master 的 `r is not None` 判断会把新增的 490 个文件全判为失败，bin 数变成 5,419,733 | `test_merge_master_1025`（15） |
| **1662040** 10-25 | 早停：失败数 ≥ R × 之前所有评测失败数的中位数（至少 3 次评测，中位数下限 1），连续 P 次即停。停止时写 `earlystop.json` 并存 `model_final`，另外始终维护 `model_best_eval.pt` 和 `best_eval.json`。看门狗新增 alive 信号 Σc_t·r_t / Σc_t | `--es_eval_fail_ratio`（None，即关）；`--es_patience`（2）；`--wd_signal {imagined_reward_norm(默认), imagined_reward_alive}` | 早停关时与 ca9266e 逐字一致 | 没有 | ✓ | `test_early_stop_alive`（11） |
| **55e0eaf** 10-25 | `_es_rollback`（回滚时早停历史同步回退）；续训落在评测槽时先评一次；`_resume_from` | — | — | 没有 | ✓ | — |
| **ecc551f** 10-25 | 骨盆全角终止：环境侧 `library_command.bad_root_ori_full`；想象侧 `heads.termination(…, ref_rq=)` 用 σ((ang−A)/T) 软项；角度用 `math_utils.quat_angle`。常开监控 `watch.py::FailureWatch`。启动器支持 `PORT`、`CUDA_VISIBLE_DEVICES`、`DRY_RUN` | `--root_ori_term A`（0 = 关）、`--root_ori_term_temp`（0.05）、`--root_ori_body`（"" 表示机器人根体） | 关时与 55e0eaf 逐位一致：termination（eager 和 compile）、step fns、collect_step、Evaluator 都比过 | f82e276 是移植版，只搬了终止部分，签名为 `(kin, ref_bp, ref_bq, scale)`，没有 `root_ori_body` 和 watch | **用我方实现**；60f6b6a 的 `root_z_term` 接在我方签名上，取 `ref_bp[:, root, 2]` | `test_root_ori_term`（本提交 19 项） |
| **f1f364b** 10-25 | 开关关闭时 `cfg_dict` 不写 `root_ori_*`；新增 `pi/root_ori_*`、`wm/root_{ang,yaw}_err_step{1,k}`；watch 关键词收紧（floor 6,208 条，spin 446 条）；启动器加 `--rdzv-conf is_host=1` | — | — | — | ✓ | +4 项，`test_root_ori_term` 合计 23 项 |
| **42b211e** 10-25 | 只改 `launch_bj31_oriterm.example.sh`：ckpt 路径和 md5 可以用环境变量覆盖 | — | — | — | ✓ | — |

### 1.4 语义冲突，以及 7d8d85b 怎么处理的

| # | 冲突 | 我方 | master 60f6b6a | 7d8d85b（已读代码） | 残留风险 |
|---|---|---|---|---|---|
| 1 | `--lr_scale` | 基础 LR 倍数；生产 run 全部为 1.0 | 7afc772：续训载入后覆盖运行时 lr_scale | 我方语义；master 的改名为 `--resume_lr_scale`（train.py:170/229/381） | **高**，见 §1.2 #1 |
| 2 | EE 默认 | 不给旗标 = 开，0.25 | `--ee_term_z` 默认 0，即关 | 随我方 | 低：生产命令都显式写了 `--ee_term_z 0.25` |
| 3 | 看门狗信号 | norm（默认）/ alive，加 floor | 默认 robust（逐步钳到 ±r_max） | 三种都可选，默认 norm | 中，见 §1.2 #3 |
| 4 | warmup / 基线 / sustain | warmup 500，从第 1 轮起计基线，一触发就回滚 | warmup 1500，warmup 结束才计基线，持续 200 轮才算 | 默认随我方；master 的行为要用 `--wd_warmup 1500 --wd_baseline_after_warmup --wd_sustain 200` 打开 | sustain 计数作用在 rank 均值上，各 rank 一致（已读 `_watchdog`） |
| 5 | 非有限的 r | 当作暂态，`wd_nf_patience` 100 | 立即回滚 | 我方 | 无 |
| 6 | 骨盆终止 | ecc551f | f82e276 | 我方 | 如果以后改用 master 的签名，`fns.py` 和 `check_fk.py` 要一起改 |
| 7 | 采样器钳位 | 5bb8c0d | 5f020a9 | 已合一 | 无 |
| 8 | 簇先验 | — | 乘法 | 混合，mix 0.5 | 中，见 §1.2 #4 |
| 9 | WM 统计 / step_mask | `stats_pairs`（窗口掩码） | `update_stats` 用 `step_mask` | `stats_pairs` 改用 step_mask，`update_stats` 保留为包装 | 无 |
| 10 | 嫁接时的随机数 | — | 全局 RNG，各 rank 种子不同，状态会分叉 | `torch.Generator(device).manual_seed(0)`（trainer.py:1091/1110） | 已在 2 rank 上验证 |
| 11 | `critic.err_norm` | — | 直接 `.update()`，DDP 下会分叉 | `dist.rms_update`（trainer.py:660） | 已验证（B、C 组） |
| 12 | policy_delay / freeze | — | `update_policy` 返回 {} 时跳过看门狗 | `if m_pi and self._watchdog(...)`；step_policy 只依赖 cfg | 无 |
| 13 | term_scale | 课程关时传 None | 总是传 ones 张量 | 我方 | 数值相同 |
| 14 | 默认数值路径 | — | e59e947 把 `window_loss` 改成按步求和再除以计数 | 用 master 的 | v2 与合并后不逐位：6 轮后 obs_norm.mean 最大差 0.029。随机输入下 window_loss 旧式 15.026455879、新式 15.026456833〔复跑〕 |
| 15 | 续训时比对哪些设置 | `_resume_diffs`：robot、variant、先验、钳位、课程、H、heads、clip 列表 | 没有 | 只到这一层 | 中，见 §1.2 #2 |

### 1.5 可复用的验证方法

**单进程逐位对照**〔复跑〕。三份代码各跑一次：

```
PYTHONPATH=<code> ~/supermimic/mjlab/.venv/bin/python scripts/train.py --motion_dirs ~/supermimic/data/lafan1 \
  --device cpu --num_envs 16 --rollouts 16 --wm_windows 64 --wm_batches 2 --iterations 6 --log_interval 2 \
  --no_eval --ckpt_interval 0 --no_compile --ee_term_z 0.25 --seed 0
```

- 比较工具：`$S/a2test/cmp_ckpt.py` 比 final ckpt 的全部张量；`$S/a2test/cmp_tb.py` 比非计时 TB 标量。
- 结果：60f6b6a 与 e6ba84b 的 ckpt 逐位相同。两边独有的键都只是日志：master 有 `wm/step_valid_frac`；合并后多出 guard/*、alive、watch、root_ori、`wm/root_*`。

**2 rank gloo torchrun 端到端**〔复跑〕。脚本 `$H/ddp/run_ddp.sh`，日志 `$H/ddp/{A,B,C}.log`。公共参数是 `--param_check_interval 2 --ckpt_interval 4 --eval_interval 4 --eval_clips 3`。

| 组 | 关键参数 | 结果 |
|---|---|---|
| A：生产开关 | `--root_ori_term 1.0 --sampler_clamp_returns --wd_signal imagined_reward_alive --wd_reward_floor -1 --es_eval_fail_ratio 1.5 --wd_force_at 10` | 起点 policy 哈希 `7af7000e5151`，与单卡 seed0 相同。it 10 两个 rank 同时回滚到 model_8，回滚后哈希 `917619962a63`，等于第 8 轮的哈希。LR 降到 1e-4，final 状态一致 |
| B：master 侧功能全开 | `--critic_errors --goal_ref_root --fixes --sampler_cap 50 --sampler_clusters 4 --sampler_boot_mean --root_z_term 0.25 --wd_signal imagined_reward_robust --wd_sustain 200 --wd_warmup 1500 --wd_baseline_after_warmup` | goal 74 维，critic 误差特征 57 维。开头、每 2 轮、final 的状态比对全部一致 |
| C：从 A 的 model_8 续训，带嫁接 | `--goal_ref_root --critic_errors --resume_lr_scale 0.25` | 打出两条 WARNING（`heads.goal_ref_root`、`critic_errors`）。嫁接 64→74、0→57，goal_norm 完成标定，开头状态一致。顺带复现了 §1.2 #9 里"lr_scale 为 1.0 时 LR 打印不准"的问题 |

**CPU 单测**〔复跑，汇总在 `$H/testlogs/SUMMARY.txt`〕

- 用 gmr python（`/home/user/miniconda3/envs/gmr/bin/python tests/<f>.py`，没有 mjlab 和 tensorboard，测试会打桩）跑我方 10 个文件，共 95 个 ok：ddp_sync 3、early_stop_alive 11、ee_termination 7、merge_master_1025 15、motions_names 3、nan_guard 5、root_ori_term 23（36 s）、sampler_clamp 8、v2_merge 11、watchdog_floor 9。
  - 但有**静默跳过**：`test_ddp_sync` 跳过了 4 项，2 进程和 3 进程各缺采样器合并（缺 mjlab）和 trainer 跳步（缺 tensorboard）两项；`test_nan_guard` 跳过 1 项。退出码仍为 0。
- master 带来的 4 个测试（`test_graft_inputs`、`test_wm_body_frame`、`test_wm_graft`、`test_sampler_cap_clusters_rootz`）：在 e6ba84b 上用 gmr python 导入就失败；用 **`PYTHONPATH=. ~/supermimic/mjlab/.venv/bin/python`** 跑，四个全部通过。b0ddc5b 已经给它们加了自定位和打桩，我没复跑。
- 用 mjlab venv 加 `PYTHONPATH=.` 跑 `test_ddp_sync` 和 `test_nan_guard`，无跳过，全部通过。
- **推荐写法**：`for t in tests/*.py; do PYTHONPATH=. ~/supermimic/mjlab/.venv/bin/python "$t"; done`。

### 1.6 bj31 GPU 冒烟清单（还没做；按规矩我不在 bj31 起进程）

1. **部署代码**：`git archive <合并 commit>` 后 rsync 到一个新目录，例如 `~/supermimic/supermimic_repro_c1`，写 `COMMIT` 文件，逐文件核对 md5。不要覆盖正在被 run 或 auto_v4 使用的目录。
2. **单测**：在 bj31 上用 `~/supermimic/mjlab/.venv/bin/python` 加 `PYTHONPATH=.` 跑全部 14 个测试文件（真 mjlab，CPU）。
3. **8 rank NCCL 冒烟**：`NPROC=8 bash scripts/launch_bj31_ddp.sh smoke_c1 <§3.3 的参数，去掉 --resume> --iterations 300 --eval_interval 200 --ckpt_interval 200 --param_check_interval 100`。逐项核对：
   - [ ] 日志里有 `NCCL version 2.30.7+cuda13.3`；
   - [ ] 每张卡只有本 rank 一个进程；
   - [ ] 每次都出现 `[dist] … identical`；
   - [ ] ms/it 与 §3.4 的数字相符；
   - [ ] rank 0 评测时其余 rank 在等，评测后继续；
   - [ ] rank 日志的大小：用来验证 60f6b6a 的 warn_overflow 修复是否生效；
   - [ ] 用 model_200 `--resume` 再起一次。
4. **从 174000 续训**：用 §3.3 的命令，加 `--iterations 174200 --param_check_interval 50`。对照旧代码上 C2 臂的日志，应满足：
   - 打印 `lr_scale 0.0625 restored from checkpoint`；
   - 打印 `resumed … at iteration 174000 (policy lr 1.310e-05 = lr x lr_scale 0.0625 x cosine 0.2349)`；
   - **没有** `WARNING resume from another setup`；
   - `MotionLibrary: 255008 clips, 265826690 frames …, 5428418 bins`；
   - EE on（4 个体）、root-orientation on、`[dist] start: … identical`；
   - **不能触发嫁接**：goal 应为 64 维，WM 为航向系。
5. **同机双任务**：起两个 4 卡任务，`PORT` 分别为 29521 和 29522，间隔 ≥ 300 s。再用同一个 PORT 起第三个任务，应当立即失败。最后用 `stop_bj31_ddp.sh` 停掉，确认没有残留进程。
6. **导出链**：`REPO=<新目录> XS=$REPO/scripts RUN_DIR=… TAG=… bash /data2/supermimic/eval_export/scripts/export.sh 174000`。期望：
   - `check_onnx_vs_torch` 通过；
   - ONNX sha256 与 `1aa8742b…` 一致，或至少数值等价；
   - `eval_ckpt.py --ckpt model_174000.pt` 能读出 root_ori 是开的。

---

## 2. DDP 实现要点：必须保留的不变量

下面每一条都在 7d8d85b 里核到了。改 trainer 时请逐条对照，勾选版在 `MERGE_CHECKLIST.md` §C。

1. **单卡路径不变。** `DistCtx.from_env` 在 `WORLD_SIZE ≤ 1` 时返回禁用的上下文，所有 helper 退回单卡原代码。检查方法：§1.5 的单进程逐位对照。
2. **设备绑定的顺序。**
   - 先 `torch.cuda.set_device(LOCAL_RANK)`；
   - 再用 `torch.zeros(1, device)` 绑定 CUDA context，让 Warp 的默认设备落在本卡；
   - 然后 `init_process_group("nccl", device_id=…)`；
   - 另建一个 **gloo 组**，专门做 barrier 和 Python 对象广播。
   - 两个组的超时都是 `--dist_timeout_min` 120 分钟，必须长于 rank 0 的一次评测加一次建缓存。
   - `args.device = cuda:LOCAL_RANK`。CPU 测试全部用 gloo。
3. **种子。** 每个 rank 的 torch、numpy、env 种子都是 `seed + rank`，各自有一套 env、replay 和动作噪声。评测子集用 `random.Random(cfg.seed)`，不加 rank。rank 0 的起点与单卡 seed0 相同，初始 policy 哈希都是 `7af7000e5151`。
4. **文件表和 log_dir 只由 rank 0 解析，再经 gloo 广播。** 原因有两个：缓存键依赖有序的路径表；8 个 rank 的时间戳可能跨秒。`--motion_list` 只要缺一个文件，rank 0 就在广播前报错，否则其余 rank 会永远卡在 barrier。
5. **rank 0 先建 env 和缓存，其余 rank 在 barrier 等，然后各自 mmap 同一份缓存。** 合并后，`bin_clusters` 的特征和 k-means 结果也缓存在库目录里，同样由 rank 0 先生成。
6. **开训时广播初始参数。** `broadcast_modules_(policy, critic, wm)`，包括参数和 buffer。
7. **梯度同步。**
   - 每次 backward 之后，把非空梯度拼成一个平铺张量，`all_reduce(SUM)` 后除以 world，拷回，再走原来的裁剪和 `opt.step()`。
   - backward 的次数：WM 每轮 `batches_per_update` 次，policy 1 次，critic `epochs×minibatches` = 20 次。
   - **不用 DDP 包装**，因为编译的是自由函数，不是 module 的 forward。
   - **是否全有限，在规约后的张量上判断。** 任一 rank 出现 NaN/inf，或者求和溢出，规约结果在所有 rank 上都非有限，所以各 rank 的结论一致，不需要额外通信。各 rank 同时丢弃梯度、跳过该模块的 step，计数记在 `guard/skips_*`。
   - 各 rank 的 WM 和 critic 损失是"各自掩码均值的均值"，没有按有效样本数加权。这是设计上的选择。
8. **归一化器只能走 `dist.rms_update`。**
   - 做法：各 rank 算本批的 (n, mean, var)，一次 `all_gather`，然后每个 rank 按 rank 顺序用 Chan 公式（float64）合并，再调 `update_moments`，结果逐位一致。
   - 某个 rank 的批为空时也必须参与通信，否则会死锁。
   - 目前覆盖：policy 的 obs/goal（一次通信）、WM 的 in/tgt/err（`stats_pairs`）、`critic.err_norm`。
   - **今后新加任何 RunningMeanStd，都必须走这条路径。** master 的 `critic.err_norm.update()` 就是在自动合并里混过去的例子。
9. **凡是会改变学到状态的随机数，各 rank 必须相同。** 例如嫁接用的 `_graft_wm` / `_calibrate_goal_norm`，取库帧时用 `torch.Generator(device).manual_seed(0)`。
10. **采样器合并。**
    - 两次合并之间，各 rank 用本地统计采样。
    - 每 `sampler_sync_interval` 轮（默认等于 log_interval，即 50），以及每个 ckpt 槽和自检槽，做一次 `merge_sampler_bins`：
      - dV = V − V0；
      - `bin_G` 取本期被访问过的 bin 按访问次数加权的平均，没人访问的 bin 保持合并值；
      - `bin_visits` = V0 + ΣdV；
      - `bin_weight_hist` 取各 rank 平均。
    - 开训、续训、回滚之后，`_dist_start` 广播 rank 0 的三张表（`bin_weight_hist` 不在 ckpt 里），重置合并快照，并做一次全状态比对。
11. **跨 rank 一致性自检。**
    - `_state_groups` 把以下各组拷到 CPU，做 blake2b 哈希，经 gloo all_gather 后与 rank 0 比对：
      - policy、critic、wm 的 state_dict（含归一化器）；
      - 三个 Adam 的 state 和 param_groups；
      - 采样器（合并后的三张表）；
      - `sampler_prior`；
      - `counters`；
      - `library`（`MotionLibrary.fingerprint`）；
      - 开课程时加 `curriculum` 组。
    - 任何一组不同，所有 rank 都抛 RuntimeError，由 torchrun 收掉整组。
    - 时机：开始、每 `param_check_interval`（1000）轮、每次回滚后、结束时。
    - 开销：bj31 全量每次 0.3–0.4 s。
    - 灵敏度：注入 1 ulp 的故障，下一次检查就报 `policy differs on ranks [1]`。
    - **新增任何学到的状态，都要加进 `_state_groups`。**
    - 局限：不检查各 rank 的环境配置。
12. **rank 0 独占的操作和 barrier。**
    - 只有 rank 0 写 TB、`train_cfg.json`、`motion_files.txt`，存 ckpt，做评测，打印。其余 rank 在 **gloo barrier** 等待。
    - **是否进 barrier，只能由 cfg 和迭代数决定**，不能依赖 rank 0 运行时的状态（例如 `eval_enabled`），否则会错配、死锁。
13. **日志聚合。** 先用 crc 校验各 rank 的键集合一致，再取均值；回合统计和 watch 计数求和；`dist/*_max` 记最慢的 rank；`time/env_fps` 是全部 rank 的合计。
14. **早停判决一致。** rank 0 的评测摘要（出错时为 None）经 gloo `broadcast_object` 发给所有 rank，各 rank 用同一份历史做同一判决，在同一轮退出循环。存 `model_best_eval.pt` 时，其余 rank 在 barrier 等。
15. **看门狗判决一致。**
    - 每个 rank 先按 floor 钳一次，再对 r 和 critic loss 做 `reduce_scalars(mean)`。EMA、窗口、中位数、sustain 都作用在 rank 均值上。
    - 回滚时，所有 rank `load` 同一个文件，置 `buffer.size = 0`，LR 乘以 `wd_lr_factor`，再调 `_dist_start`。
    - `_note_reward_floor` 在第 k ≤ 3 次以及之后每 100 次时，各 rank 一起 `all_gather` 原始值，用来定位是哪个 rank 越界。
16. **续训。** 所有 rank 读同一个 ckpt。`_resume_diffs` 的告警只由 rank 0 打印。开训时 `_dist_start` 再比对一次。
17. **进程生命周期。**
    - torch elastic 用 `start_new_session=True` 启动每个 rank，所以只杀 agent 的进程组会留下孤儿 rank 继续占卡。
    - bc9d507 让 rank 设 PDEATHSIG；如果 prctl 生效前 agent 已经死了，rank 就自杀。
    - 停止一律用 `stop_bj31_ddp.sh`。正常结束时才调 `dctx.close()`。
18. **启动器 `scripts/launch_bj31_ddp.sh`**（与 v2 逐字节相同）。
    - 用 `setsid nohup` 起 torchrun。默认 `--standalone`。
    - 设了 `PORT`（别名 `RDZV_PORT` / `MASTER_PORT`）时，改用 `--nnodes 1 --rdzv-backend c10d --rdzv-endpoint localhost:$PORT --rdzv-id <run> --rdzv-conf is_host=1`。**`is_host=1` 不能去掉**：没有它，端口被占时会静默以 client 身份接入别人的 TCPStore，结果是停一个任务连带杀掉另一个。
    - 设了 `CUDA_VISIBLE_DEVICES` 子集时，NPROC 默认取子集里的卡数，NPROC 更大则报错。
    - `THREADS` 每 rank 默认 24。`PYTHONPATH` 固定为脚本所在的仓库。清掉 `LD_PRELOAD`、`PYTHONPATH` 和代理变量。`NCCL_DEBUG=VERSION`。
    - 日志：控制台只放 rank 0（`runs/<run>.log`），各 rank 的完整输出在 `runs/<run>_ranks/`，进程组号写在 `runs/<run>.pgid`。同名日志已存在时拒绝启动。

---

## 3. 训练配方与结论

### 3.1 统计口径与噪声

- **v4 的定义**：评测站 `g1_motionatlas_v4` × `kimodo_g1_motion_official_v3`，共 3134 段，1 段 = 0.0319pp。失败分三类：
  - `body_pos`：本文叫 EE，即末端高度失败；
  - `root_ori`：骨盆完整角（含 yaw）> 1.0 rad；
  - `root_pos`：骨盆 z 误差 > 0.25。
- **单个 ckpt 的噪声约 ±0.3–0.5pp**。三个例子：
  - 92k 和 96k 同为 88.70%，但有 64 段在成功和失败之间互换；
  - bj27 188k 的 +0.44 到 192k 就消失了；
  - 8 卡主线 176k 是 91.32%，前后两个点分别是 91.83 和 91.64。
- **只用同起点配对差，或三点及以上的均值下结论。** 单点差 ≤ 0.5pp 一律视为噪声。参考平台带：r15ws 140–154k 共 8 个点，90.53 ± 0.12。
- **13 个 ckpt 的整体情况**（T 168k、8 卡主线 170–182k、C2 176–184k）：
  - 均值 91.56%；
  - 稳定成功 2711 段，成败摇摆 247 段，从未成功 176 段；
  - 13 个 ckpt 成功段的并集上限是 94.38%。
- **内部评测 `eval/failed`**（1024 条固定名单，确定性执行）：
  - 用处是报警：在退化段的 32 个 ckpt 上，它与 v4 的相关系数是 −0.984，140k 时已从约 10 升到 20，比悬崖早约 2.8 万轮；
  - 局限是分辨率：平台期它分不出 ±0.5pp。8 卡主线 184–188k 的 v4 掉了约 0.4，内部评测一直是 6/1024。
- **训练侧指标不能用来选点**：想象回报、WM 误差、训练失败率在退化全程都很平滑。
- **对照的设计方法**：同一个起点 ckpt、同卡数、同一份新代码，C 臂只关掉开关。两臂错开 ≥ 300 s 启动，端口不同。判据写成：
  - 后段三点均值 T ≥ C + 0.2pp；
  - 机制项，例如 root_ori ≤ 85，且比 C 少 ≥ 15；
  - EE 增加 10–40 条算改标签，不算退步。

### 3.2 哪些有效、哪些无效、哪些未知

| 类别 | 改动 | 证据 | 效应（v4 pp） | 噪声与混杂 | 建议 |
|---|---|---|---|---|---|
| 有效 | 骨盆全角朝向终止 `--root_ori_term 1.0 --root_ori_term_temp 0.05` | bj31 同起点 4+4 卡，六个点 T 全部领先 | 168–172k 三点均值 **+1.12**（91.66 vs 90.54），六点均值 +1.08；root_ori 44 vs 约 101，EE +16 | 远大于 ±0.5。在 bj27 上只有 +0.2–0.3，因为 EE 多了约 25 条，抵掉了收益 | 保留，作为默认开 |
| 有效 | EE 终止（环境硬终止 + 想象软终止） | bj27 同起点单点，40k：77.41 vs 68.6；8 卡 20k 的 EE 失败 346，旧配方 913 | **+8.8** | 单点，但效应远大于噪声 | 保留 |
| 有效 | H 从 32 改到 48 | bj27 88k：86.98 vs 85.07，MPJPE 0.100→0.078；H64 为 86.73 | **+1.9** | 与数据、课程、LR 混杂 | 保留 H48；不用 H64 |
| 有效 | 采样器回报钳位 | bj27 136k：89.85 vs 89.02（单点）；bj31 r52 止住了崩塌，但没超过此前的峰值 | +0.8 / 0 | 单点 | 保留 |
| 有效 | 来源加权采样 | bj27 48k：82.29 vs 78.62；剂量翻倍无额外收益 | +3.7 | 单点 | 保留 |
| 有效 | 从单卡最好链热启动，再用 8 卡续训 | ws86k：87.01 → 88.70；r15ws：89.85 → 90.94；与 bj27 r17 同起点比 +0.3/+0.4 | +0.2 到 +1.7 | r17 掉在 LR 陷阱里，对照不干净 | 可用，但不能确定比单卡链更好 |
| 有效 | 看门狗 `--wd_reward_floor -1` + `--wd_signal imagined_reward_alive` | ws86k 误回滚 3 次，每次 LR 减半；之后各线 0 次 | 不再误降 LR | — | 保留 |
| 已实现，从未触发 | 早停 `--es_eval_fail_ratio 1.5 --es_patience 2` | r15ws 之后各线只记过单次 BAD 1/2（共 5 次） | 离线回放显示会在 152k 触发 | 失败数小时过敏 | 保留，当作退化保险；修过敏问题 |
| 无效 | 8 卡从零训全量三层数据（H32，不加权，33 kg 模型） | 52k 峰值 84.75，之后退化（60k 81.78） | 比 bj27 单卡 v2 从零到 86k 的 87.01 低 2.3 | 配方不同 | 不要这样从零开 |
| 无效 / 有害 | H32 长续训（执行噪声 σ=0.1） | ws86k_b：116k 88.51 → 212k 79.61 | −8.9 | 机制已用干预实验复现（§5.2） | 只信确定性指标；早停，或给噪声退火 |
| 无效 | root_ori 收紧到 0.8 | bj31 同起点，五对点全部为负 | −0.28；root_ori −10，但 EE +12、root_pos +7 | bj27 是 +0.18（四点）；叠加地面曝光后更低 | 保持 1.0 |
| 无效 | 骨盆高度（root_pos）终止，包括 master 的 `--root_z_term` | 42 条逐帧复跑，34/42 的 torso z 判据已经先触发；40/42 是机器人低于参考 | 预期约 0 | — | 不做；如果一定要试，必须做 T/C |
| 无效 | 原始看门狗（读不加权的想象奖励） | 误报 3 次 | — | — | 已替换 |
| 无效（bj27） | 后期 lr_scale 提到 0.5 / ckpt 权重平均 / UTD 2.5 / 本体系 WM / condim 3 / 难集放宽终止课程 | bj27 各组对照 | −0.4 / 0 / 0（且慢 15%）/ 0 / −0.7 / −0.6 | 多为单点 | 不再试 |
| 未知 | 执行动作噪声是不是退化的**原因** | 只有干预实验（加噪能救回），没有"不加噪"的对照训练 | — | 另一种解释是"WM 看得见抖动，只是奖励容忍它"，没有排除 | 见 §7 |
| 未知 | 8 卡是否会加快漂移 | 8 卡主线 184–188k 均值 91.17，比 170–182k 低 0.41；同轮数下 C2（4 卡）高 0.12，182–186k 高 0.29–0.42 | — | 在噪声边缘 | 需要多点同起点的 4 卡对 8 卡对照 |
| 未知 | 8 倍 batch 的真实样本效率 | "1.6 倍/轮"是经验估计 | — | 配方混杂 | 需要同配方同起点的单卡对 8 卡对照 |
| 未知 | bj27 的 ref_root 和地面曝光叠加到本线的效果 | 模型估计分别为 −0.10 和 −0.30pp，没有实测 | — | 估计方法粗，脚本中途报错 | 见 §7 |
| 未知 | r15ws 起步 LR 是产出那个 ckpt 时 LR 的 4–17 倍，影响多大 | 没做 A/B | — | — | 可以忽略，或做一次短 A/B |

### 3.3 推荐的续训配置与 LR 算法

**起点**：model_174000，md5 `2bd84993…`，路径 `/data2/supermimic/warmstart_ori8g_model_174000.pt`。

**生产命令**取自 `runs/g1v2_ro10_8g_168k.log` 第 2 行，代码 42b211e，启动器包了一层 `torchrun --standalone --nproc_per_node 8 … scripts/train.py`：

```
--log_root /data2/supermimic/logs --cache_dir /data2/supermimic/cache
--run_name <新名字> --num_threads 24 --motion_list /data2/supermimic/prep_1025/g1v2_r15_255008_bj31.txt --robot g1
--robot_variant eval+evalcontact --ee_term_z 0.25
--source_weights noetix_kungfu=10,noetix_kungfu_dance_show=15,sportsmeeting=10,lafan1=15,g1_all=10,mixamo_cn=2,virtual_point=3
--name_weights tengkong=200,kongfan=200,shoufan=200,erqijiao=200,danpaijiao=200,xuanfeng=200,waibailian=100,lihetui=100,saotang=100,ceti=100,houti=100,zhengti=100
--sampler_clamp_returns --num_envs 4096 --rollouts 4096 --H 48 --wm_batches 10 --wm_windows 2048 --wm_k 8 --lr 8.9252e-4
--lr_decay_iters 250000 --iterations <绝对终点> --seed 0 --wd_drop 0.2 --wd_reward_floor -1.0 --wd_signal imagined_reward_alive
--num_workers 64 --eval_clip_names /data2/supermimic/lists/eval_1024_g1atlasv1.txt --eval_interval 2000 --eval_envs 1024
--es_eval_fail_ratio 1.5 --es_patience 2 --ckpt_interval 2000 --log_interval 50 --param_check_interval 1000
--resume /data2/supermimic/warmstart_ori8g_model_174000.pt --root_ori_term 1.0 --root_ori_term_temp 0.05
```

这个 run 的 `train_cfg.json` 里还有：`world_size 8`、`base_lr_scale 1.0`、`wd_warmup 500`、`sampler_sync_interval 0`、`dist_timeout_min 120.0`、`action_noise_std 0.1`。heads 里含 `ee_termination True / ee_term_z 0.25 / ee_term_z_temp 0.02 / root_ori_term 1.0 / root_ori_term_temp 0.05 / root_ori_body ""`。

**在合并后的代码上用这条命令的注意事项**

- **不要加 `--lr_scale`。** 需要改运行时倍数时用 `--resume_lr_scale`，并且先读下面的 LR 算法；bj27 的经验是后期调高 LR 有害。
- 看门狗 alive 信号加 floor 已经显式写在命令里，与合并后的默认值无关。
- **不要随手加** `--fixes`、`--goal_ref_root`、`--critic_errors`、`--sampler_cap`、`--sampler_clusters`、`--sampler_boot_mean`、`--root_z_term`。它们要么触发嫁接，要么改变前向或采样，而续训检查只覆盖其中一部分（§1.2 #2）。要加就按 T/C 对照来做（§7）。
- 合并后的数值轨迹与 42b211e 不逐位一致（§0.4 第 4 条），所以同样的命令续训，结果只能在统计意义上对照。
- `--iterations` 是**绝对终点**。从 160k 续到 172000，最后的 ckpt 就是 `model_172000.pt` 加 `model_final.pt`，没有 GMT 那种差一的问题。

**LR 算法**

实际 policy LR = `--lr` × **ckpt 里继承的 lr_scale** × λ(it)，三个优化器共用同一套规则。

- 余弦系数：λ(t) = 0.03 + 0.97·½(1 + cos(π·t/250000))。
- 常用取值：λ(52k) = 0.900，λ(86k) = 0.744，λ(104k) = 0.6415，λ(136k) = 0.4482，λ(160k) = 0.3085，λ(168k) = 0.2655，λ(174k) = 0.2349，λ(188k) ≈ 0.17。
- 看门狗每回滚一次，lr_scale 减半，并且会写进之后的所有 ckpt。本线的 ckpt 带 `lr_scale 0.0625`，继承自 bj27 r15 的 4 次回滚。
- 例子：
  - 8.9252e-4 × 0.0625 × 0.2349 = **1.310e-5**，即从 174k 起步；
  - 168k 起步是 1.481e-5；
  - 160k 起步是 1.721e-5；
  - 从 188k 起步约 9.5e-6，这个值是推算的。
- **陷阱**：照 r15 配方写 `--lr 5e-5`，在本线 ckpt 上实际 LR 只有约 5e-7。bj27 的 r16–r18 就掉进过这个坑，继承了 1/16，实际 LR 只有约 1.1–1.4e-6，那几轮的结论作废。
- **核对方法**：从 r15ws 起，日志会打印 `resumed … (policy lr X = lr x lr_scale Y x cosine Z)`，每次续训后先看这一行。

**每卡和全局规模**：每个 rank 跑 4096 env、4096 条想象 rollout、自己的 replay。8 卡全局 batch 是 32768/32768，4 卡是 16384/16384。**LR 不随卡数缩放**。

### 3.4 吞吐

bj27 和 bj31 是同一型号的硬件：RTX PRO 5000 72GB Blackwell，256 核。下表的 fps 指全局 env-step/s，等于 world × 4096 ÷ (ms/it)。

| 配置 | run | ms/it p50（p10–p90） | 耗时分解 c / wm / pi | fps |
|---|---|---|---|---|
| 单卡 H32 | bj27 g1v2_scratch | 约 511–520 | 43 / 166 / 260（LAFAN run 的数据） | 约 7.9k |
| 单卡 H48 | bj27 h48_r9 | 720 | — | 约 5.7k（推算） |
| 8 卡 H32 | g1_full_ee_ddp8 | 623（597–833） | 58 / 205 / 360 | 52.1k |
| 8 卡 H32 | ws86k_b | 636（630–767） | 60 / 207 / 368 | 51.2k |
| 8 卡 H48 | r15ws / 8g_168k | 951–952 / 892–895 | 60 / 207 / 682；61 / 223 / 609 | 34.2k / 36.5k |
| 4 卡 H48，两臂同机 | T / C | 1055 / 988–996（p10 816 / 782） | 65 / 254 / 735；67 / 256 / 673 | 15.4k / 16.3k |
| 4 卡 H48，两臂同机 | T2 / C2 | 1117–1176 / 1092–1125（p10 约 830 / 780） | 71 / 273 / 832；72 / 279 / 773 | 13.7k / 14.4k |

- 8 卡每轮比单卡多 20–32%。估算的通信只有 25–30 ms，其余开销的来源没有测过。
- 4 卡的 p50 偏高，原因是两臂和 auto_v4 导出在抢 CPU。看 p10，4 卡和 8 卡相近。
- **含评测、存 ckpt 的平均墙钟**：g1_full 0.73 s/it，ws86k_b 0.75，r15ws 1.17，8g_168k 1.37，T/C 1.36/1.21，T2/C2 1.46/1.30。每 2000 轮出一个 ckpt，约需 40–60 分钟。
- 显存与内存（H32 冒烟时测的，H48 没有单独测）：每卡约 23 GB；主机内存稳态 43–56G，编译期峰值 122G。前 50 轮在编译，0.8–4 s/it 属正常。

---

## 4. 实验时间线

### 4.1 Run 总表

| run | 时间（北京） | 起点 | commit / bj31 代码目录 | 卡 / H | 配方要点 | 起步 policy LR | 迭代 | v4 | 结局 |
|---|---|---|---|---|---|---|---|---|---|
| `g1_full_ee_ddp8` | 10-22 17:06 → 10-23 约 11:37 | 从零 | bc9d507 / `supermimic_repro` | 8 / 32 | 三层数据 365,855 条；只加 EE；mjlab g1 33.34 kg；不加权、不钳位；`wd_drop 0.08` | 2e-4 | 0→91,050 | 20k 83.18 / 40k 84.46 / 52k **84.75** / 60k 81.78 | 峰后退化；内部评测失败从 52k 的 4 涨到 88k 的 220 |
| `g1_full_ee_ddp8_r52` | 10-23 11:39 → 约 13:56 | 上一条的 52000 | 5bb8c0d / `_clamp` | 8 / 32 | 加 `--sampler_clamp_returns` | 4.5e-5 | 52k→61,750 | 56k 83.34 / 60k 83.73 | 止住了崩塌，但没超过峰值 |
| `g1v2_ddp8_ws86k` | 10-23 13:57 → 19:28 | bj27 g1v2_scratch 86000（87.01） | df670e6 / `_v2` | 8 / 32 | g1 层 254,518 条；来源加权 7 项；`eval+evalcontact` | 5.0e-5，3 次回滚后依次降到 2.24e-5 / 1.12e-5 / 5.41e-6 | 86k→105,700 | 92k **88.70** / 96k 88.70 | 看门狗误回滚 3 次 |
| `g1v2_ddp8_ws86k_b` | 10-23 19:29 → 10-24 约 18:20 | ws86k 的 104000 | ca9266e / `_v2wd` | 8 / 32 | 同上，加 `--wd_reward_floor -1` | 2.5e-5，到 212k 约 3.3e-6 | 104k→213,200 | 108–136k 在 87.46–88.51；168k 84.24；212k 79.61 | 持续退化，触发了 10-24 的诊断（§5.2） |
| `g1v2_r15ws_ddp8` | 10-25 12:31 → 22:17 | bj27 r15 136000（89.85） | 55e0eaf / `_r15ws` | 8 / **48** | 255,008 条（加 phc 功夫 490 条）；钳位；12 个功夫关键词；早停；alive 信号 | 2.5e-5 | 136k→166,000 | 160k **90.94**；15 点均值 90.57 | 为 T/C 让卡 |
| `g1v2_ro10_160k_T4` | 10-25 22:21 → 10-26 约 03:00 | r15ws 的 160000 | 42b211e / `_oriterm` | 4 / 48 | **`--root_ori_term 1.0`** | 1.721e-5 | 160k→172k | 168k **91.80**；168–172k 均值 91.66 | 赢，交付 168000 |
| `g1v2_ro0_160k_C4` | 10-25 22:27 → 约 02:39 | 同上 | 同上 | 4 / 48 | 对照，不开骨盆终止 | 1.721e-5 | 160k→172k | 168–172k 均值 90.54 | — |
| `g1v2_ro10_8g_168k` | 10-26 03:10 → 11:00 | T 臂的 168000 | 42b211e | 8 / 48 | 骨盆终止 1.0 | 1.481e-5 | 168k→188,500 | 174k **91.83（#1）**；170–182k 均值 91.58；184–188k 均值 91.17 | 为第二轮让卡 |
| `g1v2_ro08_174k_T4` | 10-26 11:01 → 15:4x | 8 卡主线的 174000 | 42b211e | 4 / 48 | 骨盆终止 **0.8** | 1.310e-5 | 174k→185,850 | 176–184k 均值 91.19 | 用户暂停 |
| `g1v2_ro10_174k_C4` | 10-26 11:07 → 15:39 | 同上 | 42b211e | 4 / 48 | 骨盆终止 1.0（对照） | 1.310e-5 | 174k→186,000 | 五点均值 91.48，六点 91.49；182k 91.74 | 正常结束 |

### 4.2 关键事件与决定

- **10-22：用户拍板在 bj31 训全量 G1**，数据是三层 365,855 条，8 卡数据并行，从零开始。
  - 代码依次为 c59c8ce、62058a5、b9415fc。
  - 第一次冒烟在第 118 轮出现 NaN：NaN 梯度经 all-reduce 写进了所有卡的权重，看门狗把带 NaN 的 model_200 当成健康 ckpt，回滚 6 次后放弃。之后修成 01c4d58。
  - 冒烟子任务两次卡死，bj31 空转约 15 h。
  - 16:1x 用户要求加回 EE，冒烟 850 轮，622–625 ms/it。17:06 正式起训。
- **10-23 上午：发现采样器塌缩**，`top1_prob` 恒为 0.5000。11:40 用户拍板，从 52k 起用钳位续训，即 r52。钳位后没有超过峰值，于是改为从 bj27 v2 的 86000 热启动，即 ws86k。依据是：
  - bj27 单卡 v2 已经到 87.01；
  - v2 配方带来源加权和评测站 35 kg 模型；
  - 按"1.6 倍/轮"估算，8 卡从零追上 87% 要十几个小时。
- **10-23 下午：ws86k 看门狗误回滚 3 次**，原因是死后 WM 发散，详见 §5.3。修成 ca9266e（floor），然后续训 ws86k_b。
- **10-24：ws86k_b 持续退化。** 诊断出机制是确定性闭环抖动，详见 §5.2。会话因额度中断约 21 h，无人值守，18:2x 用户叫停。代码 1662040 和 55e0eaf 加了早停和 alive 信号。
- **10-25 12:31：接 bj27 r15 热启动，改用 H48**，即 r15ws。摔倒差距的诊断（§5.4）指向骨盆全角判据盲区。代码 ecc551f、f1f364b、42b211e 实现骨盆终止。CPU 前置检查 D1 在训练模型上只有 39/101，没过 60% 的门槛，用户看过后仍批准。22:21 起第一轮 T/C。
- **10-26 03:10：T 赢**，按事先批准的方案把 T 臂 168000 放回 8 卡主线，174000 到了 91.83%。09:5x 用户批准第二轮（0.8 对 1.0），11:01 起跑，结果 0.8 五对点全负，保持 1.0。同时完成 root_pos 证伪（§5.5）和与榜首的差距分析（§5.6）。15:4x 用户暂停。

---

## 5. 诊断方法与工具

### 5.1 重置采样器塌缩

- **指纹**：TB 的 `sampler/top1_prob` 恒为 **0.5000**。
  - 由来：采样概率 = 0.5·均匀 + 0.4·softmax + 0.1·历史（历史是 softmax 的 EMA），见 repo_v2 `library_command.py` L388–393。softmax 和历史两部分全压在同一个 bin 上，加起来就是 0.5。
  - `ep_return_norm ≈ −1` 也是它造成的。
  - 默认配方下所有 G1 run 都有这个现象，包括 bj27 master 线；E2 从零训的那条是 0.35。
- **机制**（`trainer.py::_finish_episodes`）：
  - 片段超时时用 `boot = r_last/(1−γ)` 自举，即乘 100。
  - 末帧奖励由惩罚项主导时为负（joint_limit −2.0，action_rate −0.1），所以这个 bin 的 G 远小于 0。
  - 温度 25、g_max 10 时，权重 ∝ exp(−2.5·G)。在 627 万个 bin 中，一个 bin 只要比其余低约 6，就能独占 softmax。
  - 实例：
    - g1_full_ee_ddp8：`g1__sonic_230213_sitting_legs_bend_arms_back_start_003__A185` 的末 bin，G 从 −18.9（24k）降到 −25.0（40k）再到 −31.4（60k）；
    - ws86k：`g1__sonic_220721_sit_on_heels_start_002__A036_M` 的末 bin，约一半的重置落在这里。
- **修法与效果**：
  - `--sampler_clamp_returns`。离线模拟 60k：有效 bin 数从 1 变成 47,610；功夫的 softmax 份额约 12%，均匀采样时是 0.4%。
  - bj27 同起点 +0.8。bj31 的 r52 没超过峰值。
- **钳位后出现的新集中**：
  - 148k 时，G=0 的坐、蹲 bin 拿走 11.6% 的重置；
  - master 60f6b6a 的提交说明记录，r25 208k 上 90% 的 softmax 份额落在 1,588 个 bin 上。
  - 对应的新开关 `--sampler_cap`、`--sampler_boot_mean` 直接针对这个问题，但**本线没有任何对照数据**。
- **工具**：
  - `$S/sampler_probe.py`（RUN 写死为 g1_full）、`sampler_probe2.py`（RUN_DIR 和 CDIR 用环境变量传入）：按 MotionLibrary 规则重建 bin 表（只取 ok 且 end>start+1 的片段，每 50 帧一个 bin），重算 softmax，看质量落在哪些 bin 上。
  - `sampler_sim_clamp.py`：离线模拟钳位后的份额。
  - `sampler_r52.py`。
  - `$S/diag_1025/taskB/scripts/sampler_probe3.py`：额外加上来源/名字先验和 0.5/0.4/0.1 混合，按簇报告重置份额。
  - 都在 bj31 上只读、CPU 运行。**开了 cap 或 clusters 之后，要把新的 `sampling_probabilities` 公式同步进探针**，否则读数不对。

### 5.2 峰后持续退化：策略依赖训练时的执行噪声

- **干预实验**（审查已逐格复现）。CPU MuJoCo，用 bundle 场景，在 212k 内部评测失败的 77 条片段上统计失败数：

| 条件 | 88k | 116k | 164k | 168k | 188k | 212k |
|---|---|---|---|---|---|---|
| 确定性执行（同评测） | 5 | 5 | 8 | 25 | 39 | 52 |
| 执行动作加 σ=0.1 噪声（同训练） | 2 | 2 | 1 | 1 | 2 | 2 |

  - 212k 的剂量关系：σ = 0 / 0.02 / 0.05 / 0.1 时，失败数为 52 / 47 / 17 / 2。
  - 单独加噪的对比：只给执行动作加噪，失败 2；只给观测里的上一帧动作加噪，30；只加观测噪声，28；关节摩擦设为 0，44。
  - 结论：退化来自对执行端抖动的依赖，不是观测分布偏移，也不是摩擦。
- **相关代码**：真实采集时 `a_exec = a + 0.1·randn`，回放里存的也是带噪动作；想象 rollout、内部评测、v4 和部署都是确定性执行。
- **指纹**：
  - 确定性执行下，腿部动作变化率 0.111 → 0.271；加噪时一直约 0.105。
  - ‖∂a_t/∂a_{t−1}‖ 从 1.505 涨到 1.806，对角从 −0.028 变成 +0.063，即变成正反馈；谱半径 0.46 → 0.57。
  - 关节位置通路的增益也涨了 7–8%，所以动作历史通路是"涨得最多"，不是"唯一在涨"。
  - 抖动指标（固定成功集上 `body_local_lin_acc_error` 的中位数）：ws86k 线 2.66 → 4.85；bj27 H48 链 2.52 → 2.59；r15ws 2.83–3.08；榜首 2.33–2.36。
- **已排除**：归一化器、采样器分布、WM 误差、critic、吞吐，以及"LR 太高触发台阶"。另外，"只是训太久"与 bj27 H48 链抖动几乎不涨的事实矛盾。
- **没有定论**：
  - 没有"不加噪"的对照训练；
  - 另一种解释"WM 看得见抖动，只是奖励容忍它"没有排除；
  - H32 谱系、8 倍 batch、数据、LR 混在一起，分不开。
- **工具**：
  - `$S/diag_1024/taskC/scripts/sm_roll2.py`：clean / act / obs 三种条件，输入是导出的 bundle。
  - `jac2.py`（同目录）：算雅可比。
  - 两个脚本的 `sys.path` 都写死了 `~/supermimic/supermimic_repro_v2wd`，换代码后要改。
  - 抖动统计：`$S/jitter_r15ws.py`、`jitter_r15ws2.py`、`jitter_oriterm.py`、`$S/diag_1024/review/jitter_lines.py`。在 bj31 上只读评测站的 CSV，改脚本里的 `RUNS` 列表即可。注意固定集随参与比较的 run 而变，**只在同一张表内可比**。

### 5.3 想象奖励尖峰与看门狗误回滚

- **机制**：
  - 尖峰出在想象中"死后"的步：alive < 6e-8，c 在 fp32 下为 0。
  - WM 的 `step` 对归一化输入不钳位，SiLU MLP 在远处近似线性外推，而 joint_limit 是唯一无界的奖励项。
  - **尖峰不进策略梯度**，因为策略目标带 c 权重；但当时的看门狗读的是不加权的 `pi/imagined_reward_norm`。
  - `$S/diag_1024/taskB/wm_probe.py` 实测：同样扰动下 32 步后 Σ|q|，106k 为 1e11–1e12，212k 为 1e5–1e7。
  - 尖峰速率与掉分不相关。
  - 保留意见："死后 WM 发散"是从"想象回报不动"反推的，没有埋点直接证实。
- **修法**：`--wd_reward_floor -1.0`（ca9266e）加 `--wd_signal imagined_reward_alive`（1662040）。用测试模拟 r16 的尖峰：norm 信号回滚 5 次，alive 信号 0 次。r15ws 之后 bj31 的所有 run 都没有误回滚。
- **与 master e59e947 的关系**：e59e947 是逐步钳位后取 robust 均值，加 1500 轮内不记 best、持续 200 轮才算。它针对的是"续训时回放回填，把 best 顶高"，和我们针对的"死后发散"是**两种不同的失败模式**，两套都要保留。
- **TB 要盯的标量**：`guard/reward_floored`、`pi/imagined_reward_alive`、`guard/skips_*`、`guard/sim_nonfinite`。

### 5.4 摔倒差距：骨盆全角判据的盲区

| | 高度 | 朝向 | EE |
|---|---|---|---|
| 评测站 official_v3 | 骨盆 z 误差 > 0.25 | **骨盆完整角（含 yaw）> 1.0 rad** | 踝/腕 z > 0.25 |
| SuperMimic 训练（改前） | torso z > 0.25 | torso 重力 z 差 > 0.8（不看 yaw，约 78° 才判死） | 踝/腕 z > 0.25 |
| 榜首训练 | 骨盆 z > 0.15（参考低位时放宽到 0.75） | 骨盆完整角 > 0.447 | z > 0.15（同样自适应） |

- **148k 的诊断**：
  - 在评测判 root_ori 的那一步或之前，训练的任何判据只触发了 **5/101**；只把锚点换成骨盆，也只有 4/101。
  - 失败时 torso 的倾斜/yaw 误差是 0.26/0.20，骨盆是 0.86/0.60，腰把两者解耦了。
  - 判失败后继续跑 6 s，有 57/101 仍在跟踪。
  - 加 σ=0.1 噪声后仍有 76/101 失败，说明和 §5.2 的退化不是同一个机制。
- **软项的等效阈值**：软项是逐步 hazard，H 步内存活 50% 的角度是 **A − 4.23T**（H=48）。A=1.0、T=0.05 时约 0.79 rad；A=0.8 时约 0.59 rad。
- **CPU 前置检查**：
  - D1：训练模型上 39/101（38.6%），没过门槛，其中 24 条在训练模型上根本不失败；评测模型上 97/101。
  - D3：参考骨盆四元数 40/40 一致，误差 ≤ 6.7e-8。
- **T/C 对照要比 `eval/failed_with_ori`**。T 臂的 `failed` 包含 root_ori，C 臂的不含，两者不能直接比。`train/would_fail_ori` 只能在同一臂内看趋势。
- **工具**：
  - `$S/diag_1025/taskB/scripts/{trace_ori.py,trace_waist.py,trace_ana.py,list_ro.py,crit.py,xtab.py,runs_cmp.py}`；
  - `$S/diag_1025/oriterm/scripts/{d1_replay.py,trace_states.py,d1_evalmodel.py,d3_refquat.py,d1_analyze.py,common.py,prep_d1.py}`；
  - 输出在 bj31 `/data2/supermimic/diag_1025/oriterm/d1_*_summary.txt`、`d3.json`。
  - 评测站侧用 `/home/ubuntu/miniconda3/envs/noetix_eval/bin/python -I` 跑，训练侧用 mjlab venv。

### 5.5 root_pos 不是判据盲区（10-26）

- 方法：用评测站 runner、原版 xml 和 ONNX，对 174000 的 42 条 root_pos 失败逐帧复跑。
- 结果：
  - 42/42 都复现了；
  - torso z 判据在评测帧或之前触发 34/42，晚 1–5 帧 6 条，不触发 2 条；
  - 任一训练判据在评测帧或之前触发 35/42；
  - **40/42 是机器人低于参考**，即跳不到参考高度或塌下去，参考骨盆 z 常在 1.1–1.35 m。
  - 榜首在这 42 条上：EE 失败 21、成功 14、root_pos 失败 7。
- 结论：骨盆高度对齐终止预期没有收益，剩下的是能力问题（腾空高度）。
- 工具：`$S/diag_1026_rootpos/{mklist.py,trace_rootpos.py,summ.py,gap.py}`。
  - `trace_rootpos.py` 用评测站 conda python 运行，`ONNX=…/storage/models/<sha256>/model.onnx`，`OMP_NUM_THREADS=1 nice -n 19 timeout 3600`，分 6 块。
  - 输出 `out174.jsonl`，bj31 上的目录是 `/data2/supermimic/diag_1026_rootpos/`。

### 5.6 差距分析

- **方法**（`gap_next.py`）：在多个 ckpt 上算每段的通过概率，用"期望通过差"和稳健集（p_b ≥ 2/3 且 p_a ≤ 1/3）代替单点的"谁败谁过"。
- **单点对比**：

| run | v4 | 失败 | EE | ori | pos | MPJPE |
|---|---|---|---|---|---|---|
| 我们 174k | 91.83 | 256 | 171 | 43 | 42 | 0.0851 |
| C2 182k | 91.74 | 259 | 176 | 37 | 46 | 0.0850 |
| bj27 floor10_r25 208k | 91.70 | 260 | 189 | 40 | 31 | 0.0847 |
| 榜首 | 91.42 | 269 | 250 | 1 | 18 | 0.0853 |

- **与榜首配对**（174k）：
  - 我们败、榜首过 61 条：按原因 EE 32 / root_ori 15 / root_pos 14；按来源 gmt_eval 25、功夫 17；按失败时的参考相位，直立 23、腾空 23。
  - 榜首败、我们过 74 条，其中 72 条是 EE，倒立相位 40。
  - 两边都败 195 条，并集上限 93.78%。
- **分簇成功率**（我方 13 点均值 / bj27 floor 均值 / 榜首）：
  - 跳跃/腾空 **75.0 / 78.5 / 81.4**；
  - 起身/躺/地面 82.5 / 79.7 / 80.3；
  - 翻腾 28.1 / 28.5 / 16.1；
  - breaking 48.0 / 46.7 / 45.1；
  - 坐 84.6 / 92.0 / 87.0。
  - GROUND_GETUP 中的 SIDE_RIGHT：我们 2/7，榜首 7/7；GROUND_CRAWL：14/25 对 19/25。
  - **剩余差距最集中的是跳跃/腾空、右侧躺起身、手膝爬行。**
- **与 bj27 最好线配对**：174k 对 floor10_r25 208k 为 43 / 47 / 213。6 个 floor10 ckpt 的均值 91.38，我们 13 点均值 91.56。
- **工具**：
  - `$SP/diag_1026_next/{gap_next.py,gap_next2.py,gap_next.log,dump_union.py,union_fail.tsv}`；
  - bj31 `/data2/supermimic/diag_1026_next/{gap_next.log,gap_next2.log,gap_next.json}`。
  - ⚠ `gap_next2.py` 在后半段因 TypeError 中断，只有前几段输出可用。

### 5.7 脚本索引：建议拷进仓库

| 用途 | 脚本 | 运行环境 |
|---|---|---|
| ckpt / TB 逐位比较 | `$S/a2test/{cmp_ckpt.py,cmp_tb.py}` | 本机 |
| 2 rank DDP 端到端 | `$H/ddp/run_ddp.sh` | 本机，mjlab venv |
| 单测批跑 | `$H/run_tests.sh` | 本机 |
| v4 汇总表 | `$H/v4table.py`（读站点 API 和 per-segment CSV） | bj31 只读 |
| 采样器探针 | `$S/sampler_probe{,2}.py`、`sampler_sim_clamp.py`、`sampler_r52.py`、`diag_1025/taskB/scripts/sampler_probe3.py` | bj31 CPU |
| 退化：闭环干预、雅可比、WM 外推 | `$S/diag_1024/taskC/scripts/{sm_roll2.py,sm_roll3.py,jac2.py,…}`、`diag_1024/taskB/wm_probe.py` | bj31 CPU |
| 抖动指纹 | `$S/jitter_{r15ws,r15ws2,oriterm}.py`、`diag_1024/review/jitter_{lines,r52}.py` | bj31 只读 CSV |
| 骨盆和腰的轨迹 | `$S/diag_1025/taskB/scripts/*`、`diag_1025/oriterm/scripts/*` | 评测站 python 和 mjlab venv |
| root_pos 逐帧 | `$S/diag_1026_rootpos/*` | 评测站 python |
| 差距与稳定性 | `$SP/diag_1026_next/gap_next{,2}.py` | bj31 只读 |
| 导出、上传、分析 | `$S/evalsite/remote/{auto_v4.sh,export.sh,submit.sh,analyze.py,check_onnx_vs_torch.py,by_source.py,run_report.py,fail_stats.py,meta_diff.py,golden_check.py,export_bundle.py,export_eval_onnx.py}`，与 bj31 `/data2/supermimic/eval_export/scripts/` 同步 | bj31 |
| 守护 | `$S/guard_arm.sh <run> <rank数> <正常ms/it>`、`guard_sm.sh` | 本机，经 ssh 轮询 |
| 启动示例 | `$S/start_{oriterm_160k,oriT_8g_168k,round2_174k}.sh`，即 bj31 `/data2/supermimic/diag_1025/oriterm/start_*.sh` | bj31 |

---

## 6. 运维与坑

### 6.1 机器与环境（bj31）

- **硬件**：256 核，内存 743G，无 swap。GPU 是 8×RTX PRO 5000 72GB（sm_120），驱动 580.126.20，只有 PCIe P2P，没有 NVLink。NCCL all_reduce busbw 约 30 GB/s。
- **磁盘**（10-26）：根盘只剩 268G（已用 87%），**不要往根盘写缓存或日志**。`/data2` 剩 2.7T。`/dev/shm` 372G，SuperMimic 不用它。
- **venv**：`/home/ubuntu/supermimic/mjlab/.venv`，从 bj27 原样 rsync 过来，路径必须与 bj27 完全一致，因为 shebang 和 editable finder 都写死了路径。
  - 版本：python 3.13.16、torch 2.14.1+cu130、mujoco 3.16.0、warp 1.18.0、mujoco_warp 3.16.0。
  - NCCL 用 venv 自带的 **2.30.7+cuda13.3**，Blackwell 上 4 卡和 8 卡都长时间跑过，日志里 `NCCL WARN` 为 0。
- **规则**：**不要 `source ~/morty_workspace/gmt_env.sh`，也不要 LD_PRELOAD 机器上的 nccl_2277**（那是 cu12 版）。启动器已经 unset 了这些变量。启动后看 banner 是不是 `NCCL version 2.30.7+cuda13.3`。
- **Warp 内核缓存**：`~/.cache/warp`，cuda:0 首次编译 31.6 s。

### 6.2 代码部署

- **一个 commit 一个目录，不在原地改**：
  - 本机 `git archive <commit>`，`rsync -a --delete` 到 `~/supermimic/supermimic_repro_<tag>/`；
  - 写 `COMMIT` 文件，逐文件核对 md5（每次 57–64 个文件）；
  - 现有目录的对应关系见 §4.1 和 §8。
- 启动器设 `PYTHONPATH=<脚本所在仓库>`，所以 venv 的 editable 安装（指向 `supermimic_repro`，即 bc9d507）不起作用。但 `export.sh` 默认的 `REPO` 是 bc9d507，**导出 v2 及以后的 ckpt 必须传 `REPO=… XS=$REPO/scripts`**。
- **不要覆盖正在运行的 run 或 auto_v4 用的目录和脚本**，因为 bash 是边读边执行的。

### 6.3 名单与缓存

- **名单**：`/data2/supermimic/lists/`，带 `MD5SUMS`。
  - `atlas_full_blacklist_motion50_1010.json`
  - `train_files_365855_bj31.txt`
  - `eval_1024_g1atlasv1.txt`：按 basename，2 条带空格，40 条非 ASCII。
  - `g1v2_scratch_254518_bj31.txt`
  - `/data2/supermimic/prep_1025/g1v2_r15_255008_bj31.txt`：r15 起所有线用的数据，等于 254,518 加 `phc_extra_490_bj31.txt`。
- 训练名里有 1,490 条带空格、8,839 条非 ASCII，**只能用 Python 处理名单**。
- **数据源**：bj31 `~/motion_dataset/g1_atlas_full`，481,068 条，869G，全部缺 `body_names` 键。
- **缓存**：`/data2/supermimic/cache/`，共 572G，训练时 mmap 加页缓存。

| 目录 | clips / 帧 / bin | 大小 | 建法 | 使用者 |
|---|---|---|---|---|
| `lib_365855_b608e70f1d` | 365,855 / 305,173,599 / 6,269,660 | 209G | `build_cache.py`，64 workers，1851 s | 已废弃的全量线，可以回收（需用户同意） |
| `lib_254518_926db30c44` | 254,518 | 182G | 启动时由 rank 0 现建，约 26.5 min（另一说是 75 min，见 §0.5 #7） | ws86k 系；也是 extend 的底库 |
| `lib_255008_3b84d5f141` | 255,008 / 265,826,690 / 5,428,418 | 182G | `extend_cache.py`，166 s（另一说是 11 min，见 §0.5 #8） | r15ws 及之后的所有线 |

- **缓存键**：`sha1("\n".join(完整路径) + "|" + 关节名 + "|" + 体名 + "|fps")[:10]`。名单有任何变化就会重建。可以用 `/data2/supermimic/prep_1025/cache_key_chk.py` 或 `extend_cache.py --dry_run` 预测键名。
- **换数据集要在旧 run 还在跑时先把缓存建好**，否则 8 卡会在 barrier 上空等。
- **校验工具**：`/data2/supermimic/tools/`，包括 fk_ref_check、verify_cache、list_check、xcheck_bj27、vel_outliers。

### 6.4 启动与停止

- **启动**：`[CUDA_VISIBLE_DEVICES=… PORT=… NPROC=… THREADS=24 DRY_RUN=1] bash scripts/launch_bj31_ddp.sh <run> <train.py 参数>`。
- **同机跑两路 4 卡**：
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29521 NPROC=4 THREADS=24 bash scripts/launch_bj31_ddp.sh runA ...
  CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29522 NPROC=4 THREADS=24 bash scripts/launch_bj31_ddp.sh runB ...
  ```
  - 两路错开 ≥ 300 s（实际用 330 s），让第一路先完成 env 构建、缓存映射和编译。
  - 端口不能重复。用过的有 29511/29512 和 29521/29522。
  - 每路全局 batch 减半，LR 不重算。
- **远端启动脚本一律后台运行**：`setsid nohup bash <script> > runs/<x>_start.log 2>&1 < /dev/null & disown`。原因是脚本里的前台 `sleep 330` 遇到 ssh 断开会收到 SIGHUP，第二路就起不来。
- **启动后要核对的日志行**：
  - `[launch] … commit <hash>`、`[train] num_threads=24`；
  - `[MotionLibrary] loaded cache …`，确认没有重建缓存；
  - `data parallel: N ranks x 4096 envs / 4096 rollouts`；
  - `[dist] start: learned state identical on all N ranks`；
  - `resumed from … (policy lr … = lr x lr_scale … x cosine …)`；
  - `clamp_returns: 0 of 5428418 loaded bin returns clamped`；
  - `watchdog signal: pi/imagined_reward_alive`；
  - `early stop: eval failed >= 1.5 x median…`；
  - `root-orientation termination: on …`；
  - 如果出现 `WARNING resume from another setup: …`，要逐条确认是预期的。
- **停止**：`{ bash scripts/stop_bj31_ddp.sh <run> || true; }`，然后轮询 `nvidia-smi --query-compute-apps=pid` 直到为空，最多等 60 s。
  - 停止脚本 kill 之后只 sleep 1 s，`/proc/<pid>` 还在就打印 still alive 并 exit 1。CUDA 退出较慢时，即使已经杀成功也会这样。10-25 第一次 T/C 启动就因为 `set -e` 在这里中断。
- **热启动 ckpt 的规范**：
  - 复制成 `/data2/supermimic/warmstart_<来源>_model_<it>.pt`，先写 `.part` 再 `mv`，复制前后都核对 md5。原因：活着的 run 回滚后可能重写同名的 `model_<it>.pt`。
  - 现存 5 个：`g1v2_scratch_86000`（e398a943…）、`g1v2_clamp_r15_136000`（aa465024…）、`r15ws_160000`（c0818a37…）、`oriT_168000`（311b86bd…）、`ori8g_174000`（2bd84993…）。
  - bj27 和 bj31 之间没有直连，经本机 `scp -3` 中转，约 290 MB 要 1.5–2 min。

### 6.5 日志与 ckpt 的位置

- **控制台日志**：`/data2/supermimic/runs/<run>.log`，旁边是 `<run>.pgid`、`<run>_ranks/<id>/attempt_0/<rank>/{stdout,stderr}.log` 和 `*_start.log`。
- **run 目录**：`/data2/supermimic/logs/<YYYY-MM-DD_HH-MM-SS>_<run>/`，内容包括：
  - `model_<it>.pt`，每 2000 轮一个；
  - `model_best_eval.pt` 和 `best_eval.json`；
  - `model_final.pt`，只在正常结束或早停时才有；
  - `eval.jsonl`、`train_cfg.json`、`motion_files.txt`、`eval_clip_names{,_missing}.txt`、`watchdog.log`（有回滚时才有）、TB events。
  - `logs/last_run.txt` 以追加方式记录。
- **ckpt 大小**：H48 v2 线是 291,457,448 B。
- **磁盘占用**（10-26）：logs 51G，runs 25G（几乎全是 rank 日志），eval_export 15G。

### 6.6 自动导出与 v4 流水线

- **启动方式**：`RUN=<run> NAME="<前缀>" STEP=2000 FROM=<首个 it> GRACE=1800 REPO=… XS=… setsid nohup bash auto_v4.sh > auto_v4_<run>.log 2>&1 &`，在 `/data2/supermimic/eval_export/scripts/` 下运行。
  - 已提交的 it 记在 `auto_v4_<run>.done`，要重交某个点就删掉对应行。
  - 失败后每 10 min 重试一次，没有上限。
  - 训练进程消失、并且没有待交的 ckpt 时，自行退出。
- **`export.sh`** 只用 CPU：
  - 先等 ckpt 文件"年龄 ≥ 90 s 且大小稳定"，因为 `torch.save` 是原地写；
  - 各类缓存都隔离在 `eval_export/` 下；
  - 流程是 `export_bundle.py` → `export_eval_onnx.py --metric_bodies <14 体> --damping actuator` → `check_onnx_vs_torch.py`（阈值 1e-4，实测约 3e-6）→ 打印 sha256；
  - 单次 9–11 s。日志里的 `Warp CUDA error 100` 属预期，无害。
- **`submit.sh`**：
  - 上传 `POST /api/v1/policies/upload`，`observation_spec_id=supermimic_g1_v1` 是**写死的**；
  - 建评测 `POST /api/v1/runs`，参数为 `g1_motionatlas_v4` / `kimodo_g1_motion_official_v3` / official / workers 64；
  - 评测站队列是串行的，每个 v4 跑 9–18 min；
  - 命名约定：`SuperMimic G1 <线说明> model_<it> (bj31 <run> <MM-DD>)`，`analyze.py` 按名字里的 "SuperMimic G1" 筛选。
- **修过的两个 bug**：
  - 非数字的 ckpt 名（`model_best_eval.pt`）在 `set -u` 下导致崩溃；
  - auto_v4 比训练先启动时会立即退出，C 臂的 162000 因此漏交，现在加了 `GRACE=1800`。
- **评测站 profile**：
  - `supermimic_g1_v1`：`obs_hist[1,10,93]` + `goal[1,10,64]`，输出 `action[1,29]`。
  - `supermimic_g1_v2`：ref_root 布局，goal 10×74。
  - `e2_supermimic_v2`。
- **评测站注意事项**：
  - `sudo -n systemctl restart noetix-eval` 会杀掉正在跑的评测，重启前先确认站点空闲。
  - **`/data1/eval` 不要碰**。
  - 评测站规则：上榜的策略只能用本体感知，不能用里程计。

### 6.7 CPU 争用

评测站的 v4 批次和训练在同一台机器上，会抢 CPU。观测到的情况：

| 负载 | 训练速度 |
|---|---|
| load 70 | 625 → 约 990 ms/it |
| load 112–121 | 1.4–2.1 s/it |
| load 226 | T2 从 0.8 s/it 变成 3.8 s/it |

- 只影响速度，不影响结果。
- 可调的是评测 workers 数。10-09 的对照：并行 128 改成 32，训练变慢的幅度从两倍多降到约 13%，代价是每个 G1 评测从 12 min 变成 18 min。
- 守护脚本在有 `evaluate_mujoco` 进程时不报"变慢"。

### 6.8 警告：哪些无害，哪些存疑

- **`solver iterations limit reached`**：rank 日志里刷屏，每个 rank 约 1.3 GB，8 个 rank 合计约 10 GB/天。原因是 `warn_overflow=0` 是 `wp.static` 编译期常量，运行时设置无效。master 60f6b6a 改成设置后重新 capture graph，**还没在 bj31 上验证**。在此之前要定期清理 `runs/*_ranks`。
- **`nefc overflow (njmax 250)`**：会进控制台日志（8g 主线 985 行），每个 rank 399–516 次。与想象奖励尖峰在时间上不重合，影响未证实。
- **导出时的 `Warp CUDA error 100`、`torch.jit.script` FutureWarning、`motions.py:307` 非可写 numpy 的 UserWarning**：都无害。

### 6.9 坑表（全部实际踩过）

| # | 坑 | 表现 | 修法 / 规矩 |
|---|---|---|---|
| 1 | 用了 gmt_env.sh，或 LD_PRELOAD 了 cu12 版 NCCL | 加载错的 libnccl | 启动器已 unset；看 banner |
| 2 | npz 缺 `body_names` | 按名字重排时 KeyError | c59c8ce 回退到固定 37 体 |
| 3 | 远端命令不加 timeout | 子任务卡死两次，bj31 空转约 15 h | 一律 `timeout 90 ssh -n -o ConnectTimeout=15`；长任务 setsid 后台运行并轮询 |
| 4 | 第一次冒烟出 NaN | 写进所有卡的权重，回滚 6 次后放弃 | 01c4d58；长跑之前先冒烟 |
| 5 | 只杀 pgid | 留下孤儿 rank 继续占卡 | bc9d507 加 PDEATHSIG，并且只用 stop 脚本停 |
| 6 | stop 脚本 exit 1 | `set -e` 的启动脚本中断 | `{ stop || true; }` 然后轮询 nvidia-smi |
| 7 | 同机两个任务用同一个端口 | 静默串联，停一个连带杀另一个 | `is_host=1`，端口各不相同 |
| 8 | 启动脚本里有前台 sleep，ssh 一断 | 第二臂起不来 | `setsid nohup` 后台运行 |
| 9 | 启动时才建缓存 | 8 卡空等 | 提前建好，或用 extend |
| 10 | `--motion_list` 部分文件缺失 | rank 0 崩溃，其余 rank 卡在 barrier | df670e6 改成广播前直接报错 |
| 11 | 单个 rank 出现有限但极大的负奖励 | 误回滚，LR 连续减半 | floor 加 alive 信号 |
| 12 | 名单里有空格和中文 | shell 分词出错 | 只用 Python 处理 |
| 13 | 改了正在运行的脚本或目录 | 进程读到半新半旧的内容 | 一个 commit 一个目录 |
| 14 | export.sh 默认用旧代码包 | 报 robot_variant 相关错误 | 显式传 `REPO=… XS=…` |
| 15 | ckpt 正在写时就导出 | 读到半截文件 | 等文件 ≥ 90 s 且大小稳定 |
| 16 | `model_best_eval.pt` 遇上 `set -u` | auto_v4 崩溃 | 跳过非数字名 |
| 17 | auto_v4 比训练先启动 | 立即退出，漏交 | `GRACE=1800` |
| 18 | submit.sh 写死 v1 profile | ref_root ckpt 不兼容，无限重试 | 按 `goal_layout` 选 profile |
| 19 | 评测批次抢 CPU | 训练慢 1.5–4.75 倍 | 属暂态；必要时降评测 workers |
| 20 | rank 日志膨胀 | 约 10 GB/天 | 验证 60f6b6a 的修复；在此之前定期清理 |
| 21 | `pgrep/pkill -f` 匹配到自己 | 杀掉自己的会话，或误判"已在跑" | 模式写成 `[s]cripts/train.py .*--run_name X( \|$)`；kill 和启动不要写在同一条 ssh 命令里 |
| 22 | 早停在失败数很小时过敏 | 单次 BAD 共出现 5 次 | 判断仍以 v4 为准；修判据 |
| 23 | 跨配置续训 | 采样器错位，或被静默跳过 | 起训后逐条检查 WARNING |
| 24 | 开了骨盆终止的 ckpt 带 `root_ori_*` 键 | 旧代码读不了 | 用 f1f364b 之后的代码，或合并后的 master |
| 25 | 本机 shell 是 zsh | 不加引号的 `$VAR` 不分词 | 用数组，或直接写字面量 |

### 6.10 守护脚本

- 用法：`$S/guard_arm.sh <run> <rank数> <正常ms/it>`，在本机经 ssh 每 180 s 轮询一次。
- 检查内容：
  - rank 进程数是否少于预期；
  - 日志超过 600 s 没更新；
  - 日志末尾出现 Traceback；
  - 新出现的 `[guard]`（不含 floored 行）和 `[watchdog]` 行；
  - 变慢（超过 1.5 倍）；
  - 有没有别人的 GPU 进程。
- ssh 连续 3 圈失败才报警。
- **缺口**：用 kill -9 停掉的 run 不会写结束标志，守护会每 3 min 报一次 ranks=0。所以停 run 时要手动停对应的守护。

---

## 7. 下一步建议（按证据强弱排序）

**先做的前置工作**

- **P0：在 bj31 上跑通合并后的代码**，按 §1.6 的冒烟清单。然后从 174000 起，**用合并后的代码跑一个 4 卡 C 臂当新基线**。因为数值轨迹已不能和 42b211e 逐位对上（§0.4 第 4 条），之后所有 T/C 都应该以这个基线为参照。
- **P0：修 §1.2 的 #1（`--lr_scale` 防护）和 #2（`_resume_diffs` 补比对项）**，再开始任何续训。

**实验候选**（全部用 4+4 同起点 T/C，用后段三点均值判定；起点建议 174000）

| 优先 | 实验 | 现有证据 | 证据强度 | 做法与前提 |
|---|---|---|---|---|
| 1 | **在我们 1.0 骨盆线上叠加 `--goal_ref_root`（根部运动）** | bj27 上，在骨盆终止之上 +0.38pp（4 点匹配）/ +0.51（自报 3 点）；翻腾 +2.5、GROUND_GETUP +2.75。**但移植估计是 −0.10pp**：bj27 稳健修好的 11 条，在我们线上的期望失败只有 5.0 条 | 中：bj27 的实测是正的，我们线上只有模型估计 | 新增的 goal 列零初始化嫁接（`_graft_inputs`，已合并；2 rank 嫁接已验证，C 组）。提交要用 v2 profile，所以要用 `submit_v4.sh`。部署走 SuperMimicV2 模式。策略是本体感知的，可以上榜。注意 goal 维度 64→74 会改变 ONNX 布局 |
| 2 | **地面曝光**（name_weights 加起身、躺、爬） | bj27 两点 +0.19（GROUND_GETUP +4.5）、四点约 +0.3–0.4；188k 的 +0.44 到 192k 消失。**移植估计 −0.30pp**：bj27 修好的 10 条，在我们线上的期望失败只有 1.7 条 | 弱 | 目标簇是 SIDE_RIGHT 起身（2/7 对 7/7）和手膝爬行。训练集里 GROUND_ 前缀的片段是 0 条，这是数据缺口，等动捕数据（见记忆 mocap-capture-plan-1013）。可以和第 1 项做 2×2，但先单独做 |
| 3 | **钳位之后的采样器集中问题**：`--sampler_cap`、`--sampler_boot_mean` | 机制证据清楚：G=0 的坐、蹲 bin 拿走 11.6% 的重置；r25 上 90% 的份额落在 1,588 个 bin 上。效果在本线没有数据 | 机制强，效果未知 | T/C，同时盯坐、蹲 bin 的份额和 `top1_prob`。探针脚本要同步新公式。簇先验的语义已改（§1.2 #4），先问清楚再用 |
| 4 | **确定性闭环（退化保险）** | 干预实验强；没有"不加噪"的对照训练 | 中 | 先做 CPU 判别实验（约 30 min）：在想象里算腿部动作变化率，与真实确定性执行的 0.267、加噪的 0.114 对比，区分"WM 看不见抖动"和"奖励容忍抖动"。再按结论在 "25% env 不加噪 / 噪声退火" 和 "加大 action_rate、加 jerk 惩罚" 之间做 A/B。H48 线目前抖动不涨，所以优先级是中 |
| 5 | **跳跃/腾空**（对榜首差 6.4pp） | root_pos 分析：40/42 是跳不到参考高度 | 现象清楚，手段未知 | 候选：腾空段的 WM 损失加权；只对腾空片段的难集课程。都没有对照数据 |
| 6 | 8 卡对 4 卡、单卡对 8 卡的样本效率 | "1.6 倍/轮"是经验估计；C2 的 4 卡在 182–186k 比 8 卡高 0.29–0.42 | 弱 | 用同配方同起点做多点对照。会影响以后要不要继续用 8 卡 |
| 7 | 稠密骨盆朝向奖励 | 现在的 `root_ori` 奖励挂在 torso 锚点上（master heads.py L253） | 无 | 如果终止项已经饱和，可以试 |
| 低 | `--root_z_term` 骨盆高度终止 | §5.5 已证伪它针对的是盲区 | 反证 | 不建议做 |
| 低 | 收紧 root_ori 到 0.8 | 两条线都不支持 | 反证 | 如果还想试，先看 `wm/root_yaw_err_*`，再考虑"高动态放宽集"（按名字保留 1.0）。这个方案写过但没跑 |
| 小项 | 早停过敏、RSI 偏置、环境侧子步滞后、`wd_reward_floor` 取值校验 | §1.2 #9 | — | 修代码即可 |

**和 bj27 线怎么叠加**：bj27 的最好线（floor10_r25 208000，91.70）同时带骨盆终止 1.0、ref_root 和地面曝光，我们的 174000（91.83）只带骨盆终止。两条线并集的上限是 94.67%（我们 13 点与 bj27 的组合），说明两边会的段不完全重合。但移植估计表明，bj27 那两项修好的段大多我们本来就能过。所以**不要直接把 bj27 的配方整套搬过来，要逐项做 T/C**，顺序是先 ref_root（证据最强、能修翻腾和起身），再地面曝光（等数据）。如果两项都是正的，再做 2×2 检查能否叠加。

---

## 8. 文件与路径索引

### 8.1 本机

| 路径 | 内容 |
|---|---|
| `~/Workspace/supermimic` | 整合中的仓库。我只读，master = b0ddc5b。分支 `ddp-bj31`（5bb8c0d）、`ddp-bj31-v2`（42b211e） |
| `$S/repo`、`$S/repo_v2` | 两个分支的 worktree |
| `$H/mt`、`$H/m60f` | 合并后 master 和 60f6b6a 的 scratch 克隆 |
| `$H/{testlogs,bit,ddp}/` | 今天的复跑产物：单测日志、逐位对照、2 rank 端到端 |
| `$S/ddp_design.md` | DDP 设计文档，包括逐位对照记录和故障注入 |
| `$S/env_bj31.md`、`$S/cache_bj31.md`、`$S/bj31.md`、`$S/bj27.md` | 环境和缓存笔记 |
| `$S/merge_trial_1026` | 试合并的记录（975decb 那一版） |
| `$S/evalsite/remote/` | 导出、上传、分析脚本的本机副本 |
| `$SP/diag_1026_next/` | 差距分析 |
| `/home/user/Workspace/Noetix_GMT/deliverables/policies_1022/supermimic_g1_full_ee_ddp8_20000/` | 8 卡从零线 20k 的交付包 |
| `…/deliverables/policies_1025/supermimic_g1_r15ws_{148000,158000,160000}/` | r15ws 的交付包 |
| `…/deliverables/policies_1026/supermimic_g1_oriterm_{T_168000,8g_174000}/` | 骨盆终止线的交付包 |
| `$MEM/supermimic-bj31-1022.md`（M-bj31） | 本线主笔记 |
| `$MEM/supermimic-ddp-degradation-rootcause-1024.md` | 退化根因 |
| `$MEM/supermimic-fall-gap-pelvis-ori-1025.md` | 骨盆盲区与 T/C |
| `$MEM/supermimic-v2-iteration-1022.md` | bj27 各轮实验 |
| `$MEM/supermimic-eval-site-1022.md`、`supermimic-refroot-site-deploy-1026.md`、`supermimic-repro-1021.md`、`supermimic-oss-review-1025.md`、`feedback-eval-site-proprio-only.md`、`gmt-training-gotchas.md` | 其它相关笔记 |
| `$WF/wf_<id>/journal.jsonl` | 工作流结果：df97fa16（DDP 实现）、0bfb9f8a（EE 与冒烟）、0b711611（退化诊断）、489d1978（钳位）、ae810087（floor）、6f8fe82a（r15ws 合并与早停）、5c6906ac（摔倒差距）、74aad1e4（骨盆终止）、e6613df9（合并 18f528f 时的侦察）、aff2ea08（bj31 就绪情况）、14bf2c8c（导出与评测） |

### 8.2 bj31（`tencent_bj31`，只读）

| 路径 | 内容 |
|---|---|
| `~/supermimic/mjlab/.venv` | 训练用 venv |
| `~/supermimic/supermimic_repro{,_clamp,_v2,_v2wd,_r15ws,_oriterm_test,_oriterm}/` | 各 commit 的代码目录：bc9d507 / 5bb8c0d / df670e6 / ca9266e / 55e0eaf / f1f364b / 42b211e，每个目录有 `COMMIT` 文件 |
| `~/motion_dataset/g1_atlas_full` | 数据源，481,068 条 |
| `/data2/supermimic/cache/lib_*` | 三份缓存 |
| `/data2/supermimic/lists/`、`prep_1025/` | 名单（带 MD5SUMS）、`cache_key_chk.py`、`prep.log` |
| `/data2/supermimic/logs/<ts>_<run>/` | ckpt、`train_cfg.json`、`eval.jsonl`、`best_eval.json` |
| `/data2/supermimic/runs/` | 控制台日志、rank 日志、`*_start.log`、`*.pgid` |
| `/data2/supermimic/warmstart_*.pt` | 5 个热启动 ckpt |
| `/data2/supermimic/eval_export/` | `scripts/`、`auto_v4_<run>.{log,done}`、bundles、ONNX、`run_<id8>.json` |
| `/data2/supermimic/diag_1025/oriterm/` | D1/D3 输出、`start_*.sh` |
| `/data2/supermimic/diag_1026_rootpos/`、`diag_1026_next/` | root_pos 复跑、差距分析 |
| `/data2/supermimic/tools/` | 缓存和名单的校验工具 |
| 评测站 `http://127.0.0.1:8010/api/v1`（只用 GET） | leaderboard、runs |
| `/home/ubuntu/work/NOetix_GMT_eval/storage/runs/<id>/evaluation_per_segment.csv` | 逐段结果 |
| `~/work/supermimic_eval_onnx/evaluator_r18/eval_r18.sh` | 私有离线 v4，与站点逐位一致（核过 G1 208000、E2 156000） |
| `/data1/eval` | 评测站存储，**不要碰** |

### 8.3 bj27（`tencent_bj27`，只读，另一会话的线）

| 路径 | 内容 |
|---|---|
| `~/supermimic/round*_launch.sh`、`*_auto.sh` | 用旧语义 `--lr_scale` 的 16 个脚本（round19c…round27） |
| `~/supermimic/round28_launch.sh` | 已改用 `--resume_lr_scale`；显式写了 `--ee_term_z 0.25`、`--fixes` 和看门狗 opt-in；正在跑 master 的 `--root_z_term` |

---

*本文所有"已核"的结论截至 2026-10-26 17:1x。master 之后若再前进，请以 `git log master` 为准，重新核对 §1.2 的各项。*
