# tools：诊断与运维脚本

这里是两条训练线（bj27 单卡配方线、bj31 8 卡数据并行线）在 10-22 ~ 10-26 做诊断时写的脚本，从各自会话的临时目录收进仓库，结论见根目录 README 的"实验结论"和 bj31 线的交接文档（`docs/bj31_8gpu_handoff_1026/README.md` §5）。

它们是当时一次性的分析工具，不是维护中的库：很多脚本写死了 bj31 的 `/data2/supermimic`、bj27 的 `~/supermimic` 或会话临时目录（`/tmp/claude-1000/...`）的路径，复用前先改路径。不影响训练代码，也不在 `tests/` 里。

| 目录 | 内容 | 运行环境 |
|---|---|---|
| `diag/compare/` | `cmp_ckpt.py`、`cmp_tb.py`：两份代码跑同样命令后，逐位比较 final ckpt 的全部张量 / 非计时 TB 标量（合并前后"开关关闭即逐位一致"的验证就用它）；`run_ddp.sh`：2 rank gloo torchrun 端到端（A 生产开关 + 强制回滚 / B 全功能 / C 续训嫁接） | 本机 CPU，mjlab venv |
| `diag/v4/` | `v4table.py`：读评测站 API + per-segment CSV 汇总 v4；`gap_next.py` / `gap_next2.py` / `dump_union.py`：多 ckpt 的逐段通过概率、与榜首的稳健差距、成功段并集（`gap_next2.py` 后半段有 TypeError） | bj31，只读 |
| `diag/sampler/` | 重置采样器探针：`sampler_probe{,2,3}.py`（份额、top1、坐 / 蹲 G=0 的 bin 吃掉多少重置）、`sampler_sim_clamp.py`、`sampler_r52.py`；`sampler_analysis.py`：从 ckpt 的 `bin_G` 重建采样分布，看 softmax 份额的集中度和封顶效果；`cluster_build.py`：预建 bin 运动学聚类缓存并按来源比较先验 | bj31 / bj27 CPU |
| `diag/degradation_1024/` | 8 卡线"峰后持续退化"的诊断：闭环干预（`sm_roll{,2,3}.py`：确定性执行 vs 加噪）、动作雅可比（`jac{,2}.py`）、WM 外推（`wm_probe.py`）、ckpt 扫描与 TB 趋势（`ckscan.py`、`tbtrend.py` 等）、抖动指纹（`jitter_*.py`） | bj31 CPU / 只读 CSV |
| `diag/pelvis_ori_1025/` | 骨盆全角朝向终止的论证：评测站 root_ori 失败逐帧复跑（`d1_*.py`、`trace_*.py`）、参考骨盆四元数一致性（`d3_refquat.py`）、腰部解耦（`trace_waist.py`）、`sampler_probe3.py` | 评测站 python + mjlab venv |
| `diag/rootpos_1026/` | 评测站 root_pos 失败逐帧复跑（结论：不是判据盲区，40/42 是跳不到参考高度） | 评测站 python |
| `bj31/` | bj31 运维：`auto_v4.sh`（每 STEP 轮自动导出 + 上站）、`export.sh`、`submit.sh`（10-26 起按 ONNX 的 goal_layout 选 v1/v2 观测配置）、`check_onnx_vs_torch.py`、`analyze.py`、`meta_diff.py`；守护脚本 `guard_arm.sh` / `guard_sm.sh`；骨盆终止对照的启动示例 `start_*.sh` | bj31 |

评测站相关的通用工具（插件、站点补丁、离线 v4 评测器、提交脚本、失败分析）在 `eval_site/`。
