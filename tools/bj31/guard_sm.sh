#!/bin/bash
# SuperMimic bj31 8 卡训练守护(run g1v2_ddp8_ws86k_b,10-23 19:3x 从 ws86k 104000 续,看门狗奖励下限 -1):只在异常时输出。每 3 分钟查:rank 进程数 / 日志更新时间 / Traceback /
# [guard] 跳步与 [watchdog] 回滚 / 每轮耗时(>1.5×625 ms)/ 别人的 GPU 进程(评测站 noetix_eval 除外)/ 训练结束。
# bj31 公网 ssh 偶发 kex 断开:连续 3 圈连不上才报。
RUN=g1v2_r15ws_ddp8; BASE=900; fails=0
# 已报过的 [guard]/[watchdog] 行数存盘,守护重挂时不重复报;[guard] 只数非有限跳步行(奖励下限 floored 行是预期内的,不报)
ST=$(dirname "$0")/guard_sm_state_$RUN; read -r seen_g seen_w < "$ST" 2>/dev/null; seen_g=${seen_g:-0}; seen_w=${seen_w:-0}
while true; do
  s=$(timeout 60 ssh -o ConnectTimeout=15 tencent_bj31 "L=/data2/supermimic/runs/$RUN.log; n=\$(pgrep -fc '[s]cripts/train.py .*--run_name $RUN'); age=\$(( \$(date +%s) - \$(stat -c %Y \$L) )); \
e=\$(tail -c 300000 \$L | grep -a -E 'Traceback|Exception|Error' | grep -v -i warn | tail -1 | cut -c1-120); \
g=\$(grep -a '\[guard\]' \$L | grep -v -c 'floored'); w=\$(grep -a -c '\[watchdog\]' \$L); \
ms=\$(grep -a -E '^\[default0\]:\[it ' \$L | tail -5 | grep -o '[0-9]* ms/it' | awk '{print \$1}' | sort -n | sed -n 3p); \
it=\$(grep -a -E '^\[default0\]:\[it ' \$L | tail -1 | awk '{print \$2}'); \
f=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | while read p; do c=\$(tr '\0' ' ' < /proc/\$p/cmdline 2>/dev/null); case \"\$c\" in *train.py*|*noetix*|*NOetix*|*evaluate_mujoco*) ;; *) echo -n \"[\$p:\${c:0:40}]\";; esac; done); \
fin=\$(grep -a -c -E 'training finished|final: learned state|\[earlystop\]' \$L); ev=\$(pgrep -fc '[e]valuate_mujoco.py'); echo \"\$n|\$age|\$ms|\$it|\$g|\$w|\$fin|\$ev|\$f|\$e\"" 2>/dev/null)
  if [ -z "$s" ]; then fails=$((fails + 1)); [ $fails -ge 3 ] && echo "$(date +%H:%M) sm-bj31: ssh 连续 $fails 圈无响应"; sleep 180; continue; fi
  fails=0
  IFS='|' read -r n age ms it g w fin ev f e <<< "$s"
  [ "${fin:-0}" -ge 1 ] && [ "${n:-0}" -eq 0 ] && { echo "$(date +%H:%M) sm-bj31: 训练结束或提前停止(it ${it}),看 earlystop.json"; exit 0; }
  if [ "${n:-0}" -lt 8 ] || [ "${age:-0}" -gt 600 ] || [ -n "$e" ]; then echo "$(date +%H:%M) sm-bj31: ranks=$n log_age=${age}s it=$it err=[$e]"; fi
  # 变慢只在开始/结束各报一次(>1.5× 报变慢,<1.2× 才报恢复,防在阈值附近反复)(bj31 评测站跑 v4 时 CPU 争用会让每轮从 ~600 涨到 ~950 ms,是暂时的)
  # 10-23:评测站有 evaluate_mujoco 进程时(v4 在跑)不报变慢,只在没有评测时报
  if [ -n "$ms" ] && [ "${ev:-0}" -eq 0 ] && [ "$ms" -gt $((BASE * 3 / 2)) ]; then [ "${slow:-0}" = 0 ] && echo "$(date +%H:%M) sm-bj31: 每轮 ${ms} ms(正常 ~${BASE})变慢 it=$it"; slow=1
  elif [ -n "$ms" ] && [ "${slow:-0}" = 1 ] && [ "$ms" -lt $((BASE * 6 / 5)) ]; then echo "$(date +%H:%M) sm-bj31: 速度恢复 ${ms} ms it=$it"; slow=0; fi
  [ "${g:-0}" -gt "$seen_g" ] && { echo "$(date +%H:%M) sm-bj31: 出现 [guard] 跳步行 共 $g 行 it=$it"; seen_g=$g; echo "$seen_g $seen_w" > "$ST"; }
  [ "${w:-0}" -gt "$seen_w" ] && { echo "$(date +%H:%M) sm-bj31: 出现 [watchdog] 行 共 $w 行 it=$it"; seen_w=$w; echo "$seen_g $seen_w" > "$ST"; }
  [ -n "$f" ] && echo "$(date +%H:%M) sm-bj31: 有别人的 GPU 进程 $f"
  sleep 180
done
