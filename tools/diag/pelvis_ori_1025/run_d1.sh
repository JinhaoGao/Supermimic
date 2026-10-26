#!/usr/bin/env bash
# D1 / D3 on bj31, CPU only (CUDA_VISIBLE_DEVICES empty), nice 19; every write under /data2/supermimic/diag_1025/oriterm.
#   bash run_d1.sh prep | d1 <proto> <nproc> | d3 | tests
set -u
D=/data2/supermimic/diag_1025/oriterm
S=$D/scripts_d
export CUDA_VISIBLE_DEVICES= TMPDIR=$D/tmp XDG_CACHE_HOME=$D/cache WARP_CACHE_PATH=$D/cache/warp TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor \
  TRITON_CACHE_DIR=$D/cache/triton OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=$D/cache/mpl \
  REPO=/home/ubuntu/supermimic/supermimic_repro_oriterm_test D1_CACHE=$D/cache_d1
unset PYTHONPATH LD_PRELOAD http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
PY=/home/ubuntu/supermimic/mjlab/.venv/bin/python
cd $D
case "$1" in
  prep)
    nice -n 19 timeout 3000 $PY $S/prep_d1.py $D/d1_list.tsv $D/d1_prep.json > $D/d1_prep.log 2>&1; echo "prep rc=$?";;
  d1)
    P=$2; N=$3; O=$D/d1_$P; mkdir -p $O; rm -f $O/chunk_*.jsonl $O/DONE
    for i in $(seq 0 $((N - 1))); do
      PROTO=$P nice -n 19 timeout 5400 $PY $S/d1_replay.py $D/d1_list.tsv $O/chunk_$i.jsonl $i $N > $O/chunk_$i.log 2>&1 &
    done
    wait; cat $O/chunk_*.jsonl > $O/d1_$P.jsonl; echo done > $O/DONE; echo "d1 $P: $(wc -l < $O/d1_$P.jsonl) records";;
  d1e)
    N=$2; O=$D/d1_evalmodel; mkdir -p $O/dump; rm -f $O/chunk_*.jsonl $O/dump/chunk_*.jsonl $O/DONE
    EPY=/home/ubuntu/miniconda3/envs/noetix_eval/bin/python
    for i in $(seq 0 $((N - 1))); do
      nice -n 19 timeout 5400 $EPY -I $S/trace_states.py $D/d1_list.tsv $O/dump $i $N > $O/dump/chunk_$i.log 2>&1 &
    done
    wait; cat $O/dump/chunk_*.jsonl > $O/dump/trace_states.jsonl; echo "dump: $(wc -l < $O/dump/trace_states.jsonl) records"
    for i in $(seq 0 $((N - 1))); do
      nice -n 19 timeout 5400 $PY $S/d1_evalmodel.py $D/d1_list.tsv $O/dump $O/chunk_$i.jsonl $i $N > $O/chunk_$i.log 2>&1 &
    done
    wait; cat $O/chunk_*.jsonl > $O/d1_evalmodel.jsonl; echo done > $O/DONE; echo "d1e: $(wc -l < $O/d1_evalmodel.jsonl) records";;
  d3)
    nice -n 19 timeout 3000 $PY $S/d3_refquat.py $D/d1_list.tsv $D/d3 > $D/d3.log 2>&1; echo "d3 rc=$?";;
  tests)
    T=$D/tests_rerun_$(date +%m%d_%H%M); mkdir -p $T; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    cd $REPO
    echo "[tests] $(date '+%F %T') $(cat COMMIT) mjlab: $($PY -c 'import mjlab, os; print(os.path.dirname(mjlab.__file__))' 2>&1 | tail -1)" | tee $T/summary.txt
    for t in tests/test_*.py; do
      s=$(date +%s); nice -n 19 timeout 1500 $PY $t > $T/log_$(basename $t .py).txt 2>&1; rc=$?
      echo "$t rc=$rc $(( $(date +%s) - s ))s ok=$(grep -c '^ok ' $T/log_$(basename $t .py).txt) skipped=$(grep -c 'skipped:' $T/log_$(basename $t .py).txt) stubbed=$(grep -c 'stubbed via' $T/log_$(basename $t .py).txt)" | tee -a $T/summary.txt
    done
    echo "[tests] done $(date '+%F %T') -> $T" | tee -a $T/summary.txt;;
esac
