#!/bin/bash
# usage: run_trace.sh <list> <tag> <noise> <nproc>
cd /data2/supermimic/diag_1025/taskB
LIST=$1; TAG=$2; NOISE=$3; NP=$4
ONNX=/home/ubuntu/work/NOetix_GMT_eval/storage/models/f801f1e6873e78224281d1382363684af7f33c314d14e4e242c630aa73d2f1ec/model.onnx
mkdir -p out_$TAG
split -n r/$NP -d $LIST out_$TAG/chunk_
for c in out_$TAG/chunk_*; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NOISE=$NOISE CONT=6 timeout 3600 nice -n 19 \
    /home/ubuntu/miniconda3/envs/noetix_eval/bin/python -I trace_ori.py $ONNX $c.jsonl $(cat $c) > $c.log 2>&1 &
done
wait
echo done > out_$TAG/DONE
