#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=2

RESUME=$1
ARCH=mbv2_ca

DATASET=${2-pitts}
SCALE=${3-250k}


$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS  --use_env \
examples/test.py --launcher pytorch \
    -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
    --test-batch-size 16 -j 2 \
    --vlad --reduction \
    --resume ${RESUME}
    # --sync-gather
