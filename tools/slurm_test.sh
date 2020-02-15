#!/usr/bin/env bash

set -x
PARTITION=$1
JOB_NAME=$2
CONFIG=$3
CHECKPOINT=$4
GPUS=${5}
GPUS_PER_NODE=$GPUS
CPUS_PER_TASK=$GPUS
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:6}
#PY_ARGS=${PY_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${CPUS_PER_TASK} \
    --cpus-per-task=2 \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python3 -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher='slurm' ${PY_ARGS}
