#!/usr/bin/env bash

set -x
PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=${4}
RESUME_FROM=$5
GPUS_PER_NODE=$GPUS
CPUS_PER_TASK=$GPUS
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-""}

srun -p ${PARTITION} \
     --job-name=${JOB_NAME} \
     --gres=gpu:${GPUS_PER_NODE} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --cpus-per-task=4 \
     --kill-on-bad-exit=1 \
     ${SRUN_ARGS} \
     python3 -u tools/train.py ${CONFIG} --resume_from=${RESUME_FROM} --launcher="slurm" ${PY_ARGS} \
