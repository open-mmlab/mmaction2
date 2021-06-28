#!/usr/bin/env bash

DATASET=$1
if [ "$DATASET" == "gym" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support gym now."
        exit 0
fi

DATA_DIR="../../../data/posec3d/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget https://download.openmmlab.com/mmaction/posec3d/${DATASET}_train.pkl
wget https://download.openmmlab.com/mmaction/posec3d/${DATASET}_val.pkl

mv ${DATASET}_train.pkl ${DATA_DIR}
mv ${DATASET}_val.pkl ${DATA_DIR}
