#!/usr/bin/env bash

DATASET=$1
if [ "$DATASET" == "kinetics400" ] || [ "$1" == "kinetics600" ] || [ "$1" == "kinetics700" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support kinetics400, kinetics600 or kinetics700"
        exit 0
fi

DATA_DIR="../../../data/${DATASET}/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi


wget https://download.openmmlab.com/mmaction/dataset/${DATASET}/annotations/kinetics_train.csv
wget https://download.openmmlab.com/mmaction/dataset/${DATASET}/annotations/kinetics_val.csv
wget https://download.openmmlab.com/mmaction/dataset/${DATASET}/annotations/kinetics_test.csv

mv kinetics_train.csv ${DATA_DIR}/kinetics_train.csv
mv kinetics_val.csv ${DATA_DIR}/kinetics_val.csv
mv kinetics_test.csv ${DATA_DIR}/kinetics_test.csv
