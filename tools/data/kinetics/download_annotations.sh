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

wget https://storage.googleapis.com/deepmind-media/Datasets/${DATASET}.tar.gz

tar -zxvf ${DATASET}.tar.gz --strip-components 1 -C ${DATA_DIR}/
mv ${DATA_DIR}/train.csv ${DATA_DIR}/kinetics_train.csv
mv ${DATA_DIR}/validate.csv ${DATA_DIR}/kinetics_val.csv
mv ${DATA_DIR}/test.csv ${DATA_DIR}/kinetics_test.csv

rm ${DATASET}.tar.gz
rm ${DATA_DIR}/*.json
