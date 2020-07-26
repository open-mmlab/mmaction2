#!/usr/bin/env bash

DATA_DIR="../../../data/kinetics400/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz

tar -zxvf kinetics400.tar.gz --strip-components 1 -C ${DATA_DIR}/
mv ${DATA_DIR}/train.csv ${DATA_DIR}/kinetics_train.csv
mv ${DATA_DIR}/validate.csv ${DATA_DIR}/kinetics_val.csv
mv ${DATA_DIR}/test.csv ${DATA_DIR}/kinetics_test.csv

rm kinetics400.tar.gz
rm ${DATA_DIR}/*.json
