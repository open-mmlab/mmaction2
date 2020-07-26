#!/usr/bin/env bash

DATA_DIR="../../../data/ucf101/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}

wget https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar --no-check-certificate
unrar x UCF101.rar
mv ./UCF-101 ./videos

cd "../../tools/data/ucf101"
