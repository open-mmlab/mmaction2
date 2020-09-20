#!/usr/bin/env bash

DATA_DIR="../../../data/hvu/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

git clone https://github.com/holistic-video-understanding/HVU-Dataset.git

cd HVU-Dataset
unzip -o HVU_Train_V1.0.zip
unzip -o HVU_Val_V1.0.zip
mv HVU_Train_V1.0.csv ${DATA_DIR}/hvu_train.csv
mv HVU_Val_V1.0.csv ${DATA_DIR}/hvu_val.csv
mv HVU_Tags_Categories_V1.0.csv ${DATA_DIR}/hvu_categories.csv

cd ..
rm -rf HVU-Dataset
