#!/usr/bin/env bash

DATA_DIR="../../../data/ucf101/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate

unzip -j UCF101TrainTestSplits-RecognitionTask.zip -d ${DATA_DIR}/
rm UCF101TrainTestSplits-RecognitionTask.zip
