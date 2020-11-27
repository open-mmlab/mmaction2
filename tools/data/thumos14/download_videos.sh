#!/usr/bin/env bash

DATA_DIR="../../../data/thumos14/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}

wget https://storage.googleapis.com/thumos14_files/TH14_validation_set_mp4.zip
wget https://storage.googleapis.com/thumos14_files/TH14_Test_set_mp4.zip

if [ ! -d "./videos/val" ]; then
  mkdir -p ./videos/val
fi
unzip -j TH14_validation_set_mp4.zip -d videos/val

if [ ! -d "./videos/test" ]; then
  mkdir -p ./videos/test
fi
unzip -P "THUMOS14_REGISTERED" -j TH14_Test_set_mp4.zip -d videos/test

cd "../../tools/data/thumos14/"
