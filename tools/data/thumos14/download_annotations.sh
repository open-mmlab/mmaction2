#!/usr/bin/env bash

DATA_DIR="../../../data/thumos14/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi
cd ${DATA_DIR}

wget http://crcv.ucf.edu/THUMOS14/Validation_set/TH14_Temporal_annotations_validation.zip --no-check-certificate
wget http://crcv.ucf.edu/THUMOS14/test_set/TH14_Temporal_annotations_test.zip --no-check-certificate

if [ ! -d "./annotations_val" ]; then
  mkdir ./annotations_val
fi
unzip -j TH14_Temporal_annotations_validation.zip -d annotations_val

if [ ! -d "./annotations_test" ]; then
  mkdir ./annotations_test
fi
unzip -j TH14_Temporal_annotations_test.zip -d annotations_test

rm TH14_Temporal_annotations_validation.zip
rm TH14_Temporal_annotations_test.zip

cd "../../tools/data/thumos14/"
