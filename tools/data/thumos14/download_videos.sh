#! /usr/bin/bash env

DATA_DIR="../../../data/thumos14/"

cd ${DATA_DIR}

wget https://storage.googleapis.com/thumos14_files/TH14_validation_set_mp4.zip
wget https://storage.googleapis.com/thumos14_files/TH14_Test_set_mp4.zip

if [ ! -d "./videos/val" ]; then
  mkdir ./videos
  mkdir ./videos/val
fi
unzip -j TH14_validation_set_mp4.zip -d videos/val

if [ ! -d "./videos/test" ]; then
  mkdir ./videos
  mkdir ./videos/test
fi
unzip -P "THUMOS14_REGISTERED" TH14_Test_set_mp4.zip -d videos/test

cd "../../tools/data/thumos14/"
