#!/usr/bin/env bash

DATA_DIR="../../../data/mit/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}

wget http://data.csail.mit.edu/soundnet/actions3/split1/Moments_in_Time_Raw.zip
unzip Moments_in_Time_Raw.zip
rm Moments_in_Time_Raw.zip

if [ ! -d "./videos" ]; then
  mkdir ./videos
fi
mv ./training ./videos && mv ./validation ./video

if [ ! -d "./annotations" ]; then
  mkdir ./annotations
fi

mv *.txt annotations && mv *.csv annotations

cd "../../tools/data/mit"
