#!/usr/bin/env bash

DATA_DIR="../../../data/diving48/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}

wget http://www.svcl.ucsd.edu/projects/resound/Diving48_rgb.tar.gz --no-check-certificate
tar -zxvf Diving48_rgb.tar.gz
mv ./rgb ./videos

cd -
