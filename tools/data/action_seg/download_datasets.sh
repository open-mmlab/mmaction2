#!/usr/bin/env bash

set -e

DATA_DIR="../../../data"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}
wget https://zenodo.org/record/3625992/files/data.zip --no-check-certificate

# sudo apt-get install unzip
unzip data.zip
rm data.zip

mv  data action_seg

cd -
