#!/usr/bin/env bash

set -e

DATA_DIR="../../../data/hmdb51/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}

mkdir -p ./videos
cd ./videos

wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar --no-check-certificate

# sudo apt-get install unrar
unrar x ./hmdb51_org.rar
rm ./hmdb51_org.rar

# extract all rar files with full path
for file in *.rar; do unrar x $file; done

rm ./*.rar
cd "../../../tools/data/hmdb51"
