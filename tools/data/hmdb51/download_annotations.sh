#!/usr/bin/env bash

set -e

DATA_DIR="../../../data/hmdb51/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar --no-check-certificate

# sudo apt-get install unrar
unrar x test_train_splits.rar
rm test_train_splits.rar

mv  testTrainMulti_7030_splits/*.txt ./
rmdir testTrainMulti_7030_splits

cd -
