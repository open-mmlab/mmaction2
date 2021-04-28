#!/usr/bin/env bash

DATA_DIR="../../../data/diving48/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}

wget http://www.svcl.ucsd.edu/projects/resound/Diving48_vocab.json
wget http://www.svcl.ucsd.edu/projects/resound/Diving48_V2_train.json
wget http://www.svcl.ucsd.edu/projects/resound/Diving48_V2_test.json

cd -
