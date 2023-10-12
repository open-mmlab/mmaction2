#!/usr/bin/env bash

DATA_DIR="../../../data/CharadesSTA/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}

URL="https://raw.githubusercontent.com/Alvin-Zeng/DRN/master/data/dataset/Charades"
wget ${URL}/Charades_frames_info.json
wget ${URL}/Charades_duration.json
wget ${URL}/Charades_fps_dict.json
wget ${URL}/Charades_sta_test.txt
wget ${URL}/Charades_sta_train.txt
wget ${URL}/Charades_word2id.json
