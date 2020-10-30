#!/usr/bin/env bash

set -e

DATA_DIR="../../../data/ava/videos"
ANNO_DIR="../../../data/ava/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt -P ${ANNO_DIR}

cat ${ANNO_DIR}/ava_file_names_trainval_v2.1.txt |
while read vid;
    do wget -c "https://s3.amazonaws.com/ava-dataset/trainval/${vid}" -P ${DATA_DIR}; done

echo "Downloading finished."
