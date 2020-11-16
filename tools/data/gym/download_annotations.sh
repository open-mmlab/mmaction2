#!/usr/bin/env bash

set -e

DATA_DIR="../../../data/gym/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget https://sdolivia.github.io/FineGym/resources/dataset/finegym_annotation_info_v1.0.json -O $DATA_DIR/annotation.json
wget https://sdolivia.github.io/FineGym/resources/dataset/gym99_train_element_v1.0.txt -O $DATA_DIR/gym99_train_org.txt
wget https://sdolivia.github.io/FineGym/resources/dataset/gym99_val_element.txt -O $DATA_DIR/gym99_val_org.txt
