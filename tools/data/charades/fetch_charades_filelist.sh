#!/usr/bin/env bash

set -e

DATA_DIR="../../../data/charades/annotations"

wget https://download.openmmlab.com/mmaction/dataset/charades/annotations/charades_train_list_rawframes.csv -P ${DATA_DIR}
wget https://download.openmmlab.com/mmaction/dataset/charades/annotations/charades_val_list_rawframes.csv -P ${DATA_DIR}
