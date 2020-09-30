#!/usr/bin/env bash

set -e

DATA_DIR="../../../data/ava/"

wget https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/filelist/ava_dense_proposals_train.FAIR.recall_93.9.pkl -P ${DATA_DIR}
wget https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/filelist/ava_dense_proposals_val.FAIR.recall_93.9.pkl -P ${DATA_DIR}
wget https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/filelist/ava_dense_proposals_test.FAIR.recall_93.9.pkl -P ${DATA_DIR}
