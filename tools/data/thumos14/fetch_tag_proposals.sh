#!/usr/bin/env bash

PROP_DIR="../../../data/thumos14/proposals"

if [[ ! -d "${PROP_DIR}" ]]; then
  echo "${PROP_DIR} does not exist. Creating";
  mkdir -p ${PROP_DIR}
fi

wget https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/filelist/thumos14_tag_val_normalized_proposal_list.txt -P ${PROP_DIR}
wget https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/filelist/thumos14_tag_test_normalized_proposal_list.txt -P ${PROP_DIR}
