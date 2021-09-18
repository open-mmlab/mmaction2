#!/usr/bin/env bash

PROP_DIR="../../../data/thumos14/proposals"

if [[ ! -d "${PROP_DIR}" ]]; then
  echo "${PROP_DIR} does not exist. Creating";
  mkdir -p ${PROP_DIR}
fi

wget https://download.openmmlab.com/mmaction/dataset/thumos14/thumos14_tag_val_normalized_proposal_list.txt -P ${PROP_DIR}
wget https://download.openmmlab.com/mmaction/dataset/thumos14/thumos14_tag_test_normalized_proposal_list.txt -P ${PROP_DIR}
