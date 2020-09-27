#!/usr/bin/env bash

DATA_DIR="../../../data/ucf24/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

# download ucf24.tar.gz
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1o2l6nYhd-0DDXGP-IPReBP4y1ffVmGSE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1o2l6nYhd-0DDXGP-IPReBP4y1ffVmGSE" -O ucf24.tar.gz && rm -rf /tmp/cookies.txt

echo "success download ucf24.tar.gz"

tar -zxvf ucf24.tar.gz --strip-components 1 -C ${DATA_DIR}/
rm ucf24.tar.gz
rm ${DATA_DIR}/splitfiles/finalAnnots.mat
rm -r ${DATA_DIR}/train_data

echo "untar finished."
