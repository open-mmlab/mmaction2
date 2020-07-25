#! /usr/bin/bash env

DATA_DIR="../../../data/ucf101/"

cd ${DATA_DIR}

wget https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar --no-check-certificate
unrar x UCF101.rar
mv ./UCF-101 ./videos

cd "../../../tools/data/ucf101"
