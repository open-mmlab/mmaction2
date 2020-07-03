#! /usr/bin/bash env

DATA_DIR="../../../data/ucf101/"

cd ${DATA_DIR}

wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x UCF101.rar
mv ./UCF-101 ./videos

cd "../../../data_tools/ucf101"
