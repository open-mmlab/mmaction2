#!/usr/bin/env bash

# set up environment
conda env create -f environment.yml
source activate hvu
pip install mmcv
pip install --upgrade youtube-dl

DATA_DIR="../../../data/hvu"
ANNO_DIR="../../../data/hvu/annotations"
python download.py ${ANNO_DIR}/hvu_train.csv ${DATA_DIR}/videos_train
python download.py ${ANNO_DIR}/hvu_val.csv ${DATA_DIR}/videos_val

source deactivate hvu
conda remove -n hvu --all
