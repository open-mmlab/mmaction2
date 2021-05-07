#!/usr/bin/env bash

# set up environment
conda env create -f environment.yml
source activate gym
pip install mmcv
pip install --upgrade youtube-dl

DATA_DIR="../../../data/gym"
ANNO_DIR="../../../data/gym/annotations"
python download.py ${ANNO_DIR}/annotation.json ${DATA_DIR}/videos

source deactivate gym
conda remove -n gym --all
