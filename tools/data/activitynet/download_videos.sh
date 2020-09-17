#!/usr/bin/env bash

# set up environment
conda env create -f environment.yml
source activate activitynet
pip install --upgrade youtube-dl

DATA_DIR="../../../data/ActivityNet"
python download.py ${DATA_DIR}/video_info_new.csv ${DATA_DIR}/videos

source deactivate activitynet
conda remove -n activitynet --all
