#!/usr/bin/env bash

# set up environment
conda env create -f environment.yml
source activate activitynet
pip install --upgrade youtube-dl
pip install mmcv

DATA_DIR="../../../data/ActivityNet"
python download.py --bsn

source deactivate activitynet
conda remove -n activitynet --all
