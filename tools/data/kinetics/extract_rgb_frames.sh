#!/usr/bin/env bash

DATASET=$1

cd ../
python build_rawframes.py ../../data/${DATASET}/videos_train/ ../../data/${DATASET}/rawframes_train/ --level 2  --ext mp4 --task rgb  --new-width 340 --new-height 256
echo "Raw frames (RGB only) generated for train set"

python build_rawframes.py ../../data/${DATASET}/videos_val/ ../../data/${DATASET}/rawframes_val/ --level 2 --ext mp4 --task rgb  --new-width 340 --new-height 256
echo "Raw frames (RGB only) generated for val set"

cd ${DATASET}/
