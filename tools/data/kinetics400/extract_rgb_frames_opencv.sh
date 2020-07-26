#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/kinetics400/videos_train/ ../../data/kinetics400/rawframes_train/ --level 2  --ext mp4 --task rgb --new-width 340 --new-height 256 --use-opencv
echo "Raw frames (RGB only) generated for train set"

python build_rawframes.py ../../data/kinetics400/videos_val/ ../../data/kinetics400/rawframes_val/ --level 2 --ext mp4 --task rgb --new-width 340 --new-height 256 --use-opencv
echo "Raw frames (RGB only) generated for val set"

cd kinetics400/
