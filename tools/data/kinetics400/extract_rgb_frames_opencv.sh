#! /usr/bin/bash env

cd ../
python build_rawframes.py ../../data/kinetics400/videos_train/ ../../data/kinetics400/rawframes_train/ --level 2  --ext mp4 --task rgb  --use-opencv
echo "Raw frames (RGB only) generated for train set"

python build_rawframes.py ../../data/kinetics400/videos_val/ ../../data/kinetics400/rawframes_val/ --level 2 --ext mp4 --task rgb  --use-opencv
echo "Raw frames (RGB only) generated for val set"

cd kinetics400/
