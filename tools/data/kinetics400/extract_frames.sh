#! /usr/bin/bash env

cd ../
python build_rawframes.py ../../data/kinetics400/videos_train/ ../../data/kinetics400/rawframes_train/ --level 2 --flow_type tvl1 --ext mp4
echo "Raw frames (RGB and tv-l1) Generated for train set"

python build_rawframes.py ../../data/kinetics400/videos_val/ ../../data/kinetics400/rawframes_val/ --level 2 --flow_type tvl1 --ext mp4
echo "Raw frames (RGB and tv-l1) Generated for val set"

cd kinetics400/
