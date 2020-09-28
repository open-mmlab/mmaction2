#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/hvu/videos_train/ ../../data/hvu/rawframes_train/ --level 1 --flow-type tvl1 --ext mp4 --task both  --new-short 256
echo "Raw frames (RGB and tv-l1) Generated for train set"

python build_rawframes.py ../../data/hvu/videos_val/ ../../data/hvu/rawframes_val/ --level 1 --flow-type tvl1 --ext mp4 --task both  --new-short 256
echo "Raw frames (RGB and tv-l1) Generated for val set"

cd hvu/
