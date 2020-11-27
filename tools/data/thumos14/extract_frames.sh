#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/thumos14/videos/val/ ../../data/thumos14/rawframes/val/ --level 1 --flow-type tvl1 --ext mp4 --task both
echo "Raw frames (RGB and tv-l1) Generated for val set"

python build_rawframes.py ../../data/thumos14/videos/test/ ../../data/thumos14/rawframes/test/ --level 1 --flow-type tvl1 --ext mp4 --task both
echo "Raw frames (RGB and tv-l1) Generated for test set"

cd thumos14/
