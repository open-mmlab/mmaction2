#! /usr/bin/bash env

cd ../
python build_rawframes.py ../../data/thumos14/videos/validation/ ../../data/thumos14/rawframes/validation/ --level 1 --flow_type tvl1 --ext mp4 --task both
echo "Raw frames (RGB and tv-l1) Generated for val set"

python build_rawframes.py ../../data/thumos14/videos/test/ ../../data/thumos14/rawframes/test/ --level 1 --flow_type tvl1 --ext mp4 --task both
echo "Raw frames (RGB and tv-l1) Generated for test set"

cd thumos14/
