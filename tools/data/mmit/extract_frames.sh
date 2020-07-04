#! /usr/bin/bash env

cd ../
python build_rawframes.py ../../data/mmit/videos/ ../../../data/mmit/rawframes/ --task both --level 2 --flow_type tvl1 --ext mp4
echo "Raw frames (RGB and Flow) Generated"
cd mmit/
