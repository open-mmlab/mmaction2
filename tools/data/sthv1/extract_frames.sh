#! /usr/bin/bash env

cd ../
python build_rawframes.py ../../data/sthv1/videos/ ../../data/sthv1/rawframes/ --task both --level 1 --flow_type tvl1 --ext webm
echo "Raw frames (RGB and tv-l1) Generated"
cd sthv1/
