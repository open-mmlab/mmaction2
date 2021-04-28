#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/diving48/videos/ ../../data/diving48/rawframes/ --task both --level 1 --flow-type tvl1 --ext mp4
echo "Raw frames (RGB and tv-l1) Generated"
cd -
