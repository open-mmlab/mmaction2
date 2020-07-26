#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/mit/videos/training ../../data/mit/rawframes/training/ --level 2 --flow-type tvl1 --ext mp4 --task both
echo "Raw frames (RGB and tv-l1) Generated for train set"

python build_rawframes.py ../../data/mit/vides/validation/ ../../data/mit/rawframes/validation/ --level 2 --flow-type tvl1 --ext mp4 --task both
echo "Raw frames (RGB and tv-l1) Generated for val set"

cd mit/
