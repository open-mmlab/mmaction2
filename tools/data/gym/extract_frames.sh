#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/gym/subactions/ ../../data/gym/subaction_frames/ --level 1 --flow-type tvl1 --ext mp4 --task both --new-short 256
echo "Raw frames (RGB and tv-l1) Generated"

cd gym/
