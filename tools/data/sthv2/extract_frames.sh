#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/sthv2/videos/ ../../data/sthv2/rawframes/ --task both --level 1 --flow-type tvl1 --ext webm
echo "Raw frames (RGB and tv-l1) Generated"
cd sthv2/
