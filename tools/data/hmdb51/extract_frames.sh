#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/hmdb51/videos/ ../../data/hmdb51/rawframes/ --task both --level 2 --flow-type tvl1
echo "Raw frames (RGB and Flow) Generated"
cd hmdb51/
