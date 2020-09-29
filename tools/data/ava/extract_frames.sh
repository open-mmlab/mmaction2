#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/ava/videos/ ../../data/ava/rawframes/ --task both --level 1 --flow-type tvl1
echo "Raw frames (RGB and Flow) Generated"
cd ava/
