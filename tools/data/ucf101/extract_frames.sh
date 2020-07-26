#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/ucf101/videos/ ../../data/ucf101/rawframes/ --task both --level 2 --flow-type tvl1
echo "Raw frames (RGB and Flow) Generated"
cd ucf101/
