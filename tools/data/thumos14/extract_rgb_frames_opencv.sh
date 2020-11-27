#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/thumos14/videos/val/ ../../data/thumos14/rawframes/val/ --level 1 --ext mp4 --task rgb --use-opencv
echo "Raw frames (RGB only) generated for val set"

python build_rawframes.py ../../data/thumos14/videos/test/ ../../data/thumos14/rawframes/test/ --level 1 --ext mp4 --task rgb --use-opencv
echo "Raw frames (RGB only) generated for test set"

cd thumos14/
