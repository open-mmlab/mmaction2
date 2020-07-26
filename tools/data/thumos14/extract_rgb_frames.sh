#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/thumos14/videos/validation/ ../../data/thumos14/rawframes/validation/ --level 1 --ext mp4 --task rgb
echo "Raw frames (RGB only) generated for val set"

python build_rawframes.py ../../data/thumos14/videos/test/ ../../data/thumos14/rawframes/test/ --level 1 --ext mp4 --task rgb
echo "Raw frames (RGB only) generated for test set"

cd thumos14/
