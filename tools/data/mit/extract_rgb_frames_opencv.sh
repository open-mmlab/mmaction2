#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/mit/videos/training ../../data/mit/rawframes/training/ --level 2 --ext mp4 --task rgb --use-opencv
echo "Raw frames (RGB only) generated for train set"

python build_rawframes.py ../../data/mit/videos/validation ../../data/mit/rawframes/validation/ --level 2 --ext mp4 --task rgb --use-opencv
echo "Raw frames (RGB only) generated for val set"

cd mit/
