#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/ava/videos_15min/ ../../data/ava/rawframes/ --task rgb --level 1 --mixed-ext
echo "Genearte raw frames (RGB only)"

cd ava/
