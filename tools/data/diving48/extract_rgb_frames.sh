#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/diving48/videos/ ../../data/diving48/rawframes/ --task rgb --level 1  --ext mp4
echo "Genearte raw frames (RGB only)"

cd -
