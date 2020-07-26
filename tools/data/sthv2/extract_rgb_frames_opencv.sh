#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/sthv2/videos/ ../../data/sthv2/rawframes/ --task rgb --level 1 --ext webm --use-opencv
echo "Genearte raw frames (RGB only)"

cd sthv2/
