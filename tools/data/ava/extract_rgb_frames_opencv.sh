#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/ava/videos/ ../../data/ava/rawframes/ --task rgb --level 1 --use-opencv
echo "Genearte raw frames (RGB only)"

cd ava/
