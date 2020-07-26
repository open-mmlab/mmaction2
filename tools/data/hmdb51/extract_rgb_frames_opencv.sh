#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/hmdb51/videos/ ../../data/hmdb51/rawframes/ --task rgb --level 2 --ext avi --use-opencv
echo "Genearte raw frames (RGB only)"

cd hmdb51/
