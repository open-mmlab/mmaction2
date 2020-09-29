#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/ava/videos/ ../../data/ava/rawframes/ --task rgb --level 1
echo "Genearte raw frames (RGB only)"

cd ava/
