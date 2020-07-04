#! /usr/bin/bash env

cd ../
python build_rawframes.py ../../data/sthv1/videos/ ../../data/sthv1/rawframes/ --task rgb --level 1  --ext webm
echo "Genearte raw frames (RGB only)"

cd sthv1/
