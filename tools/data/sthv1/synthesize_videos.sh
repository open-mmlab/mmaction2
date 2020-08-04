#!/usr/bin/env bash

cd ../
python build_videos.py ../../data/sthv1/rawframes/ ../../data/sthv1/videos/ --fps 12 --level 1
echo "Genearte videos"

cd sthv1/
