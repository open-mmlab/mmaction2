#!/usr/bin/env bash

cd ../
python build_videos.py ../../data/sthv1/rawframes/ ../../data/sthv1/videos/ --fps 12 --level 1 --start-idx 1 --filename-tmpl '%05d'
echo "Encode videos"

cd sthv1/
