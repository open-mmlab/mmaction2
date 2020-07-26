#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/sthv1/rawframes/ ../../data/sthv1/rawframes/ --task flow --level 1 --flow-type tvl1 --input-frames
echo "Flow (tv-l1) Generated"
cd sthv1/
