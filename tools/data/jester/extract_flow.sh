#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/jester/rawframes/ ../../data/jester/rawframes/ --task flow --level 1 --flow-type tvl1 --input-frames
echo "Flow (tv-l1) Generated"
cd jester/
