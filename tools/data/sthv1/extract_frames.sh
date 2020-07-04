#! /usr/bin/bash env

FLOW_TYPE=$1

cd ../
python build_rawframes.py ../../data/sthv1/videos/ ../../data/sthv1/rawframes/ --task both --level 1 --flow_type ${FLOW_TYPE} --ext webm
echo "Raw frames (RGB and tv-l1) Generated"
cd sthv1/
