#! /usr/bin/bash env

FLOW_TYPE=$1

cd ../
python build_rawframes.py ../../data/ucf101/videos/ ../../data/ucf101/rawframes/ --task both --level 2 --flow_type ${FLOW_TYPE}
echo "Raw frames (RGB and Flow) Generated"
cd ucf101/
