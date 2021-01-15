#!/usr/bin/env bash

DATASET=$1
if [ "$DATASET" == "kinetics400" ] || [ "$1" == "kinetics600" ] || [ "$1" == "kinetics700" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support kinetics400, kinetics600 or kinetics700"
        exit 0
fi

cd ../
python build_rawframes.py ../../data/${DATASET}/videos_train/ ../../data/${DATASET}/rawframes_train/ --level 2  --ext mp4 --task rgb --new-short 256 --use-opencv
echo "Raw frames (RGB only) generated for train set"

python build_rawframes.py ../../data/${DATASET}/videos_val/ ../../data/${DATASET}/rawframes_val/ --level 2 --ext mp4 --task rgb --new-short 256 --use-opencv
echo "Raw frames (RGB only) generated for val set"

cd ${DATASET}/
