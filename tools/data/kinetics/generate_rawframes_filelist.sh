#!/usr/bin/env bash

DATASET=$1
if [ "$DATASET" == "kinetics400" ] || [ "$1" == "kinetics600" ] || [ "$1" == "kinetics700" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support kinetics400, kinetics600 or kinetics700"
        exit 0
fi

cd ../../../
PYTHONPATH=. python tools/data/build_file_list.py ${DATASET} data/${DATASET}/rawframes_train/ --level 2 --format rawframes --num-split 1 --subset train --shuffle
echo "Train filelist for rawframes generated."

PYTHONPATH=. python tools/data/build_file_list.py ${DATASET} data/${DATASET}/rawframes_val/ --level 2 --format rawframes --num-split 1 --subset val --shuffle
echo "Val filelist for rawframes generated."
cd tools/data/${DATASET}/
