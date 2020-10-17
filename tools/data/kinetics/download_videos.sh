#!/usr/bin/env bash

# set up environment
conda env create -f environment.yml
source activate kinetics
pip install --upgrade youtube-dl

DATASET=$1
if [ "$DATASET" == "kinetics400" ] || [ "$1" == "kinetics600" ] || [ "$1" == "kinetics700" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support kinetics400, kinetics600 or kinetics700"
        exit 0
fi

DATA_DIR="../../../data/${DATASET}"
ANNO_DIR="../../../data/${DATASET}/annotations"
python download.py ${ANNO_DIR}/kinetics_train.csv ${DATA_DIR}/videos_train
python download.py ${ANNO_DIR}/kinetics_val.csv ${DATA_DIR}/videos_val

source deactivate kinetics
conda remove -n kinetics --all
