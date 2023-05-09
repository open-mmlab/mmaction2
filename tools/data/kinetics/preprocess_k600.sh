#!/usr/bin/env bash

set -x

DOWNLOAD_DIR=$1
DATA_ROOT=$2

cat $DOWNLOAD_DIR/Kinetics600/raw/*.tar.gz*  | tar -xvz -C $(dirname $DATA_ROOT)
mv $(dirname $DATA_ROOT)/Kinetics600 $DATA_ROOT
