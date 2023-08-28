#!/usr/bin/env bash

set -x

DOWNLOAD_DIR=$1
DATA_ROOT=$2

cat $DOWNLOAD_DIR/OpenMMLab___Kinetics-400/raw/*.tar.gz*  | tar -xvz -C $(dirname $DATA_ROOT)
mv $(dirname $DATA_ROOT)/Kinetics-400 $DATA_ROOT
