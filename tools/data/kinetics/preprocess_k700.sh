#!/usr/bin/env bash

set -x

DOWNLOAD_DIR=$1
DATA_ROOT=$2

cat $DOWNLOAD_DIR/OpenMMLab___Kinetics_700/raw/*.tar.gz*  | tar -xvz -C $(dirname $DATA_ROOT)
mv $(dirname $DATA_ROOT)/Kinetics_700 $DATA_ROOT
