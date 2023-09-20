#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

cat $DOWNLOAD_DIR/OpenDataLab___sthv2/raw/*.tar.gz  | tar -xvz -C $(dirname $DATA_ROOT)
tar -xvf $DATA_ROOT/sthv2.tar -C $(dirname $DATA_ROOT)
rm $DATA_ROOT/sthv2.tar
