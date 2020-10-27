#!/usr/bin/env bash

cd ../../../
PYTHONPATH=. python tools/data/build_file_list.py sthv1 data/sthv1/rawframes/ --rgb-prefix '0' --num-split 1 --level 1 --subset train --format rawframes --shuffle
PYTHONPATH=. python tools/data/build_file_list.py sthv1 data/sthv1/rawframes/ --rgb-prefix '0' --num-split 1 --level 1 --subset val --format rawframes --shuffle
echo "Filelist for rawframes generated."

cd tools/data/sthv1/
