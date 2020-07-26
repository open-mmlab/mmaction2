#!/usr/bin/env bash

cd ../../../
PYTHONPATH=. python tools/data/build_file_list.py mit data/mit/rawframes/training/ --level 2 --format rawframes --num-split 1 --subset train --shuffle
echo "Train filelist for rawframes generated."

PYTHONPATH=. python tools/data/build_file_list.py mit data/mit/rawframes/validation/ --level 2 --format rawframes --num-split 1 --subset val --shuffle
echo "Val filelist for rawframes generated."
cd tools/data/mit/
