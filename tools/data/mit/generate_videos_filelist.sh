#!/usr/bin/env bash

cd ../../../
PYTHONPATH=. python tools/data/build_file_list.py mit data/mit/videos/training/ --level 2 --format videos --num-split 1 --subset train --shuffle
echo "Train filelist for videos generated."

PYTHONPATH=. python tools/data/build_file_list.py mit data/mit/videos/validation/ --level 2 --format videos --num-split 1 --subset val --shuffle
echo "Val filelist for videos generated."
cd tools/data/mit/
