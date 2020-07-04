#! /usr/bin/bash env

cd ../../../
PYTHONPATH=. python tools/data/build_file_list.py mmit data/mmit/videos/ --level 2 --format videos --num_split 1 --subset train --shuffle
echo "Train filelist for videos generated."

PYTHONPATH=. python tools/data/build_file_list.py mmit data/mmit/videos/ --level 2 --format videos --num_split 1 --subset val --shuffle
echo "Val filelist for videos generated."
cd tools/data/mmit/
