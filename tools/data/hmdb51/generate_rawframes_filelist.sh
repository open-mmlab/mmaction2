#!/usr/bin/env bash

cd ../../../

PYTHONPATH=. python tools/data/build_file_list.py hmdb51 data/hmdb51/rawframes/ --level 2 --format rawframes --shuffle
echo "Filelist for rawframes generated."

cd tools/data/hmdb51/
