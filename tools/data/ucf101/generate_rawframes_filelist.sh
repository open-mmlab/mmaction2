#!/usr/bin/env bash

cd ../../../

PYTHONPATH=. python tools/data/build_file_list.py ucf101 data/ucf101/rawframes/ --level 2 --format rawframes --shuffle
echo "Filelist for rawframes generated."

cd tools/data/ucf101/
