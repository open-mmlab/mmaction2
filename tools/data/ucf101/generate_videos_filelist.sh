#!/usr/bin/env bash

cd ../../../

PYTHONPATH=. python tools/data/build_file_list.py ucf101 /local_datasets/ucf101/videos/ --level 2 --format videos --shuffle ## 데이터 경로 수정
echo "Filelist for videos generated."

cd tools/data/ucf101/
