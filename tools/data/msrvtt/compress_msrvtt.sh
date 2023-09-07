#!/usr/bin/env bash

FPS=2
SIZE=224
DATA_DIR="../../../data/msrvtt/videos"
OUT_DIR="../../../data/msrvtt/videos_2fps_224"

python compress.py \
    --input_root=${DATA_DIR} --output_root=${OUT_DIR} \
    --fps=${FPS} --size=${SIZE} --file_type=video --num_workers 24
