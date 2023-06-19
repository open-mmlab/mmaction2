#!/usr/bin/env bash

DATA_DIR="../../../data/video_retrieval/msrvtt"
mkdir -p ${DATA_DIR}


if [ -f "msrvtt_data.zip" ]; then
    echo "msrvtt_data.zip exists, skip downloading!"
else
    echo "Downloading msrvtt_data.zip."
    wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
fi

echo "Processing annotations started."
unzip -q msrvtt_data.zip -d ${DATA_DIR}
python prepare_msrvtt.py
echo "Processing annotations completed."

if [ -f "MSRVTT.zip" ]; then
    echo "MSRVTT.zip exists, skip downloading!"
else
    echo "Downloading MSRVTT.zip."
    wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
fi

echo "Processing videos started."
unzip -q MSRVTT.zip -d ${DATA_DIR}
mkdir -p "${DATA_DIR}/videos/" && find "${DATA_DIR}/MSRVTT/videos/all" -name "video*.mp4" -exec mv {} "${DATA_DIR}/videos/" \;
echo "Processing videos completed."

rm -rf "${DATA_DIR}/MSRVTT"
rm -rf "${DATA_DIR}/msrvtt_data"
rm msrvtt_data.zip
rm MSRVTT.zip
echo "The preparation of the msrvtt dataset has been successfully completed."
