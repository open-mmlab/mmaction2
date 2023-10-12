#!/usr/bin/env bash

DATA_DIR="../../../data/video_retrieval/msvd"
mkdir -p ${DATA_DIR}


if [ -f "msvd_data.zip" ]; then
    echo "msvd_data.zip exists, skip downloading!"
else
    echo "Downloading msvd_data.zip."
    wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msvd_data.zip
fi

echo "Processing annotations started."
unzip -q msvd_data.zip -d ${DATA_DIR}
python prepare_msvd.py
echo "Processing annotations completed."

if [ -f "YouTubeClips.tar" ]; then
    echo "YouTubeClips.tar exists, skip downloading!"
else
    echo "Downloading YouTubeClips.tar."
    wget https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar
fi

echo "Processing videos started."
tar -xf YouTubeClips.tar -C ${DATA_DIR}
mkdir -p "${DATA_DIR}/videos/" && find "${DATA_DIR}/YouTubeClips" -name "*.avi" -exec mv {} "${DATA_DIR}/videos/" \;
echo "Processing videos completed."

rm -rf "${DATA_DIR}/YouTubeClips"
rm -rf "${DATA_DIR}/msvd_data"
rm msvd_data.zip
rm YouTubeClips.tar
echo "The preparation of the msvd dataset has been successfully completed."
