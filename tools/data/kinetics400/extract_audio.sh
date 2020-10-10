#!/usr/bin/env bash

cd ../
python extract_audio.py ../../data/kinetics400/videos_train/ ../../data/kinetics400/audio_train/ --level 2 --ext mp4
echo "Audios Generated for train set"

python extract_audio.py ../../data/kinetics400/videos_val/ ../../data/kinetics400/audio_val/ --level 2 --ext mp4
echo "Audios Generated for test set"

cd kinetics400/
