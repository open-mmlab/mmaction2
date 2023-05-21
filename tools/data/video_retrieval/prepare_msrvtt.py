# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

import pandas as pd

DATA_DIR = '../../../data/video_retrieval/msrvtt'
SUFFIX = '.mp4'

raw_data_path = osp.join(DATA_DIR, 'msrvtt_data/MSRVTT_data.json')
train_csv_path = [
    osp.join(DATA_DIR, 'msrvtt_data/MSRVTT_train.9k.csv'),
    osp.join(DATA_DIR, 'msrvtt_data/MSRVTT_train.7k.csv')
]
test_csv_path = osp.join(DATA_DIR, 'msrvtt_data/MSRVTT_JSFUSION_test.csv')
train_json_path = [
    osp.join(DATA_DIR, 'train_9k.json'),
    osp.join(DATA_DIR, 'train_7k.json')
]
test_json_path = osp.join(DATA_DIR, 'test_JSFUSION.json')

with open(raw_data_path, 'r') as f:
    data = json.load(f)

sentences = data['sentences']
video_dict = {}
for sentence in sentences:
    caption = sentence['caption']
    video_id = sentence['video_id']
    if video_id not in video_dict:
        video_dict[video_id] = []
    video_dict[video_id].append(caption)

for ip, op in zip(train_csv_path, train_json_path):
    train_csv = pd.read_csv(ip)
    train_video_ids = list(train_csv['video_id'].values)
    train_video_dict = {}
    for video_id in train_video_ids:
        train_video_dict[video_id + SUFFIX] = video_dict[video_id]

    with open(op, 'w') as f:
        json.dump(train_video_dict, f)

test_data = pd.read_csv(test_csv_path)

test_video_dict = {}
for video_id, sentence in zip(test_data['video_id'], test_data['sentence']):
    test_video_dict[video_id + SUFFIX] = [sentence]

with open(test_json_path, 'w') as f:
    json.dump(test_video_dict, f)
