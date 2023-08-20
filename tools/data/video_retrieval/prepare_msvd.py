# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import pickle

DATA_DIR = '../../../data/video_retrieval/msvd'
SUFFIX = '.avi'

data_path = osp.join(DATA_DIR, 'msvd_data/raw-captions.pkl')
train_txt_path = osp.join(DATA_DIR, 'msvd_data/train_list.txt')
test_txt_path = osp.join(DATA_DIR, 'msvd_data/test_list.txt')
val_txt_path = osp.join(DATA_DIR, 'msvd_data/val_list.txt')
train_json_path = osp.join(DATA_DIR, 'train.json')
test_json_path = osp.join(DATA_DIR, 'test.json')
val_json_path = osp.join(DATA_DIR, 'val.json')

with open(data_path, 'rb') as F:
    data = pickle.load(F)

video_dict = {}
for one_data in data:
    caption = data[one_data]
    if one_data not in video_dict:
        video_dict[one_data] = []
    for cap in caption:
        video_dict[one_data].append(' '.join(cap))

with open(train_txt_path, 'r') as f:
    train_avi = f.readlines()

train_avi_list = {}
for video in train_avi:
    train_avi_list[video.strip() + SUFFIX] = video_dict[video.strip()]

with open(train_json_path, 'w') as f:
    json.dump(train_avi_list, f)

with open(test_txt_path, 'r') as f:
    test_avi = f.readlines()

test_avi_list = {}
for video in test_avi:
    test_avi_list[video.strip() + SUFFIX] = video_dict[video.strip()]
with open(test_json_path, 'w') as f:
    json.dump(test_avi_list, f)

with open(val_txt_path, 'r') as f:
    val_avi = f.readlines()

val_avi_list = {}
for video in val_avi:
    val_avi_list[video.strip() + SUFFIX] = video_dict[video.strip()]

with open(val_json_path, 'w') as f:
    json.dump(val_avi_list, f)
