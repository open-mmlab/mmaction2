"""This file processes the annotation files and generates proper annotation
files for localizers."""
import json

import numpy as np


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


data_file = '../../../data/ActivityNet'
info_file = f'{data_file}/video_info_new.csv'
ann_file = f'{data_file}/anet_anno_action.json'

anno_database = load_json(ann_file)

video_record = np.loadtxt(info_file, dtype=np.str, delimiter=',', skiprows=1)

video_dict_train = {}
video_dict_val = {}
video_dict_test = {}
video_dict_full = {}

for _, video_item in enumerate(video_record):
    video_name = video_item[0]
    video_info = anno_database[video_name]
    video_subset = video_item[5]
    video_info['fps'] = video_item[3].astype(np.float)
    video_info['rfps'] = video_item[4].astype(np.float)
    video_dict_full[video_name] = video_info
    if video_subset == 'training':
        video_dict_train[video_name] = video_info
    elif video_subset == 'testing':
        video_dict_test[video_name] = video_info
    elif video_subset == 'validation':
        video_dict_val[video_name] = video_info

print(f'full subset video numbers: {len(video_record)}')

with open(f'{data_file}/anet_anno_train.json', 'w') as result_file:
    json.dump(video_dict_train, result_file)

with open(f'{data_file}/anet_anno_val.json', 'w') as result_file:
    json.dump(video_dict_val, result_file)

with open(f'{data_file}/anet_anno_test.json', 'w') as result_file:
    json.dump(video_dict_test, result_file)

with open(f'{data_file}/anet_anno_full.json', 'w') as result_file:
    json.dump(video_dict_full, result_file)
