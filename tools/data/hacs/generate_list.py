# Copyright (c) OpenMMLab. All rights reserved.
import os

data_root = './data'

video_list = []
idx = 0
for folder in os.listdir(data_root):
    path = f'{data_root}/{folder}'
    for video in os.listdir(path):
        line = f'{folder}/{video} {idx}\n'
        idx += 1
        video_list.append(line)

with open('hacs_data.txt', 'w') as f:
    for line in video_list:
        f.write(line)
