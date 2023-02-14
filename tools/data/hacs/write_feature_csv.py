# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle

with open('result.pkl', 'rb') as f:
    features = pickle.load(f)

with open('hacs_data.txt', 'r') as f:
    video_list = f.readlines()

data_dir = '../../../data/HACS'
os.makedirs(data_dir, exist_ok=True)
feature_dir = f'{data_dir}/slowonly_feature'
os.makedirs(feature_dir)

head = ','.join([f'f{i}' for i in range(700)]) + '\n'

for feature, video in zip(features, video_list):
    video_id = video.split()[0].split('/')[1]
    csv_file = video_id.replace('mp4', 'csv')
    feat = feature['pred_scores']['item'].numpy()
    feat = feat.tolist()
    csv_path = f'{feature_dir}/{csv_file}'
    with open(csv_path, 'w') as f:
        f.write(head)
        for line in feat:
            f.write(str(line)[1:-1] + '\n')
