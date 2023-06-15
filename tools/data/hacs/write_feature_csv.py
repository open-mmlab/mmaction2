# Copyright (c) OpenMMLab. All rights reserved.
import mmengine

features = mmengine.load('result.pkl')
video_list = mmengine.list_from_file('hacs_data.txt')
feature_dir = '../../../data/HACS/slowonly_feature'
mmengine.mkdir_or_exist(feature_dir)

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
