# Copyright (c) OpenMMLab. All rights reserved.
import json
import multiprocessing
import os

import decord

with open('HACS_v1.1.1/HACS_segments_v1.1.1.json') as f:
    all_annotations = json.load(f)['database']


def parse_anno(key):
    anno = {}
    anno['duration_second'] = float(all_annotations[key]['duration'])
    anno['annotations'] = all_annotations[key]['annotations']
    anno['subset'] = all_annotations[key]['subset']

    labels = set([i['label'] for i in anno['annotations']])
    num_frames = int(anno['duration_second'] * 30)
    for label in labels:
        path = f'data/{label}/v_{key}.mp4'
        if os.path.isfile(path):
            vr = decord.VideoReader(path)
            num_frames = len(vr)
            break

    anno['feature_frame'] = anno['duration_frame'] = num_frames
    anno['key'] = f'v_{key}'
    return anno


pool = multiprocessing.Pool(16)
video_list = list(all_annotations)
outputs = pool.map(parse_anno, video_list)

train_anno = {}
val_anno = {}
test_anno = {}

for anno in outputs:
    key = anno.pop('key')
    subset = anno.pop('subset')
    if subset == 'training':
        train_anno[key] = anno
    elif subset == 'validation':
        val_anno[key] = anno
    else:
        test_anno[key] = anno

outdir = '../../../data/HACS'
with open(f'{outdir}/hacs_anno_train.json', 'w') as f:
    json.dump(train_anno, f)

with open(f'{outdir}/hacs_anno_val.json', 'w') as f:
    json.dump(val_anno, f)

with open(f'{outdir}/hacs_anno_test.json', 'w') as f:
    json.dump(test_anno, f)
