# Copyright (c) OpenMMLab. All rights reserved.
import csv
import os
import os.path as osp
from argparse import ArgumentParser

import numpy as np
from mmengine import dump, list_dir_or_file, load


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--anno_root', default='data/multisports/videos/trainval')
    parser.add_argument('--out_root', default='data/multisports_ori')
    parser.add_argument(
        '--test_data_root', default='data/multisports/videos/test')
    args = parser.parse_args()
    return args


exclude_categories = [
    'aerobic kick jump', 'aerobic off axis jump', 'aerobic butterfly jump',
    'aerobic balance turn', 'basketball save', 'basketball jump ball'
]


def parse_anno(args):
    if not osp.exists(args.out_root):
        os.makedirs(osp.join(args.out_root, 'annotations'))

    anno_path = osp.join(args.anno_root, 'multisports_GT.pkl')
    annos = load(anno_path)

    # convert proposal key to filename
    key2filename = {
        video.split('/')[1]: video + '.mp4'
        for video in annos['nframes'].keys()
    }
    test_videos = [
        file for file in list_dir_or_file(args.test_data_root, recursive=True)
        if file.endswith('.mp4')
    ]
    key2filename.update(
        {video.split('/')[1][:-4]: video
         for video in test_videos})
    proposals_path = osp.join(args.anno_root, 'MultiSports_box')
    for proposals in os.listdir(proposals_path):
        proposal_info = load(osp.join(proposals_path, proposals))
        proposal_out = dict()
        for key in proposal_info.keys():
            key_split = key.split(',')
            if key_split[0] in key2filename.keys():
                new_key = \
                    f'{key2filename[key_split[0]]},{int(key_split[1]):04d}'
            proposal_out[new_key] = proposal_info[key]
        target_path = osp.join(args.out_root, 'annotations',
                               'multisports_dense_proposals_' + proposals)
        dump(proposal_out, target_path)

    # dump labels
    labels = annos['labels']
    # label_offset = 1  # mmaction2 use 1-index label
    label_offset = 0  # multisport evaluation
    with open(
            osp.join(args.out_root, 'annotations',
                     'multisports_label_file.txt'), 'w') as f:
        for idx, label in enumerate(labels):
            # filter excluded categories
            if label in exclude_categories:
                continue
            line = f'item {{\n  name:"{label}"\n  id: {idx+label_offset}\n}}\n'
            f.write(line)

    for split in ['train', 'val']:
        out_anno_path = osp.join(args.out_root, 'annotations',
                                 f'multisports_{split}.csv')
        with open(out_anno_path, 'w') as csv_f:
            writer = csv.writer(csv_f)
            if split == 'train':
                video_list = annos['train_videos'][0]
            elif split == 'val':
                video_list = annos['test_videos'][0]
            gt_tubes = annos['gttubes']
            resolutions = annos['resolution']
            for video_id in video_list:
                vid_tubes = gt_tubes[video_id]
                h, w = resolutions[video_id]
                for label, tubes in vid_tubes.items():
                    entity_id = 0
                    for tube in tubes:
                        for frame_anno in tube:
                            frame_stamp = int(frame_anno[0])
                            entity_box = frame_anno[1:]
                            entity_box /= np.array([w, h, w, h])
                            entity_box = [f'{num:.3f}' for num in entity_box]
                            filename = video_id + '.mp4'
                            anno_line = [
                                filename, frame_stamp, *entity_box,
                                label + label_offset, entity_id
                            ]
                            writer.writerow(anno_line)
                        entity_id += 1


if __name__ == '__main__':
    args = parse_args()
    parse_anno(args)
