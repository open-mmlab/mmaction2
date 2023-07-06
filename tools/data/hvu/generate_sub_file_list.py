# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmengine


def main(annotation_file, category):
    assert category in [
        'action', 'attribute', 'concept', 'event', 'object', 'scene'
    ]

    data = mmengine.load(annotation_file)
    basename = osp.basename(annotation_file)
    dirname = osp.dirname(annotation_file)
    basename = basename.replace('hvu', f'hvu_{category}')

    target_file = osp.join(dirname, basename)

    result = []
    for item in data:
        label = item['label']
        if category in label:
            item['label'] = label[category]
            result.append(item)

    mmengine.dump(data, target_file)


if __name__ == '__main__':
    description = 'Helper script for generating HVU per-category file list.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        'annotation_file',
        type=str,
        help=('The annotation file which contains tags of all categories.'))
    p.add_argument(
        'category',
        type=str,
        choices=['action', 'attribute', 'concept', 'event', 'object', 'scene'],
        help='The tag category that you want to generate file list for.')
    main(**vars(p.parse_args()))
