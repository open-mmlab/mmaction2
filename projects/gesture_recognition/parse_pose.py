# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert and merge hand pose dataset to COCO style')
    parser.add_argument(
        '--data_root',
        type=str,
        default='./data/',
        help='the root to all involved datasets')
    parser.add_argument(
        '--out_anno_prefix',
        type=str,
        default='hand_det',
        help='the prefix of output annotation files')

    args = parser.parse_args()
    return args


def get_data_root(path):
    path = path.split('/')
    index = path.index('annotations') - 1
    root = path[index]
    if root == 'halpe':
        root = 'halpe/hico_20160224_det/images/train2015/'
    return root


def parse_coco_style(file_path, anno_idx=0):
    with open(file_path) as f:
        contents = json.load(f)

    data_root = get_data_root(file_path) + '/'
    images = contents['images']
    annos = contents['annotations']
    images_out, annos_out = [], []
    for img, anno in zip(images, annos):
        assert img['id'] == anno['image_id']
        img_out = dict(
            file_name=data_root + img['file_name'],
            height=img['height'],
            width=img['width'],
            id=anno_idx)
        anno_out = dict(
            area=anno['area'],
            iscrowd=anno['iscrowd'],
            image_id=anno_idx,
            bbox=anno['bbox'],
            category_id=0,
            id=anno_idx)
        anno_idx += 1
        images_out.append(img_out)
        annos_out.append(anno_out)
    return images_out, annos_out, anno_idx


def parse_halpe(file_path, anno_idx):

    def get_bbox(keypoints):
        """Get bbox from keypoints."""
        if len(keypoints) == 0:
            return [0, 0, 0, 0]
        x1, y1, _ = np.amin(keypoints, axis=0)
        x2, y2, _ = np.amax(keypoints, axis=0)
        w, h = x2 - x1, y2 - y1
        return [x1, y1, w, h]

    with open(file_path) as f:
        contents = json.load(f)

    data_root = get_data_root(file_path) + '/'
    images = contents['images']
    annos = contents['annotations']
    images_out, annos_out = [], []
    for img, anno in zip(images, annos):
        assert img['id'] == anno['image_id']
        keypoints = np.array(anno['keypoints']).reshape(-1, 3)
        lefthand_kpts = keypoints[-42:-21, :]
        righthand_kpts = keypoints[-21:, :]

        left_mask = lefthand_kpts[:, 2] > 0
        right_mask = righthand_kpts[:, 2] > 0
        lefthand_box = get_bbox(lefthand_kpts[left_mask])
        righthand_box = get_bbox(righthand_kpts[right_mask])

        if max(lefthand_box) > 0:
            img_out = dict(
                file_name=data_root + img['file_name'],
                height=img['height'],
                width=img['width'],
                id=anno_idx)
            anno_out = dict(
                area=lefthand_box[2] * lefthand_box[3],
                iscrowd=anno['iscrowd'],
                image_id=anno_idx,
                bbox=lefthand_box,
                category_id=0,
                id=anno_idx)
            anno_idx += 1
            images_out.append(img_out)
            annos_out.append(anno_out)

        if max(righthand_box) > 0:
            img_out = dict(
                file_name=data_root + img['file_name'],
                height=img['height'],
                width=img['width'],
                id=anno_idx)
            anno_out = dict(
                area=righthand_box[2] * righthand_box[3],
                iscrowd=anno['iscrowd'],
                image_id=anno_idx,
                bbox=righthand_box,
                category_id=0,
                id=anno_idx)
            anno_idx += 1
            images_out.append(img_out)
            annos_out.append(anno_out)
    return images_out, annos_out, anno_idx


train_files = [
    'freihand/annotations/freihand_train.json',
    'halpe/annotations/halpe_train_v1.json',
    'onehand10k/annotations/onehand10k_train.json',
    '/rhd/annotations/rhd_train.json'
]

val_files = ['onehand10k/annotations/onehand10k_test.json']


def convert2dict(data_root, anno_files):
    anno_files = [data_root + _ for _ in anno_files]

    images, annos, anno_idx = [], [], 0
    for anno_file in anno_files:
        if 'freihand' in anno_file or 'onehand10k' in anno_file \
                                   or 'rhd' in anno_file:
            images_out, annos_out, anno_idx = parse_coco_style(
                anno_file, anno_idx)
            images += images_out
            annos += annos_out
        elif 'halpe' in anno_file:
            images_out, annos_out, anno_idx = parse_halpe(anno_file, anno_idx)
            images += images_out
            annos += annos_out
        else:
            print(f'{anno_file} not supported')

    result = dict(
        images=images,
        annotations=annos,
        categories=[{
            'id': 0,
            'name': 'hand'
        }])
    return result


if __name__ == '__main__':
    args = parse_args()
    data_root = args.data_root + '/'
    prefix = args.out_anno_prefix
    os.makedirs('hand_det', exist_ok=True)

    result = convert2dict(data_root, train_files)
    with open(f'hand_det/{prefix}_train.json', 'w') as f:
        json.dump(result, f)

    result = convert2dict(data_root, val_files)
    with open(f'hand_det/{prefix}_val.json', 'w') as f:
        json.dump(result, f)
