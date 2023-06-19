# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
from mmengine import dump, load, track_iter_progress


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--det_test_result',
        default='data/multisports/annotations/ms_det_proposals.pkl')
    parser.add_argument(
        '--stad_gt',
        help='spatio-temporal action detection ground truth file',
        default='data/multisports/annotations/multisports_GT.pkl')
    parser.add_argument(
        '--out_result',
        default='data/multisports/annotations/multisports_proposals.pkl')
    args = parser.parse_args()
    return args


def dump_det_result(args):
    print('loading test result...')
    det_result = load(args.det_test_result)
    stad_gt = load(args.stad_gt)
    train_list = stad_gt['train_videos'][0]
    val_list = stad_gt['test_videos'][0]
    train_bbox_result = {}
    val_bbox_result = {}
    for sample in track_iter_progress(det_result):
        bboxes = sample['pred_instances']['bboxes']
        scores = sample['pred_instances']['scores']
        h, w = sample['ori_shape']
        bboxes[:, ::2] /= w
        bboxes[:, 1::2] /= h
        img_path = sample['img_path']
        frm_key_list = img_path.split('.jpg')[0].split('/')
        frm_key = ','.join([
            f'{frm_key_list[-3]}/{frm_key_list[-2]}.mp4',
            f'{int(frm_key_list[-1]):04d}'
        ])
        bbox = np.concatenate([bboxes, scores[:, None]], axis=1)

        vid_key = '/'.join(frm_key_list[-3:-1])
        if vid_key in train_list:
            train_bbox_result[frm_key] = bbox
        elif vid_key in val_list:
            val_bbox_result[frm_key] = bbox
        else:
            raise KeyError(vid_key)
    dump(train_bbox_result, args.out_result[:-4] + '_train.pkl')
    dump(val_bbox_result, args.out_result[:-4] + '_val.pkl')


if __name__ == '__main__':
    args = parse_args()
    dump_det_result(args)
