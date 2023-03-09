# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import numpy as np
from mmengine import dump, load
from multisports_build_tubes import BuildTubes
from rich.progress import track


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--test-result', default='')
    parser.add_argument(
        '--anno-path',
        default='data/multisports/videos/trainval/multisports_GT.pkl')
    parser.add_argument('--frm_out_path', default=None)
    parser.add_argument('--tube_out_path', default=None)
    args = parser.parse_args()
    if not args.frm_out_path:
        args.frm_out_path = args.test_result[:-4] + '-formated.pkl'
    if not args.tube_out_path:
        args.tube_out_path = args.test_result[:-4] + '_vid_dets.pkl'
    return args


label_offset = 0


def format_det_result():
    """convert test results to specified format in MultiSports competition."""
    test_results = load(args.test_result)
    annos = load(args.anno_path)
    test_videos = annos['test_videos'][0]
    resolutions = annos['resolution']
    frm_dets = []
    for pred in track(test_results, description='formating...'):
        video_key = pred['video_id'].split('.mp4')[0]
        frm_num = pred['timestamp']
        bboxes = pred['pred_instances']['bboxes']
        cls_scores = pred['pred_instances']['scores']
        for bbox, cls_score in zip(bboxes, cls_scores):
            video_idx = test_videos.index(video_key)
            pred_label = np.argmax(cls_score)
            score = cls_score[pred_label]
            h, w = resolutions[video_key]
            bbox *= np.array([w, h, w, h])
            instance_result = np.array(
                [video_idx, frm_num, pred_label - label_offset, score, *bbox])
            frm_dets.append(instance_result)
    frm_dets = np.array(frm_dets)
    dump(frm_dets, args.frm_out_path)

    BuildTubes(args.anno_path, args.frm_out_path, args.tube_out_path, K=1)


if __name__ == '__main__':
    args = parse_args()
    format_det_result()
