# Copyright (c) OpenMMLab. All rights reserved.
"""This file converts the output proposal file of proposal generator (BSN, BMN)
into the input proposal file of action classifier (Currently supports SSN and
P-GCN, not including TSN, I3D etc.)."""
import argparse

import mmcv
import numpy as np

from mmaction.core import pairwise_temporal_iou


def load_annotations(ann_file):
    """Load the annotation according to ann_file into video_infos."""
    video_infos = []
    anno_database = mmcv.load(ann_file)
    for video_name in anno_database:
        video_info = anno_database[video_name]
        video_info['video_name'] = video_name
        video_infos.append(video_info)
    return video_infos


def import_ground_truth(video_infos, activity_index):
    """Read ground truth data from video_infos."""
    ground_truth = {}
    for video_info in video_infos:
        video_id = video_info['video_name'][2:]
        this_video_ground_truths = []
        for ann in video_info['annotations']:
            t_start, t_end = ann['segment']
            label = activity_index[ann['label']]
            this_video_ground_truths.append([t_start, t_end, label])
        ground_truth[video_id] = np.array(this_video_ground_truths)
    return ground_truth


def import_proposals(result_dict):
    """Read predictions from result dict."""
    proposals = {}
    num_proposals = 0
    for video_id in result_dict:
        result = result_dict[video_id]
        this_video_proposals = []
        for proposal in result:
            t_start, t_end = proposal['segment']
            score = proposal['score']
            this_video_proposals.append([t_start, t_end, score])
            num_proposals += 1
        proposals[video_id] = np.array(this_video_proposals)
    return proposals, num_proposals


def dump_formatted_proposal(video_idx, video_id, num_frames, fps, gts,
                            proposals, tiou, t_overlap_self,
                            formatted_proposal_file):
    """dump the formatted proposal file, which is the input proposal file of
    action classifier (e.g: SSN).

    Args:
        video_idx (int): Index of video.
        video_id (str): ID of video.
        num_frames (int): Total frames of the video.
        fps (float): Fps of the video.
        gts (np.ndarray[float]): t_start, t_end and label of groundtruths.
        proposals (np.ndarray[float]): t_start, t_end and score of proposals.
        tiou (np.ndarray[float]): 2-dim array with IoU ratio.
        t_overlap_self (np.ndarray[float]): 2-dim array with overlap_self
            (union / self_len) ratio.
        formatted_proposal_file (open file object): Open file object of
            formatted_proposal_file.
    """

    formatted_proposal_file.write(
        f'#{video_idx}\n{video_id}\n{num_frames}\n{fps}\n{gts.shape[0]}\n')
    for gt in gts:
        formatted_proposal_file.write(f'{int(gt[2])} {gt[0]} {gt[1]}\n')
    formatted_proposal_file.write(f'{proposals.shape[0]}\n')

    best_iou = np.amax(tiou, axis=0)
    best_iou_index = np.argmax(tiou, axis=0)
    best_overlap = np.amax(t_overlap_self, axis=0)
    best_overlap_index = np.argmax(t_overlap_self, axis=0)

    for i in range(proposals.shape[0]):
        index_iou = best_iou_index[i]
        index_overlap = best_overlap_index[i]
        label_iou = gts[index_iou][2]
        label_overlap = gts[index_overlap][2]
        if label_iou != label_overlap:
            label = label_iou if label_iou != 0 else label_overlap
        else:
            label = label_iou
        if best_iou[i] == 0 and best_overlap[i] == 0:
            formatted_proposal_file.write(
                f'0 0 0 {proposals[i][0]} {proposals[i][1]}\n')
        else:
            formatted_proposal_file.write(
                f'{int(label)} {best_iou[i]} {best_overlap[i]} '
                f'{proposals[i][0]} {proposals[i][1]}\n')


def parse_args():
    parser = argparse.ArgumentParser(description='convert proposal format')
    parser.add_argument(
        '--ann-file',
        type=str,
        default='../../../data/ActivityNet/anet_anno_val.json',
        help='name of annotation file')
    parser.add_argument(
        '--activity-index-file',
        type=str,
        default='../../../data/ActivityNet/anet_activity_indexes_val.txt',
        help='name of activity index file')
    parser.add_argument(
        '--proposal-file',
        type=str,
        default='../../../results.json',
        help='name of proposal file, which is the'
        'output of proposal generator (BMN)')
    parser.add_argument(
        '--formatted-proposal-file',
        type=str,
        default='../../../anet_val_formatted_proposal.txt',
        help='name of formatted proposal file, which is the'
        'input of action classifier (SSN)')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    formatted_proposal_file = open(args.formatted_proposal_file, 'w')

    # The activity index file is constructed according to
    # 'https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_classification.py'
    activity_index, class_idx = {}, 0
    for line in open(args.activity_index_file).readlines():
        activity_index[line.strip()] = class_idx
        class_idx += 1

    video_infos = load_annotations(args.ann_file)
    ground_truth = import_ground_truth(video_infos, activity_index)
    proposal, num_proposals = import_proposals(
        mmcv.load(args.proposal_file)['results'])
    video_idx = 0

    for video_info in video_infos:
        video_id = video_info['video_name'][2:]
        num_frames = video_info['duration_frame']
        fps = video_info['fps']
        tiou, t_overlap = pairwise_temporal_iou(
            proposal[video_id][:, :2].astype(float),
            ground_truth[video_id][:, :2].astype(float),
            calculate_overlap_self=True)

        dump_formatted_proposal(video_idx, video_id, num_frames, fps,
                                ground_truth[video_id], proposal[video_id],
                                tiou, t_overlap, formatted_proposal_file)
        video_idx += 1
    formatted_proposal_file.close()
