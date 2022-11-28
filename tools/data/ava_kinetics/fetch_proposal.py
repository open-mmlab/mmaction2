# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import multiprocessing as mp
import os

import decord
import numpy as np
import pickle

from mmdet.utils import register_all_modules
from mmdet.apis import inference_detector, init_detector


def get_vid_from_path(path):
    video_id = path.split('/')[-1].split('.')[0]
    video_id = video_id.split('_')[:-2]
    return '_'.join(video_id)


def prepare_det_lookup(datalist):
    with open(datalist) as f:
        records = f.readlines()
    det_lookup = {}
    for record in records:
        record = record.split(',')
        video_path = record[0]
        video_id = get_vid_from_path(video_path)
        frame_id = int(record[1])
        for idx in range(frame_id-1, frame_id+2):
            proposal_id = '%s,%04d' % (video_id, idx)
            det_lookup[proposal_id] = video_path
    return det_lookup


def single_worker(rank, det_lookup, args):
    detect_list = list(det_lookup)
    detect_sublist = [detect_list[i] for i in range(len(detect_list)) 
                      if i % args.num_gpus == rank]

    model = init_detector(args.config, 
                          args.checkpoint, 
                          device='cuda:%d' % rank)

    lookup = {}
    for key in enumerate(detect_sublist):
        video_path = det_lookup[key]
        start = int(video_path.split('/')[-1].split('_')[-2])
        time = int(key.split(',')[1])
        frame_id = (time - start) * 30 + 1
        frame = decord.VideoReader(video_path).get_batch([frame_id]).asnumpy()[0]
        H, W, _ = frame    
        try:
            video_path = det_lookup[key]
            result = inference_detector(model, frame)
            bboxes = result._pred_instances.bboxes.cpu()
            scores = result._pred_instances.scores.cpu()
            labels = result._pred_instances.labels.cpu()

            bboxes = bboxes[labels == 0]
            scores = scores[labels == 0]

            bboxes = bboxes[scores > 0.7].numpy()
            scores = scores[scores > 0.7]
            if scores.numel() > 0:
                result_ = []
                for idx, (h1, w1, h2, w2) in result:
                    h1 /= H
                    h2 /= H
                    w1 /= W
                    w2 /= W
                    score = scores[idx].item()
                    result_.append((h1, w1, h2, w2, score))
                lookup[key] = np.array(result_)
        except:  # noqa: E722
            pass

    with open('tmp_person_%d.pkl' % rank, 'wb') as f:
        pickle.dump(lookup, f)
    return


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--datalist', type=str, 
                   default='../../../data/ava_kinetics/kinetics_train.csv',
                   help='the list for kinetics videos')
    p.add_argument('--config', type=str,
                   default='X-101-64x4d-FPN.py',
                   help='the human detector')
    p.add_argument('--checkpoint', type=str,
                   default='https://download.openmmlab.com/mmdetection/v2.0/'
                           'cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco/'
                           'cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_'
                           '075702-43ce6a30.pth',
                   help='the human detector checkpoint')
    p.add_argument('--picklepath', type=str,
                   default='../../../data/ava_kinetics/kinetics_proposal.pkl')
    p.add_argument('--num_gpus', type=int, default=8)

    args = p.parse_args()

    # register all modules in mmdet into the registries
    register_all_modules()

    det_lookup = prepare_det_lookup(args.datalist)

    processes = []
    for rank in range(args.num_gpus):
        ctx = mp.get_context('spawn')
        p = ctx.Process(target=single_worker, args=(rank, det_lookup, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    lookup = {}
    for k in range(args.num_gpus):
        one_lookup = pickle.load(open('tmp_person_%d.pkl' % k, 'rb'))
        os.remove('tmp_person_%d.pkl' % k)
        for key in one_lookup:
            lookup[key] = one_lookup[key]

    with open(args.picklepath, 'wb') as f:
        pickle.dump(lookup, f)
