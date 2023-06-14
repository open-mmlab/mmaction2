import copy
import os
import pickle
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from mmdet.apis import init_detector
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
from PIL import Image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('root', help='Video folder root')
    parser.add_argument('--pose_config', help='Pose config file')
    parser.add_argument('--pose_ckpt', help='Pose checkpoint file')
    parser.add_argument('--det_config', help='Hand detection config file')
    parser.add_argument('--det_ckpt', help='Hand detection checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


@torch.no_grad()
def inference_topdown(model, pose_pipeline, det_model, det_pipeline, folder):

    img_paths = [f'{folder}/{img}' for img in os.listdir(folder)]

    w, h = Image.open(img_paths[0]).size
    bbox0 = np.array([[0, 0, w, h]], dtype=np.float32)

    imgs = [cv2.imread(img_path) for img_path in img_paths]

    data_list = [
        dict(img=copy.deepcopy(img), img_id=idx)
        for idx, img in enumerate(imgs)
    ]
    data_list = [det_pipeline(data_info) for data_info in data_list]
    batch = pseudo_collate(data_list)
    bbox_results = det_model.test_step(batch)
    bboxes = [i.pred_instances.bboxes[:1].cpu().numpy() for i in bbox_results]
    scores = []
    for i in bbox_results:
        try:
            score = i.pred_instances.scores[0].item()
        except Exception as ex:
            print(ex)
            score = 0
        scores.append(score)
    data_list = []
    for img, bbox, score in zip(imgs, bboxes, scores):
        data_info = dict(img=img)
        if bbox.shape == bbox0.shape and score > 0.3:
            if score > 0.5:
                data_info['bbox'] = bbox
            else:
                w = (score - 0.1) / 0.4
                data_info['bbox'] = w * bbox + (1 - w) * bbox0
        else:
            data_info['bbox'] = bbox0
        data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
        data_info.update(model.dataset_meta)
        data_list.append(pose_pipeline(data_info))

    batch = pseudo_collate(data_list)
    results = model.test_step(batch)

    lookup = {}
    for img_path, result in zip(img_paths, results):
        keypoints = result.pred_instances.keypoints
        scores = result.pred_instances.keypoint_scores
        lookup[img_path] = (keypoints, scores, (w, h))
    return lookup


def main():
    args = parse_args()

    det_model = init_detector(
        args.det_config, args.det_ckpt, device=args.device)
    det_model.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
    det_pipeline = Compose(det_model.cfg.test_dataloader.dataset.pipeline)

    model = init_model(
        args.pose_config, args.pose_checkpoint, device=args.device)
    init_default_scope(model.cfg.get('default_scope', 'mmpose'))

    folders = [f'{args.root}/{folder}' for folder in os.listdir(args.root)]

    pose_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    # inference a single image
    lookup = {}
    L = len(folders)
    t = time.time()
    for idx, folder in enumerate(folders):
        results = inference_topdown(model, pose_pipeline, det_model,
                                    det_pipeline, folder)
        lookup.update(results)
        if idx % 100 == 99:
            eta = (time.time() - t) / (idx + 1) * (L - idx) / 3600
            print('Require %.2f hours' % eta)

    with open('jester.pkl', 'wb') as f:
        pickle.dump(lookup, f)


if __name__ == '__main__':
    main()
