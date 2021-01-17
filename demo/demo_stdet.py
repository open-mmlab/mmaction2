import argparse
import os.path as osp
import warnings

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint

from mmaction.models import build_detector

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    warnings.warn('Please install mmdet to use stdet_demo')


# We only accept videos with regular fps (30 fps) now.
def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--config',
        default=('configs/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py'),
        help='spatio temporal detection config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                 '_20201217-16378594.pth'),
        help='spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument('--video', help='video file/url')
    parser.add_argument('--label', help='label file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--out-filename', default='stdet_demo.mp4', help='output filename')
    parser.add_argument(
        '--predict-stepsize',
        default=16,
        help='give out a prediction per n frames')
    args = parser.parse_args()
    return args


def frame_extraction(video_path):
    # Load the video, extract frames into /tmp/video_name
    target_dir = osp.join('/tmp', osp.basename(osp.splitext(video_path)[0]))
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()
    return frame_paths, frames


def detection_inference(args, frame_paths):
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][4] >= args.det_score_thr]
        results.append(result)
    return results


def main():
    args = parse_args()

    frame_paths, frames = frame_extraction(args.video)
    human_detections = detection_inference(args, frame_paths)
    num_frame = len(frames)
    h, w, _ = frames[0].shape

    # All images will be resized to shortedge 256
    new_w, new_h = mmcv.rescale_size((w, h), (-1, 256))
    frames = [mmcv.imresize(img, (new_w, new_h)) for img in frames]
    w_ratio, h_ratio = new_w / w, new_h / h
    for i in range(num_frame):
        det_result = human_detections[i]
        det_result[:, 0:4:2] *= w_ratio
        det_result[:, 1:4:2] *= h_ratio

    # build the spatio temporal action detection from a config file and
    # checkpoint file/url
    # We will only use the 'model' part of the config file
    config = mmcv.Config.fromfile(args.config)
    val_pipeline = config['val_pipeline']
    sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    img_norm_cfg = config['img_norm_cfg']

    config.model.backbone.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)

    load_checkpoint(model, args.checkpoint, map_location=args.device)
    model.to(args.device)
    model.eval()

    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    for timestamp in timestamps:
        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        imgs = [frames[ind] for ind in frame_inds]
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        # 1, C, T, H, W
        input_array = np.stack(imgs).transpose((1, 0, 2, 3))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(args.device)
        input_tensor *= 2.
        # pass


if __name__ == '__main__':
    main()
