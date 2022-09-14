# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import shutil
import tempfile

import cv2
import mmengine
import mmcv
import numpy as np
import torch
from mmengine import DictAction

from mmaction.apis import (inference_recognizer, init_recognizer,
                           detection_inference, pose_inference)
from mmaction.utils import frame_extract, register_all_modules

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


# python demo/demo_skeleton.py demo/ntu_sample.avi demo/skeleton_demo.mp4 \
#     --config configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py \
#     --checkpoint https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint/slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth \
#     --det-config demo/faster_rcnn_r50_fpn_2x_coco.py \
#     --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
#     --det-score-thr 0.9 \
#     --pose-config demo/hrnet_w32_coco_256x192.py \
#     --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
#     --label-map tools/data/skeleton/label_map_ntu120.txt

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default=('configs/skeleton/posec3d/'
                 'slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu60_xsub_keypoint/'
                 'slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth'),
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster-rcnn_r50_fpn_2x_coco_infer.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=(
            'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco'
            '/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--pose-config',
        default='demo/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/skeleton/label_map_ntu60.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    frame_paths, original_frames = frame_extract(args.video, args.short_side)
    # frame_paths = frame_paths[:4]

    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # Get Human detection results.
    det_results, _ = detection_inference(args.det_config,
                                         args.det_checkpoint,
                                         frame_paths,
                                         args.det_score_thr,
                                         args.det_cat_id,
                                         args.device)
    torch.cuda.empty_cache()

    # Get Pose estimation results.
    pose_results, pose_data_samples = pose_inference(args.pose_config,
                                                     args.pose_checkpoint,
                                                     frame_paths,
                                                     det_results,
                                                     args.device)
    torch.cuda.empty_cache()

    # Register all modules in mmaction2 into the registries.

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    num_person = max([len(x['keypoints']) for x in pose_results])

    num_keypoint = 17
    keypoint = np.zeros((num_frame, num_person, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_frame, num_person, num_keypoint),
                              dtype=np.float16)
    for i, poses in enumerate(pose_results):
        keypoint[i] = poses['keypoints']
        keypoint_score[i] = poses['keypoint_scores']

    fake_anno['keypoint'] = keypoint.transpose((1, 0, 2, 3))
    fake_anno['keypoint_score'] = keypoint_score.transpose((1, 0, 2))

    register_all_modules()
    config = mmengine.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    if 'data_preprocessor' in config.model:
        config.model.data_preprocessor['mean'] = (w // 2, h // 2, .5)
        config.model.data_preprocessor['std'] = (w, h, 1.)

    model = init_recognizer(config, args.checkpoint, args.device)
    result = inference_recognizer(model, fake_anno)

    max_pred_index = result.pred_scores.item.argmax().item()
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    action_label = label_map[max_pred_index]




    vis_frames = [
        vis_pose_result(pose_model, frame_paths[i], pose_results[i])
        for i in range(num_frame)
    ]

    for frame in vis_frames:
        cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()
