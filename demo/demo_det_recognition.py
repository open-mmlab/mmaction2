# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import os
import os.path as osp
import shutil

import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import load_checkpoint

from mmaction.apis import inference_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_detector, build_model, build_recognizer
from mmaction.utils import import_module_error_func

try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import init_pose_model, inference_top_down_pose_model

except (ImportError, ModuleNotFoundError):

    @import_module_error_func('mmdet')
    def inference_detector(*args, **kwargs):
        pass

    @import_module_error_func('mmdet')
    def init_detector(*args, **kwargs):
        pass

    @import_module_error_func('mmpose')
    def init_pose_model(*args, **kwargs):
        pass

    @import_module_error_func('mmpose')
    def inference_top_down_pose_model(*args, **kwargs):
        pass


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = '004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')
plate_green = [hex2color(h) for h in plate_green]


def vis_pose_result(frame,
                    result,
                    skeleton,
                    kpt_score_thr=0.3,
                    pose_kpt_color=None,
                    pose_limb_color=None,
                    radius=4,
                    thickness=1):
    pose_result = []
    for res in result:
        pose_result.append(res['keypoints'])
    img_h, img_w, _ = frame.shape

    for kpts in pose_result:
        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)
            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                if kpt_score > kpt_score_thr:
                    r, g, b = pose_kpt_color[kid]
                    cv2.circle(frame, (int(x_coord), int(y_coord)), radius,
                               (int(r), int(g), int(b)), -1)

        # draw limbs
        if skeleton is not None and pose_limb_color is not None:
            assert len(pose_limb_color) == len(skeleton)
            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
                pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
                if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                        and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
                        and pos2[1] > 0 and pos2[1] < img_h
                        and kpts[sk[0] - 1, 2] > kpt_score_thr
                        and kpts[sk[1] - 1, 2] > kpt_score_thr):
                    r, g, b = pose_limb_color[sk_id]
                    cv2.line(
                        frame,
                        pos1,
                        pos2, (int(r), int(g), int(b)),
                        thickness=thickness)


def visualize(frames,
              annotations,
              pose_results,
              action_result,
              plate=plate_blue,
              max_num=5):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted results.
        plate (str): The plate used for visualization. Default: plate_blue.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.

    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_ = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    print(f'nf--{nf}, na--{na}')
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)  # 2
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])
    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]

            # add action result for whole video
            cv2.putText(frame, action_result, (10, 30), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)

            if j == 0:
                # add pose results
                skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                            [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
                            [4, 6], [5, 7]]
                palette = np.array([[255, 128, 0], [255, 153, 51],
                                    [255, 178, 102], [230, 230, 0],
                                    [255, 153, 255], [153, 204, 255],
                                    [255, 102, 255], [255, 51, 255],
                                    [102, 178, 255], [51, 153, 255],
                                    [255, 153, 153], [255, 102, 102],
                                    [255, 51, 51], [153, 255, 153],
                                    [102, 255, 102], [51, 255, 51],
                                    [0, 255, 0], [0, 0, 255], [255, 0, 0],
                                    [255, 255, 255]])
                pose_limb_color = palette[[
                    0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16,
                    16
                ]]
                pose_kpt_color = palette[[
                    16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
                ]]
                vis_pose_result(frame, pose_results[i], skeleton, 0.3,
                                pose_kpt_color, pose_limb_color)

            # add spatio-temporal action detection results
            for ann in anno:
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, plate[0], 2)

                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
                    text = ': '.join([text, str(score[k])])
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                               THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

    return frames_


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--stdet-config',
        default=('configs/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py'),
        help='spatio temporal detection config file path')
    parser.add_argument(
        '--stdet-checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                 '_20201217-16378594.pth'),
        help='spatio temporal detection checkpoint file/url')

    parser.add_argument(
        '--skeleton-stdet-checkpoint',
        default=('/mnt/lustre21/DATAshare2/duanhaodong/posec3d/'
                 'posec3d_ava.pth'),
        help='skeleton-based spatio temporal detection checkpoint file/url')

    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/'
                 'faster_rcnn/faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--skeleton-config',
        default='configs/skeleton/posec3d/'
        'slowonly_r50_u48_240e_ntu120_xsub_keypoint.py',
        help='skeleton-based action recognition config file path')
    parser.add_argument(
        '--skeleton-checkpoint',
        default='/mnt/lustre21/DATAshare2/duanhaodong/posec3d/'
        'posec3d_k400.pth',
        help='skeleton-based action recognition checkpoint file/url')

    parser.add_argument(
        '--rgb-config',
        default='configs/recognition/tsn/'
        'tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py',
        help='rgb-based action recognition config file path')
    parser.add_argument(
        '--rgb-checkpoint',
        default='https://download.openmmlab.com/mmaction/recognition/'
        'tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/'
        'tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth',
        help='rgb-based action recognition checkpoint file/url')
    parser.add_argument(
        '--use-skeleton-stdet',
        default=True,
        help='use skeleton-based spatio temporal detection method')

    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.6,
        help='the threshold of human detection score')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.35,
        help='the threshold of human action score')
    parser.add_argument(
        '--video',
        default='demo/test_stdet_recognition.mp4',
        help='video file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/ava/label_map.txt',
        help='label map file for spatio-temporal action detection')
    parser.add_argument(
        '--label-map-recognition',
        default='tools/data/kinetics/label_map_k400.txt',
        help='label map file for action recognition')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--out-filename',
        default='demo/test_stdet_recognition_output.mp4',
        help='output filename')
    parser.add_argument(
        '--predict-stepsize',
        default=16,
        type=int,
        help='give out a prediction per n frames')
    parser.add_argument(
        '--output-stepsize',
        default=16,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument(
        '--output-fps',
        default=6,
        type=int,
        help='the fps of demo video output')
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


def frame_extraction(video_path):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    # target_dir = osp.join('./tmp','spatial_skeleton_dir')
    os.makedirs(target_dir, exist_ok=True)
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
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        # print('len-result', len(result))
        # print(result)
        results.append(result)
        prog_bar.update()

    return results


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]

        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret


def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.

    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


def pack_result(human_detection, result, img_h, img_w):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.

    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results


def expand_bbox(bbox, h, w, ratio=2.0):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1
    new_width, new_height = width * ratio, height * ratio
    new_x1 = max(0, int(center_x - new_width / 2))
    new_x2 = min(int(center_x + new_width / 2), w)
    new_y1 = max(0, int(center_y - new_height / 2))
    new_y2 = min(int(center_y + new_height / 2), h)
    return (new_x1, new_y1, new_x2, new_y2)


def isBelong(bbox, area):
    x1, y1, x2, y2 = bbox
    return area[0] <= x1 < x2 <= area[2] and area[1] <= y1 < y2 <= area[3]


def main():
    args = parse_args()

    frame_paths, original_frames = frame_extraction(args.video)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # resize frames to shortside 256
    new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
    print(f'new_w, new_h--{new_w, new_h}, w, h--{w, h}')
    frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
    w_ratio, h_ratio = new_w / w, new_h / h
    print(f'w_ratio, h_ratio--{w_ratio, h_ratio}')

    # Get clip_len, frame_interval and calculate center index of each clip
    stdet_config = mmcv.Config.fromfile(args.stdet_config)
    stdet_config.merge_from_dict(args.cfg_options)
    val_pipeline = stdet_config.data.val.pipeline

    sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    print(f'clip_len, frame_interval--{clip_len, frame_interval}')

    window_size = clip_len * frame_interval  # 64
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    # Note that it's 1 based here
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)
    print(f'timestamps--{timestamps}')  # len=30

    # Load spatio-temporal detection label_map
    label_map = load_label_map(args.label_map)
    try:
        if stdet_config['data']['train']['custom_classes'] is not None:
            label_map = {
                id + 1: label_map[cls]
                for id, cls in enumerate(stdet_config['data']['train']
                                         ['custom_classes'])
            }
    except KeyError:
        pass

    # Get Human detection results
    center_frames = [frame_paths[ind - 1] for ind in timestamps]
    human_detections = detection_inference(args, center_frames)
    # for x in human_detections[:2]:
    #     print(x) # num_person X 5

    # print('------------')
    # ss

    pose_results = pose_inference(args, center_frames, human_detections)
    print(f'len(pose_res)--{len(pose_results)}')  # 30
    # for x in pose_results[:2]:
    #     print('pose--',len(x)) # num_person
    #     for person in x:
    #         print(person['bbox'])

    # ss

    fake_anno = dict(
        frame_dict='',
        label=-1,
        img_shape=(h, w),
        origin_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    num_person = max([len(x) for x in pose_results])

    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)  # M T V 2
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)  # M T V
    for i, poses in enumerate(pose_results):
        # print('len-pose---', len(poses))  # 7
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, i] = pose[:, :2]
            keypoint_score[j, i] = pose[:, 2]
            # print('keypoint--',pose[:, :2])
            # print('keypoint-score--', pose[:,2])

    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    skeleton_config = mmcv.Config.fromfile(args.skeleton_config)
    skeleton_config.model.cls_head.num_classes = 400  # for K400 dataset
    skeleton_pipeline = Compose(skeleton_config.test_pipeline)
    skeleton_imgs = skeleton_pipeline(fake_anno)['imgs'][None]
    skeleton_imgs = skeleton_imgs.to(args.device)
    print('skeleton-imgs-input', skeleton_imgs.shape)  # 1 20 17 48 64 64

    # Build skeleton-based recognition model
    skeleton_model = build_model(skeleton_config.model)
    load_checkpoint(
        skeleton_model, args.skeleton_checkpoint, map_location=args.device)
    skeleton_model.to(args.device)
    skeleton_model.eval()

    with torch.no_grad():
        output = skeleton_model(return_loss=False, imgs=skeleton_imgs)

    label_map_recognition = [
        x.strip() for x in open(args.label_map_recognition).readlines()
    ]
    action_idx = np.argmax(output)
    skeleton_action_result = label_map_recognition[
        action_idx]  # skeleton-based action result for the whole video
    print(f'skeleton_action_result--{skeleton_action_result}')

    # Build rgb-based recognition model
    rgb_config = mmcv.Config.fromfile(args.rgb_config)
    rgb_config.model.backbone.pretrained = None
    rgb_model = build_recognizer(
        rgb_config.model, test_cfg=rgb_config.get('test_cfg'))
    load_checkpoint(rgb_model, args.rgb_checkpoint, map_location=args.device)
    rgb_model.cfg = rgb_config
    rgb_model.to(args.device)
    rgb_model.eval()
    action_results = inference_recognizer(rgb_model, args.video,
                                          args.label_map_recognition)
    rgb_action_result = action_results[0][
        0]  # rgb-based action result for the whole video
    print(f'rgb_results--{rgb_action_result}')

    print('Performing skeleton-based SpatioTemporal Action '
          'Detection for each clip')
    skeleton_config.model.cls_head.num_classes = 81  # for AVA dataset
    skeleton_stdet_model = build_model(skeleton_config.model)
    load_checkpoint(
        skeleton_stdet_model,
        args.skeleton_stdet_checkpoint,
        map_location=args.device)
    skeleton_stdet_model.to(args.device)
    skeleton_stdet_model.eval()

    prog_bar = mmcv.ProgressBar(len(timestamps))
    skeleton_predictions = []
    for timestamp, proposal in zip(timestamps, human_detections):  # frame
        if proposal.shape[0] == 0:  # no people detected
            skeleton_predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        print('frame_inds', frame_inds)  # [7, 15, 23, 31, 39, 47, 55, 63]
        num_frame = len(frame_inds)  # 8
        stdet_frames = [frame_paths[ind] for ind in frame_inds]

        human_detection = detection_inference(args, stdet_frames)
        pose_result = pose_inference(args, stdet_frames, human_detection)

        num_person = max([len(x) for x in pose_result])
        # for x in pose_result:
        #     print(len(x),x) # len(x)=7 num_person个dict(bbox, keypoints)
        #     ss

        skeleton_prediction = []
        for i in range(proposal.shape[0]):  # num_person
            skeleton_prediction.append([])

        for i in range(proposal.shape[0]):  # num_person
            fake_anno = dict(
                frame_dict='',
                label=-1,
                img_shape=(h, w),
                origin_shape=(h, w),
                start_index=0,
                modality='Pose',
                total_frames=num_frame)
            num_person = 1

            num_keypoint = 17
            keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                                dtype=np.float16)  # M T V 2
            keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                      dtype=np.float16)  # M T V

            # pose matching
            person_bbox = proposal[i][:4]  # x1, y1, x2, y2
            area = expand_bbox(person_bbox, h, w)

            for j, poses in enumerate(pose_result):  # num_frame
                for per_pose in poses:  # num_person
                    if isBelong(per_pose['bbox'][:4], area):
                        keypoint[0, j] = per_pose['keypoints'][:, :2]
                        keypoint_score[0, j] = per_pose['keypoints'][:, 2]

            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score

            skeleton_imgs = skeleton_pipeline(fake_anno)['imgs'][None]
            skeleton_imgs = skeleton_imgs.to(args.device)

            with torch.no_grad():
                output = skeleton_stdet_model(
                    return_loss=False, imgs=skeleton_imgs)
            action_idx = np.argmax(output)

            if output[0, action_idx] > args.action_score_thr:
                skeleton_prediction[i].append(
                    (label_map[action_idx], output[0, action_idx]))

            # print('output_idx', action_idx)
            print('action_score--', output[0, action_idx])
            skeleton_stdet_result = label_map[action_idx]
            print(f'skeleton_stdet result--{skeleton_stdet_result}')

        skeleton_predictions.append(skeleton_prediction)

    prog_bar.update()

    for i in range(len(human_detections)):
        det = human_detections[i]
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)

    # Get img_norm_cfg
    img_norm_cfg = stdet_config['img_norm_cfg']
    if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
        to_bgr = img_norm_cfg.pop('to_bgr')
        img_norm_cfg['to_rgb'] = to_bgr
    img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
    img_norm_cfg['std'] = np.array(img_norm_cfg['std'])

    # Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        stdet_config['model']['test_cfg']['rcnn']['action_thr'] = .0
    except KeyError:
        pass

    stdet_config.model.backbone.pretrained = None
    stdet_model = build_detector(
        stdet_config.model, test_cfg=stdet_config.get('test_cfg'))

    load_checkpoint(
        stdet_model, args.stdet_checkpoint, map_location=args.device)
    stdet_model.to(args.device)
    stdet_model.eval()

    predictions = []  # 30 7(num_person) 2

    print('Performing SpatioTemporal Action Detection for each clip')
    assert len(timestamps) == len(human_detections)
    prog_bar = mmcv.ProgressBar(len(timestamps))
    for timestamp, proposal in zip(timestamps, human_detections):  # 30
        if proposal.shape[0] == 0:
            predictions.append(None)
            continue
        # print('proposal', proposal.shape) # num_person(7) 4

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        print('frame_inds', frame_inds)  # [7, 15, 23, 31, 39, 47, 55, 63]
        imgs = [frames[ind].astype(np.float32) for ind in frame_inds]
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(args.device)
        print('input_tensor', input_tensor.shape)  # 1 3 8 256 455

        with torch.no_grad():
            result = stdet_model(
                return_loss=False,
                img=[input_tensor],
                img_metas=[[dict(img_shape=(new_h, new_w))]],
                proposals=[[proposal]])
            result = result[0]
            prediction = []
            # N proposals
            for i in range(proposal.shape[0]):  # 7 num_person
                prediction.append([])
            # print('stdet-res--', len(result)) # 80 (num_classes)
            # for x in result:
            #     print(x.shape)  # 7(num_person) 5

            # Perform action score thr
            for i in range(len(result)):
                if i + 1 not in label_map:
                    continue
                for j in range(proposal.shape[0]):
                    if result[i][j, 4] > args.action_score_thr:
                        prediction[j].append((label_map[i + 1], result[i][j,
                                                                          4]))

            predictions.append(prediction)
        prog_bar.update()

    if args.use_skeleton_stdet:
        print('Use skeleton-based stdet')
        predictions = skeleton_predictions
    results = []
    for human_detection, prediction in zip(human_detections, predictions):
        results.append(pack_result(human_detection, prediction, new_h, new_w))

    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])  # 8
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int)

    dense_n = int(args.predict_stepsize / args.output_stepsize)  # 2
    frames = [
        cv2.imread(frame_paths[i - 1])
        for i in dense_timestamps(timestamps, dense_n)
    ]
    print('Performing visualization')
    vis_frames = visualize(frames, results, pose_results,
                           skeleton_action_result)
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                fps=args.output_fps)
    vid.write_videofile(args.out_filename)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()
