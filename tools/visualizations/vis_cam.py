# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import mmcv
import numpy as np
import torch.nn as nn
from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate

from mmaction.apis import init_recognizer
from mmaction.utils import GradCAM


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 GradCAM Visualization')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('video', help='video file/url or rawframes directory')
    parser.add_argument(
        '--use-frames',
        default=False,
        action='store_true',
        help='whether to use rawframes as input')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--target-layer-name',
        type=str,
        default='backbone/layer4/1/relu',
        help='GradCAM target layer name')
    parser.add_argument('--out-filename', default=None, help='output filename')
    parser.add_argument('--fps', default=5, type=int)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--target-resolution',
        nargs=2,
        default=None,
        type=int,
        help='Target resolution (w, h) for resizing the frames when using a '
        'video as input. If either dimension is set to -1, the frames are '
        'resized by keeping the existing aspect ratio')
    parser.add_argument(
        '--resize-algorithm',
        default='bilinear',
        help='resize algorithm applied to generate video & gif')

    args = parser.parse_args()
    return args


def build_inputs(model: nn.Module,
                 video_path: str,
                 use_frames: bool = False) -> Dict:
    """build inputs for GradCAM.

    Note that, building inputs for GradCAM is exactly the same as building
    inputs for Recognizer test stage. Codes from `inference_recognizer`.

    Args:
        model (nn.Module): Recognizer model.
        video_path (str): video file/url or rawframes directory.
        use_frames (bool): whether to use rawframes as input.
            Defaults to False.

    Returns:
        dict: Both GradCAM inputs and Recognizer test stage inputs,
        including two keys, ``inputs`` and ``data_samples``.
    """
    if not (osp.exists(video_path) or video_path.startswith('http')):
        raise RuntimeError(f"'{video_path}' is missing")

    if osp.isfile(video_path) and use_frames:
        raise RuntimeError(
            f"'{video_path}' is a video file, not a rawframe directory")
    if osp.isdir(video_path) and not use_frames:
        raise RuntimeError(
            f"'{video_path}' is a rawframe directory, not a video file")

    cfg = model.cfg

    # build the data pipeline
    test_pipeline = cfg.test_pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    if use_frames:
        filename_tmpl = cfg.test_dataloader.dataset.get(
            'filename_tmpl', 'img_{:05}.jpg')
        start_index = cfg.test_dataloader.dataset.get('start_index', 1)
        data = dict(
            frame_dir=video_path,
            total_frames=len(os.listdir(video_path)),
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality='RGB')
    else:
        start_index = cfg.test_dataloader.dataset.get('start_index', 0)
        data = dict(
            filename=video_path,
            label=-1,
            start_index=start_index,
            modality='RGB')
    data = test_pipeline(data)
    data = pseudo_collate([data])

    return data


def _resize_frames(frame_list: List[np.ndarray],
                   scale: Optional[Tuple[int]] = None,
                   keep_ratio: bool = True,
                   interpolation: str = 'bilinear') -> List[np.ndarray]:
    """Resize frames according to given scale.

    Codes are modified from `mmaction/datasets/transforms/processing.py`,
    `Resize` class.

    Args:
        frame_list (list[np.ndarray]): Frames to be resized.
        scale (tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size: the image will be rescaled as large
            as possible within the scale. Otherwise, it serves as (w, h)
            of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Defaults to True.
        interpolation (str): Algorithm used for interpolation:
            'nearest' | 'bilinear'. Defaults to ``'bilinear'``.

    Returns:
        list[np.ndarray]: Resized frames.
    """
    if scale is None or (scale[0] == -1 and scale[1] == -1):
        return frame_list
    scale = tuple(scale)
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    if max_short_edge == -1:
        scale = (np.inf, max_long_edge)

    img_h, img_w, _ = frame_list[0].shape

    if keep_ratio:
        new_w, new_h = mmcv.rescale_size((img_w, img_h), scale)
    else:
        new_w, new_h = scale

    frame_list = [
        mmcv.imresize(img, (new_w, new_h), interpolation=interpolation)
        for img in frame_list
    ]

    return frame_list


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    # Build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, args.checkpoint, device=args.device)

    inputs = build_inputs(model, args.video, use_frames=args.use_frames)
    gradcam = GradCAM(model, args.target_layer_name)
    results = gradcam(inputs)

    if args.out_filename is not None:
        try:
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            raise ImportError('Please install moviepy to enable output file.')

        # frames_batches shape [B, T, H, W, 3], in RGB order
        frames_batches = (results[0] * 255.).numpy().astype(np.uint8)
        frames = frames_batches.reshape(-1, *frames_batches.shape[-3:])

        frame_list = list(frames)
        frame_list = _resize_frames(
            frame_list,
            args.target_resolution,
            interpolation=args.resize_algorithm)

        video_clips = ImageSequenceClip(frame_list, fps=args.fps)
        out_type = osp.splitext(args.out_filename)[1][1:]
        if out_type == 'gif':
            video_clips.write_gif(args.out_filename)
        else:
            video_clips.write_videofile(args.out_filename, remove_temp=True)


if __name__ == '__main__':
    main()
