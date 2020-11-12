import argparse
import os
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.utils import GradCAM


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 GradCAM demo')

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


def build_inputs(model, video_path, use_frames=False):
    """build inputs for GradCAM.

    Note that, building inputs for GradCAM is exactly the same as building
    inputs for Recognizer test stage. Codes from `inference_recognizer`.

    Args:
        model (nn.Module): Recognizer model.
        video_path (str): video file/url or rawframes directory.
        use_frames (bool): whether to use rawframes as input.
    Returns:
        dict: Both GradCAM inputs and Recognizer test stage inputs,
            including two keys, ``imgs`` and ``label``.
    """
    if not (osp.exists(video_path) or video_path.startswith('http')):
        raise RuntimeError(f"'{video_path}' is missing")

    if osp.isfile(video_path) and use_frames:
        raise RuntimeError(
            f"'{video_path}' is a video file, not a rawframe directory")
    elif osp.isdir(video_path) and not use_frames:
        raise RuntimeError(
            f"'{video_path}' is a rawframe directory, not a video file")

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    if use_frames:
        filename_tmpl = cfg.data.test.get('filename_tmpl', 'img_{:05}.jpg')
        modality = cfg.data.test.get('modality', 'RGB')
        start_index = cfg.data.test.get('start_index', 1)
        data = dict(
            frame_dir=video_path,
            total_frames=len(os.listdir(video_path)),
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
    else:
        start_index = cfg.data.test.get('start_index', 0)
        data = dict(
            filename=video_path,
            label=-1,
            start_index=start_index,
            modality='RGB')
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    return data


def _resize_frames(frame_list,
                   scale,
                   keep_ratio=True,
                   interpolation='bilinear'):
    """resize frames according to given scale.

    Codes are modified from `mmaction2/datasets/pipelines/augmentation.py`,
    `Resize` class.

    Args:
        frame_list (list[np.ndarray]): frames to be resized.
        scale (tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size: the image will be rescaled as large
            as possible within the scale. Otherwise, it serves as (w, h)
            of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    Returns:
        list[np.ndarray]: Both GradCAM and Recognizer test stage inputs,
            including two keys, ``imgs`` and ``label``.
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

    # assign the desired device.
    device = torch.device(args.device)

    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(
        args.config,
        args.checkpoint,
        device=device,
        use_frames=args.use_frames)

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
