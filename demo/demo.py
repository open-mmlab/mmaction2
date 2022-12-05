# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
import webcolors
from mmcv import Config, DictAction

from mmaction.apis import inference_recognizer, init_recognizer


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('video', help='video file/url or rawframes directory')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--use-frames',
        default=False,
        action='store_true',
        help='whether to use rawframes as input')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--fps',
        default=30,
        type=int,
        help='specify fps value of the output video when using rawframes to '
        'generate file')
    parser.add_argument(
        '--font-scale',
        default=0.5,
        type=float,
        help='font scale of the label in output video')
    parser.add_argument(
        '--font-color',
        default='white',
        help='font color of the label in output video')
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
        default='bicubic',
        help='resize algorithm applied to generate video')
    parser.add_argument('--out-filename', default=None, help='output filename')
    args = parser.parse_args()
    return args


def get_output(video_path,
               out_filename,
               label,
               fps=30,
               font_scale=0.5,
               font_color='white',
               target_resolution=None,
               resize_algorithm='bicubic',
               use_frames=False):
    """Get demo output using ``moviepy``.

    This function will generate video file or gif file from raw video or
    frames, by using ``moviepy``. For more information of some parameters,
    you can refer to: https://github.com/Zulko/moviepy.

    Args:
        video_path (str): The video file path or the rawframes directory path.
            If ``use_frames`` is set to True, it should be rawframes directory
            path. Otherwise, it should be video file path.
        out_filename (str): Output filename for the generated file.
        label (str): Predicted label of the generated file.
        fps (int): Number of picture frames to read per second. Default: 30.
        font_scale (float): Font scale of the label. Default: 0.5.
        font_color (str): Font color of the label. Default: 'white'.
        target_resolution (None | tuple[int | None]): Set to
            (desired_width desired_height) to have resized frames. If either
            dimension is None, the frames are resized by keeping the existing
            aspect ratio. Default: None.
        resize_algorithm (str): Support "bicubic", "bilinear", "neighbor",
            "lanczos", etc. Default: 'bicubic'. For more information,
            see https://ffmpeg.org/ffmpeg-scaler.html
        use_frames: Determine Whether to use rawframes as input. Default:False.
    """

    if video_path.startswith(('http://', 'https://')):
        raise NotImplementedError

    try:
        # In case of a segment fault when import decord in the head of demo
        import decord
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        raise ImportError('Please install moviepy to enable output file.')

    # Channel Order is BGR
    if use_frames:
        frame_list = sorted(
            [osp.join(video_path, x) for x in os.listdir(video_path)])
        frames = [cv2.imread(x) for x in frame_list]
    else:
        video = decord.VideoReader(video_path)
        frames = [x.asnumpy()[..., ::-1] for x in video]

    if target_resolution:
        w, h = target_resolution
        frame_h, frame_w, _ = frames[0].shape
        if w == -1:
            w = int(h / frame_h * frame_w)
        if h == -1:
            h = int(w / frame_w * frame_h)
        frames = [cv2.resize(f, (w, h)) for f in frames]

    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale,
                               1)[0]
    textheight = textsize[1]
    padding = 10
    location = (padding, padding + textheight)

    if isinstance(font_color, str):
        font_color = webcolors.name_to_rgb(font_color)[::-1]

    frames = [np.array(frame) for frame in frames]
    for frame in frames:
        cv2.putText(frame, label, location, cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, font_color, 1)

    # RGB order
    frames = [x[..., ::-1] for x in frames]
    video_clips = ImageSequenceClip(frames, fps=fps)

    out_type = osp.splitext(out_filename)[1][1:]
    if out_type == 'gif':
        video_clips.write_gif(out_filename)
    else:
        video_clips.write_videofile(out_filename, remove_temp=True)


def main():
    args = parse_args()
    # assign the desired device.
    device = torch.device(args.device)

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, args.checkpoint, device=device)

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # test a single video or rawframes of a single video
    if output_layer_names:
        results, returned_feature = inference_recognizer(
            model, args.video, outputs=output_layer_names)
    else:
        results = inference_recognizer(model, args.video)

    labels = open(args.label).readlines()
    labels = [x.strip() for x in labels]
    results = [(labels[k[0]], k[1]) for k in results]

    print('The top-5 labels with corresponding scores are:')
    for result in results:
        print(f'{result[0]}: ', result[1])

    if args.out_filename is not None:

        if args.target_resolution is not None:
            if args.target_resolution[0] == -1:
                assert isinstance(args.target_resolution[1], int)
                assert args.target_resolution[1] > 0
            if args.target_resolution[1] == -1:
                assert isinstance(args.target_resolution[0], int)
                assert args.target_resolution[0] > 0
            args.target_resolution = tuple(args.target_resolution)

        get_output(
            args.video,
            args.out_filename,
            results[0][0],
            fps=args.fps,
            font_scale=args.font_scale,
            font_color=args.font_color,
            target_resolution=args.target_resolution,
            resize_algorithm=args.resize_algorithm,
            use_frames=args.use_frames)


if __name__ == '__main__':
    main()
