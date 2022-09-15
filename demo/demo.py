# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from operator import itemgetter
from typing import Optional, Tuple

import cv2
import numpy as np
import webcolors
from mmengine import Config, DictAction

from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.utils import register_all_modules


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
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
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
    parser.add_argument('--out-filename', default=None, help='output filename')
    args = parser.parse_args()
    return args


def get_output(video_path: str,
               out_filename: str,
               label: str,
               fps: int = 30,
               font_scale: float = 0.5,
               font_color: str = 'white',
               target_resolution: Optional[Tuple[int]] = None) -> None:
    """Get demo output using ``moviepy``.

    This function will generate video file or gif file from raw video or
    frames, by using ``moviepy``. For more information of some parameters,
    you can refer to: https://github.com/Zulko/moviepy.

    Args:
        video_path (str): The video file path.
        out_filename (str): Output filename for the generated file.
        label (str): Predicted label of the generated file.
        fps (int): Number of picture frames to read per second. Defaults to 30.
        font_scale (float): Font scale of the label. Defaults to 0.5.
        font_color (str): Font color of the label. Defaults to ``white``.
        target_resolution (Tuple[int], optional): Set to
            (desired_width desired_height) to have resized frames. If
            either dimension is None, the frames are resized by keeping
            the existing aspect ratio. Defaults to None.
    """

    if video_path.startswith(('http://', 'https://')):
        raise NotImplementedError

    try:
        import decord
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        raise ImportError('Please install moviepy and decord to '
                          'enable output file.')

    # Channel Order is `BGR`
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

    # Register all modules in mmaction2 into the registries
    register_all_modules()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    result = inference_recognizer(model, args.video)

    pred_scores = result.pred_scores.item.tolist()
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label = score_sorted[:5]

    labels = open(args.label).readlines()
    labels = [x.strip() for x in labels]
    results = [(labels[k[0]], k[1]) for k in top5_label]

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
            target_resolution=args.target_resolution)


if __name__ == '__main__':
    main()
