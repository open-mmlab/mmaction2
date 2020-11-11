import argparse
import os
import os.path as osp

import torch

from mmaction.apis import inference_recognizer, init_recognizer


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('video', help='video file/url or rawframes directory')
    parser.add_argument('label', help='label file')
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
        '--font-size',
        default=20,
        type=int,
        help='font size of the label test in output video')
    parser.add_argument(
        '--font-color',
        default='white',
        help='font color of the label test in output video')
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
               font_size=20,
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
        font_size (int): Font size of the label. Default: 20.
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
        from moviepy.editor import (CompositeVideoClip, ImageSequenceClip,
                                    TextClip, VideoFileClip)
    except ImportError:
        raise ImportError('Please install moviepy to enable output file.')

    if use_frames:
        frame_list = sorted(
            [osp.join(video_path, x) for x in os.listdir(video_path)])
        video_clips = ImageSequenceClip(frame_list, fps=fps)
    else:
        # revert the order to suit ``VideoFileClip``.
        # (weight, height) -> (height, weight)
        target_resolution = (target_resolution[1], target_resolution[0])
        video_clips = VideoFileClip(
            video_path,
            target_resolution=target_resolution,
            resize_algorithm=resize_algorithm)

    duration_video_clip = video_clips.duration
    text_clips = TextClip(label, fontsize=font_size, color=font_color)
    text_clips = (
        text_clips.set_position(
            ('right', 'bottom'),
            relative=True).set_duration(duration_video_clip))

    video_clips = CompositeVideoClip([video_clips, text_clips])

    out_type = osp.splitext(out_filename)[1][1:]
    if out_type == 'gif':
        video_clips.write_gif(out_filename)
    else:
        video_clips.write_videofile(out_filename, remove_temp=True)


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
    # test a single video or rawframes of a single video
    results = inference_recognizer(
        model, args.video, args.label, use_frames=args.use_frames)

    print('The top-5 labels with corresponding scores are:')
    for result in results:
        print(f'{result[0]}: ', result[1])

    if args.out_filename is not None:

        if args.target_resolution is not None:
            if args.target_resolution[0] == -1:
                args.target_resolution[0] = None
            if args.target_resolution[1] == -1:
                args.target_resolution[1] = None
            args.target_resolution = tuple(args.target_resolution)
        else:
            args.target_resolution = (None, None)

        get_output(
            args.video,
            args.out_filename,
            results[0][0],
            fps=args.fps,
            font_size=args.font_size,
            font_color=args.font_color,
            target_resolution=args.target_resolution,
            resize_algorithm=args.resize_algorithm,
            use_frames=args.use_frames)


if __name__ == '__main__':
    main()
