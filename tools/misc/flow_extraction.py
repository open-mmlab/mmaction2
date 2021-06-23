import argparse
import os
import os.path as osp

import cv2
import numpy as np


def flow_to_img(raw_flow, bound=20.):
    """Convert flow to gray image.

    Args:
        raw_flow (np.ndarray[float]): Estimated flow with the shape (w, h).
        bound (float): Bound for the flow-to-image normalization. Default: 20.

    Returns:
        np.ndarray[uint8]: The result list of np.ndarray[uint8], with shape
                        (w, h).
    """
    flow = np.clip(raw_flow, -bound, bound)
    flow += bound
    flow *= (255 / float(2 * bound))
    flow = flow.astype(np.uint8)
    return flow


def generate_flow(frames, method='tvl1'):
    """Estimate flow with given frames.

    Args:
        frames (list[np.ndarray[uint8]]): List of rgb frames, with shape
                                        (w, h, 3).
        method (str): Use which method to generate flow. Options are 'tvl1'
                    and 'farneback'. Default: 'tvl1'.

    Returns:
        list[np.ndarray[float]]: The result list of np.ndarray[float], with
                                shape (w, h, 2).
    """
    assert method in ['tvl1', 'farneback']
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    if method == 'tvl1':
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

        def op(x, y):
            return tvl1.calc(x, y, None)
    elif method == 'farneback':

        def op(x, y):
            return cv2.calcOpticalFlowFarneback(x, y, None, 0.5, 3, 15, 3, 5,
                                                1.2, 0)

    gray_st = gray_frames[:-1]
    gray_ed = gray_frames[1:]

    flow = [op(x, y) for x, y in zip(gray_st, gray_ed)]
    return flow


def extract_dense_flow(path,
                       dest,
                       bound=20.,
                       save_rgb=False,
                       start_idx=0,
                       rgb_tmpl='img_{:05d}.jpg',
                       flow_tmpl='{}_{:05d}.jpg',
                       method='tvl1'):
    """Extract dense flow given video or frames, save them as gray-scale
    images.

    Args:
        path (str): Location of the input video.
        dest (str): The directory to store the extracted flow images.
        bound (float): Bound for the flow-to-image normalization. Default: 20.
        save_rgb (bool): Save extracted RGB frames. Default: False.
        start_idx (int): The starting frame index if use frames as input, the
            first image is path.format(start_idx). Default: 0.
        rgb_tmpl (str): The template of RGB frame names, Default:
            'img_{:05d}.jpg'.
        flow_tmpl (str): The template of Flow frame names, Default:
            '{}_{:05d}.jpg'.
        method (str): Use which method to generate flow. Options are 'tvl1'
            and 'farneback'. Default: 'tvl1'.
    """

    frames = []
    assert osp.exists(path)
    video = cv2.VideoCapture(path)
    flag, f = video.read()
    while flag:
        frames.append(f)
        flag, f = video.read()

    flow = generate_flow(frames, method=method)

    flow_x = [flow_to_img(x[:, :, 0], bound) for x in flow]
    flow_y = [flow_to_img(x[:, :, 1], bound) for x in flow]

    if not osp.exists(dest):
        os.system('mkdir -p ' + dest)
    flow_x_names = [
        osp.join(dest, flow_tmpl.format('x', ind + start_idx))
        for ind in range(len(flow_x))
    ]
    flow_y_names = [
        osp.join(dest, flow_tmpl.format('y', ind + start_idx))
        for ind in range(len(flow_y))
    ]

    num_frames = len(flow)
    for i in range(num_frames):
        cv2.imwrite(flow_x_names[i], flow_x[i])
        cv2.imwrite(flow_y_names[i], flow_y[i])

    if save_rgb:
        img_names = [
            osp.join(dest, rgb_tmpl.format(ind + start_idx))
            for ind in range(len(frames))
        ]
        for frame, name in zip(frames, img_names):
            cv2.imwrite(name, frame)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract flow and RGB images')
    parser.add_argument(
        '--input',
        help='videos for frame extraction, can be'
        'single video or a video list, the video list should be a txt file '
        'and just consists of filenames without directories')
    parser.add_argument(
        '--prefix',
        default='',
        help='the prefix of input '
        'videos, used when input is a video list')
    parser.add_argument(
        '--dest',
        default='',
        help='the destination to save '
        'extracted frames')
    parser.add_argument(
        '--save-rgb', action='store_true', help='also save '
        'rgb frames')
    parser.add_argument(
        '--rgb-tmpl',
        default='img_{:05d}.jpg',
        help='template filename of rgb frames')
    parser.add_argument(
        '--flow-tmpl',
        default='{}_{:05d}.jpg',
        help='template filename of flow frames')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=1,
        help='the start '
        'index of extracted frames')
    parser.add_argument(
        '--method',
        default='tvl1',
        help='use which method to '
        'generate flow')
    parser.add_argument(
        '--bound', type=float, default=20, help='maximum of '
        'optical flow')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.input.endswith('.txt'):
        lines = open(args.input).readlines()
        lines = [x.strip() for x in lines]
        videos = [osp.join(args.prefix, x) for x in lines]
        dests = [osp.join(args.dest, x.split('.')[0]) for x in lines]
        for video, dest in zip(videos, dests):
            extract_dense_flow(video, dest, args.bound, args.save_rgb,
                               args.start_idx, args.rgb_tmpl, args.flow_tmpl,
                               args.method)
    else:
        extract_dense_flow(args.input, args.dest, args.bound, args.save_rgb,
                           args.start_idx, args.rgb_tmpl, args.flow_tmpl,
                           args.method)
