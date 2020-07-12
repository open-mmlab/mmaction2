import argparse
import glob
import os
import os.path as osp
import sys
import warnings
from multiprocessing import Pool

import mmcv
import numpy as np


def extract_frame(vid_item, dev_id=0):
    """Generate optical flow using dense flow.

    Args:
        vid_item (list): Video item containing video full path,
            video (short) path, video id.
        dev_id (int): Device id.

    Returns:
        bool: Whether generate optical flow successfully.
    """
    full_path, vid_path, vid_id, method, task = vid_item
    if ('/' in vid_path):
        act_name = osp.splitext(vid_path)[0].split('/')[0]
        out_full_path = osp.join(args.out_dir, act_name)
    else:
        out_full_path = args.out_dir

    if task == 'rgb':
        if args.use_opencv:
            # Not like using denseflow,
            # Use OpenCV will not make a sub directory with the video name
            video_name = osp.splitext(osp.basename(vid_path))[0]
            out_full_path = osp.join(out_full_path, video_name)

            vr = mmcv.VideoReader(full_path)
            for i in range(len(vr)):
                if vr[i] is not None:
                    w, h, c = np.shape(vr[i])
                    if args.new_short == 0:
                        out_img = mmcv.imresize(vr[i], (args.new_width,
                                                        args.new_height))
                    else:
                        if min(h, w) == h:
                            new_h = args.new_short
                            new_w = int((new_h / h) * w)
                        else:
                            new_w = args.new_short
                            new_h = int((new_w / w) * h)
                        out_img = mmcv.imresize(vr[i], (new_h, new_w))
                    mmcv.imwrite(out_img,
                                 f'{out_full_path}/img_{i + 1:05d}.jpg')
                else:
                    warnings.warn(
                        'Length inconsistent!'
                        f'Early stop with {i + 1} out of {len(vr)} frames.')
                    break
        else:
            if args.new_short == 0:
                cmd = osp.join(
                    f"denseflow '{full_path}' -b=20 -s=0 -o='{out_full_path}'"
                    f' -nw={args.new_width} -nh={args.new_height} -v')
            else:
                cmd = osp.join(
                    f"denseflow '{full_path}' -b=20 -s=0 -o='{out_full_path}'"
                    f' -ns={args.new_short} -v')
            os.system(cmd)
    elif task == 'flow':
        if args.new_short == 0:
            cmd = osp.join(
                f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
                f' -nw={args.new_width} --nh={args.new_height} -v')
        else:
            cmd = osp.join(
                f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
                f' -ns={args.new_short} -v')
        os.system(cmd)
    else:
        if args.new_short == 0:
            cmd_rgb = osp.join(
                f"denseflow '{full_path}' -b=20 -s=0 -o='{out_full_path}'"
                f' -nw={args.new_width} -nh={args.new_height} -v')
            cmd_flow = osp.join(
                f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
                f' -nw={args.new_width} -nh={args.new_height} -v')
        else:
            cmd_rgb = osp.join(
                f"denseflow '{full_path}' -b=20 -s=0 -o='{out_full_path}'"
                f' -ns={args.new_short} -v')
            cmd_flow = osp.join(
                f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
                f' -ns={args.new_short} -v')
        os.system(cmd_rgb)
        os.system(cmd_flow)

    print(f'{task} {vid_id} {vid_path} {method} done')
    sys.stdout.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--task',
        type=str,
        default='flow',
        choices=['rgb', 'flow', 'both'],
        help='which type of frames to be extracted')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2],
        default=2,
        help='directory level of data')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='number of workers to build rawframes')
    parser.add_argument(
        '--flow-type',
        type=str,
        default=None,
        choices=[None, 'tvl1', 'warp_tvl1', 'farn', 'brox'],
        help='flow type to be generated')
    parser.add_argument(
        '--out-format',
        type=str,
        default='jpg',
        choices=['jpg', 'h5', 'png'],
        help='output format')
    parser.add_argument(
        '--ext',
        type=str,
        default='avi',
        choices=['avi', 'mp4', 'webm'],
        help='video file extensions')
    parser.add_argument(
        '--use-opencv',
        action='store_true',
        help='Whether to use opencv to extract rgb frames')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    parser.add_argument(
        '--new-short',
        type=int,
        default=0,
        help='resize image short side length keeping ratio')
    parser.add_argument('--num-gpu', type=int, default=8, help='number of GPU')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume optical flow extraction instead of overwriting')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    if args.level == 2:
        classes = os.listdir(args.src_dir)
        for classname in classes:
            new_dir = osp.join(args.out_dir, classname)
            if not osp.isdir(new_dir):
                print(f'Creating folder: {new_dir}')
                os.makedirs(new_dir)

    print('Reading videos from folder: ', args.src_dir)
    print('Extension of videos: ', args.ext)

    fullpath_list = glob.glob(args.src_dir + '/*' * args.level + '.' +
                              args.ext)
    done_fullpath_list = glob.glob(args.out_dir + '/*' * args.level)
    print('Total number of videos found: ', len(fullpath_list))

    if args.resume:
        fullpath_list = set(fullpath_list).difference(set(done_fullpath_list))
        fullpath_list = list(fullpath_list)
        print('Resuming. number of videos to be done: ', len(fullpath_list))

    if args.level == 2:
        vid_list = list(
            map(lambda p: osp.join('/'.join(p.split('/')[-2:])),
                fullpath_list))
    elif args.level == 1:
        vid_list = list(map(lambda p: p.split('/')[-1], fullpath_list))

    pool = Pool(args.num_worker)
    pool.map(
        extract_frame,
        zip(fullpath_list, vid_list, range(len(vid_list)),
            len(vid_list) * [args.flow_type],
            len(vid_list) * [args.task]))
