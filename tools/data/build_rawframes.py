# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import sys
import warnings
from multiprocessing import Lock, Pool

import mmcv
import numpy as np


def extract_frame(vid_item):
    """Generate optical flow using dense flow.

    Args:
        vid_item (list): Video item containing video full path,
            video (short) path, video id.

    Returns:
        bool: Whether generate optical flow successfully.
    """
    full_path, vid_path, vid_id, method, task, report_file = vid_item
    if '/' in vid_path:
        act_name = osp.basename(osp.dirname(vid_path))
        out_full_path = osp.join(args.out_dir, act_name)
    else:
        out_full_path = args.out_dir

    run_success = -1

    if task == 'rgb':
        if args.use_opencv:
            # Not like using denseflow,
            # Use OpenCV will not make a sub directory with the video name
            try:
                video_name = osp.splitext(osp.basename(vid_path))[0]
                out_full_path = osp.join(out_full_path, video_name)

                vr = mmcv.VideoReader(full_path)
                for i, vr_frame in enumerate(vr):
                    if vr_frame is not None:
                        w, h, _ = np.shape(vr_frame)
                        if args.new_short == 0:
                            if args.new_width == 0 or args.new_height == 0:
                                # Keep original shape
                                out_img = vr_frame
                            else:
                                out_img = mmcv.imresize(
                                    vr_frame,
                                    (args.new_width, args.new_height))
                        else:
                            if min(h, w) == h:
                                new_h = args.new_short
                                new_w = int((new_h / h) * w)
                            else:
                                new_w = args.new_short
                                new_h = int((new_w / w) * h)
                            out_img = mmcv.imresize(vr_frame, (new_h, new_w))
                        mmcv.imwrite(out_img,
                                     f'{out_full_path}/img_{i + 1:05d}.jpg')
                    else:
                        warnings.warn(
                            'Length inconsistent!'
                            f'Early stop with {i + 1} out of {len(vr)} frames.'
                        )
                        break
                run_success = 0
            except Exception:
                run_success = -1
        else:
            if args.new_short == 0:
                cmd = osp.join(
                    f"denseflow '{full_path}' -b=20 -s=0 -o='{out_full_path}'"
                    f' -nw={args.new_width} -nh={args.new_height} -v')
            else:
                cmd = osp.join(
                    f"denseflow '{full_path}' -b=20 -s=0 -o='{out_full_path}'"
                    f' -ns={args.new_short} -v')
            run_success = os.system(cmd)
    elif task == 'flow':
        if args.input_frames:
            if args.new_short == 0:
                cmd = osp.join(
                    f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
                    f' -nw={args.new_width} --nh={args.new_height} -v --if')
            else:
                cmd = osp.join(
                    f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
                    f' -ns={args.new_short} -v --if')
        else:
            if args.new_short == 0:
                cmd = osp.join(
                    f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
                    f' -nw={args.new_width} --nh={args.new_height} -v')
            else:
                cmd = osp.join(
                    f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
                    f' -ns={args.new_short} -v')
        run_success = os.system(cmd)
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
        run_success_rgb = os.system(cmd_rgb)
        run_success_flow = os.system(cmd_flow)
        if run_success_flow == 0 and run_success_rgb == 0:
            run_success = 0

    if run_success == 0:
        print(f'{task} {vid_id} {vid_path} {method} done')
        sys.stdout.flush()

        lock.acquire()
        with open(report_file, 'a') as f:
            line = full_path + '\n'
            f.write(line)
        lock.release()
    else:
        print(f'{task} {vid_id} {vid_path} {method} got something wrong')
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
        '--mixed-ext',
        action='store_true',
        help='process video files with mixed extensions')
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
    parser.add_argument(
        '--use-opencv',
        action='store_true',
        help='Whether to use opencv to extract rgb frames')
    parser.add_argument(
        '--input-frames',
        action='store_true',
        help='Whether to extract flow frames based on rgb frames')
    parser.add_argument(
        '--report-file',
        type=str,
        default='build_report.txt',
        help='report to record files which have been successfully processed')
    args = parser.parse_args()

    return args


def init(lock_):
    global lock
    lock = lock_


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

    if args.input_frames:
        print('Reading rgb frames from folder: ', args.src_dir)
        fullpath_list = glob.glob(args.src_dir + '/*' * args.level)
        print('Total number of rgb frame folders found: ', len(fullpath_list))
    else:
        print('Reading videos from folder: ', args.src_dir)
        if args.mixed_ext:
            print('Extension of videos is mixed')
            fullpath_list = glob.glob(args.src_dir + '/*' * args.level)
        else:
            print('Extension of videos: ', args.ext)
            fullpath_list = glob.glob(args.src_dir + '/*' * args.level + '.' +
                                      args.ext)
        print('Total number of videos found: ', len(fullpath_list))

    if args.resume:
        done_fullpath_list = []
        with open(args.report_file) as f:
            for line in f:
                if line == '\n':
                    continue
                done_full_path = line.strip().split()[0]
                done_fullpath_list.append(done_full_path)
        done_fullpath_list = set(done_fullpath_list)
        fullpath_list = list(set(fullpath_list).difference(done_fullpath_list))

    if args.level == 2:
        vid_list = list(
            map(
                lambda p: osp.join(
                    osp.basename(osp.dirname(p)), osp.basename(p)),
                fullpath_list))
    elif args.level == 1:
        vid_list = list(map(osp.basename, fullpath_list))

    lock = Lock()
    pool = Pool(args.num_worker, initializer=init, initargs=(lock, ))
    pool.map(
        extract_frame,
        zip(fullpath_list, vid_list, range(len(vid_list)),
            len(vid_list) * [args.flow_type],
            len(vid_list) * [args.task],
            len(vid_list) * [args.report_file]))
    pool.close()
    pool.join()
