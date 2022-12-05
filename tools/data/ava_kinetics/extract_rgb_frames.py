# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import multiprocessing
import os


def extract_rgb(video_name, frame_path, video_path):
    video_id = video_name.split('.')[0]
    os.makedirs('%s/%s' % (frame_path, video_id), exist_ok=True)
    cmd = 'ffmpeg -i %s/%s -r 30 -q:v 1 %s/%s' % (video_path, video_name,
                                                  frame_path, video_id)
    cmd += '/img_%05d.jpg'
    return cmd


def run_cmd(cmd):
    os.system(cmd)
    return


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--avakinetics_root',
        type=str,
        default='../../../data/ava_kinetics',
        help='the path to save ava-kinetics dataset')
    p.add_argument(
        '--num_workers',
        type=int,
        default=-1,
        help='number of workers used for multiprocessing')
    args = p.parse_args()

    if args.num_workers > 0:
        num_workers = args.num_workers
    else:
        num_workers = max(multiprocessing.cpu_count() - 1, 1)

    root = args.avakinetics_root
    video_path = root + '/videos/'
    frame_path = root + '/rawframes/'
    os.makedirs(frame_path, exist_ok=True)

    all_cmds = [
        extract_rgb(video_name, frame_path, video_path)
        for video_name in os.listdir(video_path)
    ]

    pool = multiprocessing.Pool(num_workers)
    out = pool.map(run_cmd, all_cmds)
