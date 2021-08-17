# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import subprocess

import mmcv
from joblib import Parallel, delayed

URL_PREFIX = 'https://s3.amazonaws.com/ava-dataset/trainval/'


def download_video(video_url, output_dir, num_attempts=5):
    video_file = osp.basename(video_url)
    output_file = osp.join(output_dir, video_file)

    status = False

    if not osp.exists(output_file):
        command = ['wget', '-c', video_url, '-P', output_dir]
        command = ' '.join(command)
        print(command)
        attempts = 0
        while True:
            try:
                subprocess.check_output(
                    command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                attempts += 1
                if attempts == num_attempts:
                    return status, 'Downloading Failed'
            else:
                break

    status = osp.exists(output_file)
    return status, 'Downloaded'


def main(source_file, output_dir, num_jobs=24, num_attempts=5):
    mmcv.mkdir_or_exist(output_dir)
    video_list = open(source_file).read().strip().split('\n')
    video_list = [osp.join(URL_PREFIX, video) for video in video_list]

    if num_jobs == 1:
        status_list = []
        for video in video_list:
            video_list.append(download_video(video, output_dir, num_attempts))
    else:
        status_list = Parallel(n_jobs=num_jobs)(
            delayed(download_video)(video, output_dir, num_attempts)
            for video in video_list)

    mmcv.dump(status_list, 'download_report.json')


if __name__ == '__main__':
    description = 'Helper script for downloading AVA videos'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'source_file', type=str, help='TXT file containing the video filename')
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory where videos will be saved')
    parser.add_argument('-n', '--num-jobs', type=int, default=24)
    parser.add_argument('--num-attempts', type=int, default=5)
    main(**vars(parser.parse_args()))
