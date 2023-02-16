# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
from multiprocessing import Pool

import mmengine


def extract_audio_wav(line):
    """Extract the audio wave from video streams using FFMPEG."""
    video_id, _ = osp.splitext(osp.basename(line))
    video_dir = osp.dirname(line)
    video_rel_dir = osp.relpath(video_dir, args.root)
    dst_dir = osp.join(args.dst_root, video_rel_dir)
    os.popen(f'mkdir -p {dst_dir}')
    try:
        if osp.exists(f'{dst_dir}/{video_id}.wav'):
            return
        cmd = f'ffmpeg -i ./{line}  -map 0:a  -y {dst_dir}/{video_id}.wav'
        os.popen(cmd)
    except BaseException:
        with open('extract_wav_err_file.txt', 'a+') as f:
            f.write(f'{line}\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Extract audios')
    parser.add_argument('root', type=str, help='source video directory')
    parser.add_argument('dst_root', type=str, help='output audio directory')
    parser.add_argument(
        '--level', type=int, default=2, help='directory level of data')
    parser.add_argument(
        '--ext',
        type=str,
        default='mp4',
        choices=['avi', 'mp4', 'webm'],
        help='video file extensions')
    parser.add_argument(
        '--num-worker', type=int, default=8, help='number of workers')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    mmengine.mkdir_or_exist(args.dst_root)

    print('Reading videos from folder: ', args.root)
    print('Extension of videos: ', args.ext)
    fullpath_list = glob.glob(args.root + '/*' * args.level + '.' + args.ext)
    done_fullpath_list = glob.glob(args.dst_root + '/*' * args.level + '.wav')
    print('Total number of videos found: ', len(fullpath_list))
    print('Total number of videos extracted finished: ',
          len(done_fullpath_list))

    pool = Pool(args.num_worker)
    pool.map(extract_audio_wav, fullpath_list)
