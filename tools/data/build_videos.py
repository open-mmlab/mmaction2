# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import sys
from multiprocessing import Pool


def encode_video(frame_dir_item):
    """Encode frames to video using ffmpeg.

    Args:
        frame_dir_item (list): Rawframe item containing raw frame directory
            full path, rawframe directory (short) path, rawframe directory id.

    Returns:
        bool: Whether synthesize video successfully.
    """
    full_path, frame_dir_path, frame_dir_id = frame_dir_item
    out_full_path = args.out_dir

    img_name_tmpl = args.filename_tmpl + '.' + args.in_format
    img_path = osp.join(full_path, img_name_tmpl)

    out_vid_name = frame_dir_path + '.' + args.ext
    out_vid_path = osp.join(out_full_path, out_vid_name)

    cmd = osp.join(
        f"ffmpeg -start_number {args.start_idx} -r {args.fps} -i '{img_path}' "
        f"-vcodec {args.vcodec} '{out_vid_path}'")
    os.system(cmd)

    print(f'{frame_dir_id} {frame_dir_path} done')
    sys.stdout.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='synthesize videos')
    parser.add_argument('src_dir', type=str, help='source rawframe directory')
    parser.add_argument('out_dir', type=str, help='output video directory')
    parser.add_argument(
        '--fps', type=int, default=30, help='fps of videos to be synthesized')
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
        help='number of workers to build videos')
    parser.add_argument(
        '--in-format',
        type=str,
        default='jpg',
        choices=['jpg', 'png'],
        help='input format')
    parser.add_argument(
        '--start-idx', type=int, default=0, help='starting index of rawframes')
    parser.add_argument(
        '--filename-tmpl',
        type=str,
        default='img_%05d',
        help='filename template of rawframes')
    parser.add_argument(
        '--vcodec', type=str, default='mpeg4', help='coding method of videos')
    parser.add_argument(
        '--ext',
        type=str,
        default='mp4',
        choices=['mp4', 'avi'],
        help='video file extensions')
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

    print('Reading rgb frames from folder: ', args.src_dir)
    print('Input format of rgb frames: ', args.in_format)
    fullpath_list = glob.glob(args.src_dir + '/*' * args.level)
    done_fullpath_list = glob.glob(args.src_dir + '/*' * args.level + '.' +
                                   args.ext)
    print('Total number of rgb frame folders found: ', len(fullpath_list))

    if args.resume:
        fullpath_list = set(fullpath_list).difference(set(done_fullpath_list))
        fullpath_list = list(fullpath_list)
        print('Resuming. number of videos to be synthesized: ',
              len(fullpath_list))

    if args.level == 2:
        frame_dir_list = list(
            map(
                lambda p: osp.join(
                    osp.basename(osp.dirname(p)), osp.basename(p)),
                fullpath_list))
    elif args.level == 1:
        frame_dir_list = list(map(osp.basename, fullpath_list))

    pool = Pool(args.num_worker)
    pool.map(encode_video,
             zip(fullpath_list, frame_dir_list, range(len(frame_dir_list))))
