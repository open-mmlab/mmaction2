import argparse
import glob
import os
import os.path as osp
from multiprocessing import Pool

import cv2


def synthesize_video(frame_dir_item, dev_id=0):
    """Generate videos using opencv-python.

    Args:
        frame_dir_item (list): Rawframe item containing raw frame directory
            full path, rawframe directory (short) path, rawframe directory id.
        dev_id (int): Device id.

    Returns:
        bool: Whether synthesize video successfully.
    """
    full_path, frame_dir_path, frame_dir_id = frame_dir_item
    out_full_path = args.out_dir

    imgs = os.listdir(full_path)
    img_data = cv2.imread(osp.join(full_path, imgs[0]))
    size = (img_data.shape[1], img_data.shape[0])

    if args.ext == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid_name = frame_dir_path + '.mp4'
        out_vid_path = osp.join(out_full_path, out_vid_name)
    else:
        raise TypeError(f'The extension {args.ext} is not supported.')

    videoWrite = cv2.VideoWriter(out_vid_path, fourcc, args.fps, size)

    for img in imgs:
        img_data = cv2.imread(osp.join(full_path, img), 1)
        videoWrite.write(img_data)

    videoWrite.release()
    print(f'{frame_dir_id} {frame_dir_path} done')
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
        '--ext',
        type=str,
        default='mp4',
        choices=['mp4'],
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
        frame_dir_list = list(map(lambda p: osp.basename(p), fullpath_list))

    pool = Pool(args.num_worker)
    pool.map(synthesize_video,
             zip(fullpath_list, frame_dir_list, range(len(frame_dir_list))))
