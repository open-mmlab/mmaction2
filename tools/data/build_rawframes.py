import argparse
import glob
import os
import os.path as osp
import sys
from multiprocessing import Pool


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
        vid_name = vid_path.split('.')[0].split('/')[0]
        out_full_path = osp.join(args.out_dir, vid_name)
    else:
        out_full_path = args.out_dir

    if task == 'rgb':
        if args.new_short == 0:
            cmd = osp.join(
                f'denseflow {full_path} -b=20 -s=0 -o={out_full_path}'
                f' -nw={args.new_width} -nh={args.new_height} -v')
        else:
            cmd = osp.join(
                f'denseflow {full_path} -b=20 -s=0 -o={out_full_path}'
                f' -ns={args.new_short} -v')
        os.system(cmd)
    elif task == 'flow':
        if args.new_short == 0:
            cmd = osp.join(
                f'denseflow {full_path} -a={method} -b=20 -s=1 -o={out_full_path}'  # noqa: E501
                f' -nw={args.new_width} --nh={args.new_height} -v')
        else:
            cmd = osp.join(
                f'denseflow {full_path} -a={method} -b=20 -s=1 -o={out_full_path}'  # noqa: E501
                f' -ns={args.new_short} -v')
        os.system(cmd)
    else:
        if args.new_short == 0:
            cmd_rgb = osp.join(
                f'denseflow {full_path} -b=20 -s=0 -o={out_full_path}'
                f' -nw={args.new_width} -nh={args.new_height} -v')
            cmd_flow = osp.join(
                f'denseflow {full_path} -a={method} -b=20 -s=1 -o={out_full_path}'  # noqa: E501
                f' -nw={args.new_width} -nh={args.new_height} -v')
        else:
            cmd_rgb = osp.join(
                f'denseflow {full_path} -b=20 -s=0 -o={out_full_path}'
                f' -ns={args.new_short} -v')
            cmd_flow = osp.join(
                f'denseflow {full_path} -a={method} -b=20 -s=1 -o={out_full_path}'  # noqa: E501
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
        '--num_worker',
        type=int,
        default=8,
        help='number of workers to build rawframes')
    parser.add_argument(
        '--flow_type',
        type=str,
        default=None,
        choices=[None, 'tvl1', 'warp_tvl1', 'farn', 'brox'],
        help='flow type to be generated')
    parser.add_argument(
        '--out_format',
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
        '--new_width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new_height', type=int, default=0, help='resize image height')
    parser.add_argument(
        '--new_short',
        type=int,
        default=0,
        help='resize image short side length keeping ratio')
    parser.add_argument('--num_gpu', type=int, default=8, help='number of GPU')
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
