import argparse
import fnmatch
import glob
import os
import os.path as osp

import mmcv

annotation_root = '../../data/hvu/annotations'
tag_file = 'hvu_tags.json'
args = None


def parse_directory(path,
                    rgb_prefix='img_',
                    flow_x_prefix='flow_x_',
                    flow_y_prefix='flow_y_',
                    level=1):
    """Parse directories holding extracted frames from standard benchmarks.

    Args:
        path (str): Directory path to parse frames.
        rgb_prefix (str): Prefix of generated rgb frames name.
            default: 'img_'.
        flow_x_prefix (str): Prefix of generated flow x name.
            default: `flow_x_`.
        flow_y_prefix (str): Prefix of generated flow y name.
            default: `flow_y_`.
        level (int): Directory level for glob searching. Options are 1 and 2.
            default: 1.

    Returns:
        dict: frame info dict with video id as key and tuple(path(str),
            rgb_num(int), flow_x_num(int)) as value.
    """
    print(f'parse frames under directory {path}')
    if level == 1:
        # Only search for one-level directory
        def locate_directory(x):
            return osp.basename(x)

        frame_dirs = glob.glob(osp.join(path, '*'))

    elif level == 2:
        # search for two-level directory
        def locate_directory(x):
            return osp.join(osp.basename(osp.dirname(x)), osp.basename(x))

        frame_dirs = glob.glob(osp.join(path, '*', '*'))

    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory, prefix_list):
        """Count file number with a given directory and prefix.

        Args:
            directory (str): Data directory to be search.
            prefix_list (list): List or prefix.

        Returns:
            list (int): Number list of the file with the prefix.
        """
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x + '*')) for x in prefix_list]
        return cnt_list

    # check RGB
    frame_dict = {}
    for i, frame_dir in enumerate(frame_dirs):
        total_num = count_files(frame_dir,
                                (rgb_prefix, flow_x_prefix, flow_y_prefix))
        dir_name = locate_directory(frame_dir)

        num_x = total_num[1]
        num_y = total_num[2]
        if num_x != num_y:
            raise ValueError(f'x and y direction have different number '
                             f'of flow images in video directory: {frame_dir}')
        if i % 200 == 0:
            print(f'{i} videos parsed')

        frame_dict[dir_name] = (frame_dir, total_num[0], num_x)

    print('frame directory analysis done')
    return frame_dict


def parse_args():
    parser = argparse.ArgumentParser(description='build file list for HVU')
    parser.add_argument('--input_csv', type=str, help='path of input csv file')
    parser.add_argument(
        '--src_dir', type=str, help='source video / frames directory')
    parser.add_argument(
        '--output',
        type=str,
        help='output filename, should \
        ends with .json')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['frames', 'videos'],
        help='generate file list for frames or videos')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tag_cates = mmcv.load(tag_file)
    tag2category = {}
    for k in tag_cates:
        for tag in tag_cates[k]:
            tag2category[tag] = k

    data_list = open(args.input_csv).readlines()
    data_list = [x.strip().split(',') for x in data_list[1:]]

    if args.mode == 'videos':
        downloaded = os.listdir(args.src_dir)
        downloaded = [x.split('.')[0] for x in downloaded]
        downloaded_set = set(downloaded)
    else:
        parse_result = parse_directory(args.src_dir)
        downloaded_set = set(parse_result)

    def parse_line(line):
        tags, youtube_id, start, end = line
        start, end = int(float(start) * 10), int(float(end) * 10)
        newname = f'{youtube_id}_{start:06d}_{end:06d}'
        tags = tags.split('|')
        all_tags = {}
        for tag in tags:
            category = tag2category[tag]
            all_tags.setdefault(category,
                                []).append(tag_cates[category].index(tag))
        return newname, all_tags

    data_list = [parse_line(line) for line in data_list]
    data_list = [line for line in data_list if line[0] in downloaded_set]

    if args.mode == 'frames':
        result = [
            dict(
                frame_dir=k[0], total_frames=parse_result[k[0]][1], label=k[1])
            for k in data_list
        ]
    elif args.mode == 'videos':
        result = [dict(filename=k[0] + '.mp4', label=k[1]) for k in data_list]
    mmcv.dump(result, args.output)
