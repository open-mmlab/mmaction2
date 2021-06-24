import argparse
import glob
import json
import os.path as osp
import random

from mmcv.runner import set_random_seed
from tools.data.anno_txt2json import lines2dictlist
from tools.data.parse_file_list import (parse_directory, parse_diving48_splits,
                                        parse_hmdb51_split,
                                        parse_jester_splits,
                                        parse_kinetics_splits,
                                        parse_mit_splits, parse_mmit_splits,
                                        parse_sthv1_splits, parse_sthv2_splits,
                                        parse_ucf101_splits)


def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'dataset',
        type=str,
        choices=[
            'ucf101', 'kinetics400', 'kinetics600', 'kinetics700', 'thumos14',
            'sthv1', 'sthv2', 'mit', 'mmit', 'activitynet', 'hmdb51', 'jester',
            'diving48'
        ],
        help='dataset to be built file list')
    parser.add_argument(
        'src_folder', type=str, help='root directory for the frames or videos')
    parser.add_argument(
        '--rgb-prefix', type=str, default='img_', help='prefix of rgb frames')
    parser.add_argument(
        '--flow-x-prefix',
        type=str,
        default='flow_x_',
        help='prefix of flow x frames')
    parser.add_argument(
        '--flow-y-prefix',
        type=str,
        default='flow_y_',
        help='prefix of flow y frames')
    parser.add_argument(
        '--num-split',
        type=int,
        default=3,
        help='number of split to file list')
    parser.add_argument(
        '--subset',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='subset to generate file list')
    parser.add_argument(
        '--level',
        type=int,
        default=2,
        choices=[1, 2],
        help='directory level of data')
    parser.add_argument(
        '--format',
        type=str,
        default='rawframes',
        choices=['rawframes', 'videos'],
        help='data format')
    parser.add_argument(
        '--out-root-path',
        type=str,
        default='data/',
        help='root path for output')
    parser.add_argument(
        '--output-format',
        type=str,
        default='txt',
        choices=['txt', 'json'],
        help='built file list format')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=False,
        help='whether to shuffle the file list')
    args = parser.parse_args()

    return args


def build_file_list(splits, frame_info, shuffle=False):
    """Build file list for a certain data split.

    Args:
        splits (tuple): Data split to generate file list.
        frame_info (dict): Dict mapping from frames to path. e.g.,
            'Skiing/v_Skiing_g18_c02': ('data/ucf101/rawframes/Skiing/v_Skiing_g18_c02', 0, 0).  # noqa: E501
        shuffle (bool): Whether to shuffle the file list.

    Returns:
        tuple: RGB file list for training and testing, together with
            Flow file list for training and testing.
    """

    def build_list(split):
        """Build RGB and Flow file list with a given split.

        Args:
            split (list): Split to be generate file list.

        Returns:
            tuple[list, list]: (rgb_list, flow_list), rgb_list is the
                generated file list for rgb, flow_list is the generated
                file list for flow.
        """
        rgb_list, flow_list = list(), list()
        for item in split:
            if item[0] not in frame_info:
                continue
            if frame_info[item[0]][1] > 0:
                # rawframes
                rgb_cnt = frame_info[item[0]][1]
                flow_cnt = frame_info[item[0]][2]
                if isinstance(item[1], int):
                    rgb_list.append(f'{item[0]} {rgb_cnt} {item[1]}\n')
                    flow_list.append(f'{item[0]} {flow_cnt} {item[1]}\n')
                elif isinstance(item[1], list):
                    # only for multi-label datasets like mmit
                    rgb_list.append(f'{item[0]} {rgb_cnt} ' +
                                    ' '.join([str(digit)
                                              for digit in item[1]]) + '\n')
                    rgb_list.append(f'{item[0]} {flow_cnt} ' +
                                    ' '.join([str(digit)
                                              for digit in item[1]]) + '\n')
                else:
                    raise ValueError(
                        'frame_info should be ' +
                        '[`video`(str), `label`(int)|`labels(list[int])`')
            else:
                # videos
                if isinstance(item[1], int):
                    rgb_list.append(f'{frame_info[item[0]][0]} {item[1]}\n')
                    flow_list.append(f'{frame_info[item[0]][0]} {item[1]}\n')
                elif isinstance(item[1], list):
                    # only for multi-label datasets like mmit
                    rgb_list.append(f'{frame_info[item[0]][0]} ' +
                                    ' '.join([str(digit)
                                              for digit in item[1]]) + '\n')
                    flow_list.append(
                        f'{frame_info[item[0]][0]} ' +
                        ' '.join([str(digit) for digit in item[1]]) + '\n')
                else:
                    raise ValueError(
                        'frame_info should be ' +
                        '[`video`(str), `label`(int)|`labels(list[int])`')
        if shuffle:
            random.shuffle(rgb_list)
            random.shuffle(flow_list)
        return rgb_list, flow_list

    train_rgb_list, train_flow_list = build_list(splits[0])
    test_rgb_list, test_flow_list = build_list(splits[1])
    return (train_rgb_list, test_rgb_list), (train_flow_list, test_flow_list)


def main():
    args = parse_args()

    if args.seed is not None:
        print(f'Set random seed to {args.seed}')
        set_random_seed(args.seed)

    if args.format == 'rawframes':
        frame_info = parse_directory(
            args.src_folder,
            rgb_prefix=args.rgb_prefix,
            flow_x_prefix=args.flow_x_prefix,
            flow_y_prefix=args.flow_y_prefix,
            level=args.level)
    elif args.format == 'videos':
        if args.level == 1:
            # search for one-level directory
            video_list = glob.glob(osp.join(args.src_folder, '*'))
        elif args.level == 2:
            # search for two-level directory
            video_list = glob.glob(osp.join(args.src_folder, '*', '*'))
        else:
            raise ValueError(f'level must be 1 or 2, but got {args.level}')
        frame_info = {}
        for video in video_list:
            video_path = osp.relpath(video, args.src_folder)
            # video_id: (video_relative_path, -1, -1)
            frame_info[osp.splitext(video_path)[0]] = (video_path, -1, -1)
    else:
        raise NotImplementedError('only rawframes and videos are supported')

    if args.dataset == 'ucf101':
        splits = parse_ucf101_splits(args.level)
    elif args.dataset == 'sthv1':
        splits = parse_sthv1_splits(args.level)
    elif args.dataset == 'sthv2':
        splits = parse_sthv2_splits(args.level)
    elif args.dataset == 'mit':
        splits = parse_mit_splits()
    elif args.dataset == 'mmit':
        splits = parse_mmit_splits()
    elif args.dataset in ['kinetics400', 'kinetics600', 'kinetics700']:
        splits = parse_kinetics_splits(args.level, args.dataset)
    elif args.dataset == 'hmdb51':
        splits = parse_hmdb51_split(args.level)
    elif args.dataset == 'jester':
        splits = parse_jester_splits(args.level)
    elif args.dataset == 'diving48':
        splits = parse_diving48_splits()
    else:
        raise ValueError(
            f"Supported datasets are 'ucf101, sthv1, sthv2', 'jester', "
            f"'mmit', 'mit', 'kinetics400', 'kinetics600', 'kinetics700', but "
            f'got {args.dataset}')

    assert len(splits) == args.num_split

    out_path = args.out_root_path + args.dataset

    if len(splits) > 1:
        for i, split in enumerate(splits):
            file_lists = build_file_list(
                split, frame_info, shuffle=args.shuffle)
            train_name = f'{args.dataset}_train_split_{i+1}_{args.format}.txt'
            val_name = f'{args.dataset}_val_split_{i+1}_{args.format}.txt'
            if args.output_format == 'txt':
                with open(osp.join(out_path, train_name), 'w') as f:
                    f.writelines(file_lists[0][0])
                with open(osp.join(out_path, val_name), 'w') as f:
                    f.writelines(file_lists[0][1])
            elif args.output_format == 'json':
                train_list = lines2dictlist(file_lists[0][0], args.format)
                val_list = lines2dictlist(file_lists[0][1], args.format)
                train_name = train_name.replace('.txt', '.json')
                val_name = val_name.replace('.txt', '.json')
                with open(osp.join(out_path, train_name), 'w') as f:
                    json.dump(train_list, f)
                with open(osp.join(out_path, val_name), 'w') as f:
                    json.dump(val_list, f)
    else:
        lists = build_file_list(splits[0], frame_info, shuffle=args.shuffle)

        if args.subset == 'train':
            ind = 0
        elif args.subset == 'val':
            ind = 1
        elif args.subset == 'test':
            ind = 2
        else:
            raise ValueError(f"subset must be in ['train', 'val', 'test'], "
                             f'but got {args.subset}.')

        filename = f'{args.dataset}_{args.subset}_list_{args.format}.txt'
        if args.output_format == 'txt':
            with open(osp.join(out_path, filename), 'w') as f:
                f.writelines(lists[0][ind])
        elif args.output_format == 'json':
            data_list = lines2dictlist(lists[0][ind], args.format)
            filename = filename.replace('.txt', '.json')
            with open(osp.join(out_path, filename), 'w') as f:
                json.dump(data_list, f)


if __name__ == '__main__':
    main()
