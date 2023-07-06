# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import multiprocessing
import os
from collections import defaultdict

FPS = 30


def get_video_info(frame_folder):
    folder_name = frame_folder.split('/')[-1]
    filename = folder_name.split('_')
    video_id = '_'.join(filename[:-2])
    start = int(filename[-2])
    length = len(os.listdir(frame_folder)) // FPS
    return (video_id, start, start + length, folder_name)


def get_avaialble_clips(frame_root, num_cpus):
    folders = os.listdir(frame_root)
    folders = ['%s/%s' % (frame_root, folder) for folder in folders]
    pool = multiprocessing.Pool(num_cpus)
    outputs = pool.map(get_video_info, folders)
    lookup = defaultdict(list)
    for record in outputs:
        lookup[record[0]].append(record[1:])
    return lookup


def filter_train_list(kinetics_anotation_file, lookup):
    with open(kinetics_anotation_file) as f:
        anotated_frames = [i.split(',') for i in f.readlines()]
        anotated_frames = [i for i in anotated_frames if len(i) == 7]

    filtered = []
    for line in anotated_frames:
        if line[0] not in lookup:
            continue
        flag = False
        for start, end, video_path in lookup[line[0]]:
            if start < float(line[1]) < end:
                flag = True
                break
        if flag is False:
            continue

        frame_idx, x1, y1, x2, y2, label = list(map(float, line[1:7]))
        frame_idx, label = int(frame_idx), int(label)

        string = (f'{video_path},{frame_idx},'
                  f'{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{label},-1\n')

        filtered.append(string)
    return filtered


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--avakinetics_anotation',
        type=str,
        default='./ava_kinetics_v1_0',
        help='the directory to ava-kinetics annotations')
    p.add_argument(
        '--num_workers',
        type=int,
        default=-1,
        help='number of workers used for multiprocessing')
    p.add_argument(
        '--avakinetics_root',
        type=str,
        default='../../../data/ava_kinetics',
        help='the path to save ava-kinetics videos')
    args = p.parse_args()

    if args.num_workers > 0:
        num_workers = args.num_workers
    else:
        num_workers = max(multiprocessing.cpu_count() - 1, 1)

    frame_root = args.avakinetics_root + '/rawframes/'
    frame_root = os.path.abspath(frame_root)
    lookup = get_avaialble_clips(frame_root, num_workers)

    kinetics_train = args.avakinetics_anotation + '/kinetics_train_v1.0.csv'
    filtered_list = filter_train_list(kinetics_train, lookup)

    with open('%s/kinetics_train.csv' % args.avakinetics_root, 'w') as f:
        for line in filtered_list:
            f.write(line)
