# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import multiprocessing
from collections import defaultdict

import decord


def get_video_info(video_path):
    filename = video_path.split('/')[-1]
    filename = filename.split('.')[0].split('_')
    video_id = '_'.join(filename[:-2])
    start = int(filename[-2])
    vr = decord.VideoReader(video_path)
    length = len(vr) // vr.get_avg_fps()
    return (video_id, start, start + length, video_path)


def get_avaialble_clips(video_root, num_cpus):
    videos = os.listdir(video_root)
    videos = ['%s/%s' % (video_root, video) for video in videos]
    pool = multiprocessing.Pool(num_cpus)
    outputs = pool.map(get_video_info, videos)
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
        video_path = os.path.abspath(video_path)
        string = '%s,%d,%.3f,%.3f,%.3f,%.3f,%d,-1\n' % (video_path,
                                                        int(float(line[1])),
                                                        float(line[2]),
                                                        float(line[3]),
                                                        float(line[4]),
                                                        float(line[5]),
                                                        int(float(line[6])))
        filtered.append(string)
    return filtered


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--avakinetics_anotation', type=str, 
                   default='./ava_kinetics_v1_0',
                   help='the directory to ava-kinetics anotations')
    p.add_argument('--num_workers', type=int, default=-1,
                   help='number of workers used for multiprocessing')
    p.add_argument('--avakinetics_root', type=str,
                   default='../../../data/ava_kinetics',
                   help='the path to save ava-kinetics videos')
    args = p.parse_args()

    if args.num_workers > 0:
        num_workers = args.num_workers
    else:
        num_workers = max(multiprocessing.cpu_count() - 1, 1)
    lookup = get_avaialble_clips(args.avakinetics_root + '/videos/', 
                                 num_workers)

    kinetics_train = args.avakinetics_anotation + '/kinetics_train_v1.0.csv'
    filtered_list = filter_train_list(kinetics_train, lookup)

    with open('%s/kinetics_train.csv' % args.avakinetics_root, 'w') as f:
        for line in filtered_list:
            f.write(line)
