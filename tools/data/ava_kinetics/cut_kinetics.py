# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import multiprocessing
import os
from collections import defaultdict
from typing import List

import decord


def get_kinetics_frames(kinetics_anotation_file: str) -> dict:
    """Given the AVA-kinetics anotation file, return a lookup to map the video
    id and the the set of timestamps involved of this video id.

    Args:
        kinetics_anotation_file (str): Path to the AVA-like anotation file for
            the kinetics subset.
    Returns:
        dict: the dict keys are the kinetics videos' video id. The values are
            the set of timestamps involved.
    """
    with open(kinetics_anotation_file) as f:
        anotated_frames = [i.split(',') for i in f.readlines()]
        anotated_frames = [i for i in anotated_frames if len(i) == 7]
        anotated_frames = [(i[0], int(float(i[1]))) for i in anotated_frames]

    frame_lookup = defaultdict(set)
    for video_id, timestamp in anotated_frames:
        frame_lookup[video_id].add(timestamp)
    return frame_lookup


def filter_missing_videos(kinetics_list: str, frame_lookup: dict) -> dict:
    """Given the kinetics700 dataset list, remove the video ids from the lookup
    that are missing videos or frames.

    Args:
        kinetics_list (str): Path to the kinetics700 dataset list.
            The content of the list should be:
                ```
                Path_to_video1 label_1\n
                Path_to_video2 label_2\n
                ...
                Path_to_videon label_n\n
                ```
            The start and end of the video must be contained in the filename.
            For example:
                ```
                class602/o3lCwWyyc_s_000012_000022.mp4\n
                ```
        frame_lookup (dict): the dict from `get_kinetics_frames`.
    Returns:
        dict: the dict keys are the kinetics videos' video id. The values are
            the a list of tuples:
                (start_of_the_video, end_of_the_video, video_path)
    """
    video_lookup = defaultdict(set)
    with open(kinetics_list) as f:
        for line in f.readlines():
            video_path = line.split(' ')[0]  # remove label information
            video_name = video_path.split('/')[-1]  # get the file name
            video_name = video_name.split('.')[0]  # remove file extensions
            video_name = video_name.split('_')
            video_id = '_'.join(video_name[:-2])
            if video_id not in frame_lookup:
                continue

            start, end = int(video_name[-2]), int(video_name[-1])
            frames = frame_lookup[video_id]
            frames = [frame for frame in frames if start < frame < end]
            if len(frames) == 0:
                continue

            start, end = max(start, min(frames) - 2), min(end, max(frames) + 2)
            video_lookup[video_id].add((start, end, video_path))

    # Some frame ids exist in multiple videos in the Kinetics dataset.
    # The reason is the part of one video may fall into different categories.
    # Remove the duplicated records
    for video in video_lookup:
        if len(video_lookup[video]) == 1:
            continue
        info_list = list(video_lookup[video])
        removed_list = []
        for i, info_i in enumerate(info_list):
            start_i, end_i, _ = info_i
            for j in range(i + 1, len(info_list)):
                start_j, end_j, _ = info_list[j]
                if start_i <= start_j and end_j <= end_i:
                    removed_list.append(j)
                elif start_j <= start_i and end_i <= end_j:
                    removed_list.append(i)
        new_list = []
        for i, info in enumerate(info_list):
            if i not in removed_list:
                new_list.append(info)
        video_lookup[video] = set(new_list)
    return video_lookup


template = ('ffmpeg -ss %d -t %d -accurate_seek -i'
            ' %s -r 30 -avoid_negative_ts 1 %s')


def generate_cut_cmds(video_lookup: dict, data_root: str) -> List[str]:
    cmds = []
    for video_id in video_lookup:
        for start, end, video_path in video_lookup[video_id]:
            start0 = int(video_path.split('_')[-2])
            new_path = '%s/%s_%06d_%06d.mp4' % (data_root, video_id, start,
                                                end)
            cmd = template % (start - start0, end - start, video_path,
                              new_path)
            cmds.append(cmd)
    return cmds


def run_cmd(cmd):
    os.system(cmd)
    return


def remove_failed_video(video_path: str) -> None:
    """Given the path to the video, delete the video if it cannot be read or if
    the actual length of the video is 0.75 seconds shorter than expected."""
    try:
        v = decord.VideoReader(video_path)
        fps = v.get_avg_fps()
        num_frames = len(v)
        x = video_path.split('.')[0].split('_')
        time = int(x[-1]) - int(x[-2])
        if num_frames < (time - 3 / 4) * fps:
            os.remove(video_path)
    except:  # noqa: E722
        os.remove(video_path)
    return


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--avakinetics_anotation',
        type=str,
        default='./ava_kinetics_v1_0',
        help='the directory to ava-kinetics anotations')
    p.add_argument(
        '--kinetics_list',
        type=str,
        help='the datalist of the kinetics700 training videos')
    p.add_argument(
        '--num_workers',
        type=int,
        default=-1,
        help='number of workers used for multiprocessing')
    p.add_argument(
        '--avakinetics_root',
        type=str,
        default='../../../data/ava_kinetics',
        help='the path to save ava-kinetics dataset')
    args = p.parse_args()

    if args.num_workers > 0:
        num_workers = args.num_workers
    else:
        num_workers = max(multiprocessing.cpu_count() - 1, 1)

    # Find videos from the Kinetics700 dataset required for AVA-Kinetics
    kinetics_train = args.avakinetics_anotation + '/kinetics_train_v1.0.csv'
    frame_lookup = get_kinetics_frames(kinetics_train)
    video_lookup = filter_missing_videos(args.kinetics_list, frame_lookup)

    root = args.avakinetics_root
    os.makedirs(root, exist_ok=True)
    video_path = root + '/videos/'
    os.makedirs(video_path, exist_ok=True)
    all_cmds = generate_cut_cmds(video_lookup, video_path)

    # Cut and save the videos for AVA-Kinetics
    pool = multiprocessing.Pool(num_workers)
    _ = pool.map(run_cmd, all_cmds)

    # Remove failed videos
    videos = os.listdir(video_path)
    videos = ['%s/%s' % (video_path, video) for video in videos]
    _ = pool.map(remove_failed_video, videos)
