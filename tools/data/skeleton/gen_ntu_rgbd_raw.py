# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import mmengine
import numpy as np

training_subjects_60 = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras_60 = [2, 3]
training_subjects_120 = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82,
    83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
]
training_setups_120 = [
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32
]
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300


def read_skeleton_filter(file: str) -> Dict:
    with open(file, 'r') as f:
        skeleton_sequence = {'num_frame': int(f.readline()), 'frameInfo': []}

        for t in range(skeleton_sequence['num_frame']):
            frame_info = {'numBody': int(f.readline()), 'bodyInfo': []}

            for m in range(frame_info['numBody']):
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key,
                                    f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key,
                                        f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s: np.ndarray) -> float:  # T V C
    index = s.sum(-1).sum(-1) != 0
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + \
            s[:, :, 1].std() + \
            s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file: str, max_body: int = 4, num_joint: int = 25) -> np.ndarray:
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['num_frame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    # filter padding body
    data = data[data.sum((1, 2, 3)) != 0]
    return data


def get_names_and_labels(data_path: str,
                         task: str,
                         benchmark: str,
                         ignored_samples: Optional[List[str]] = None) -> Tuple:
    train_names = []
    train_labels = []
    val_names = []
    val_labels = []

    for filename in os.listdir(data_path):
        if ignored_samples is not None and filename in ignored_samples:
            continue

        setup_number = int(filename[filename.find('S') + 1:filename.find('S') +
                                    4])
        action_class = int(filename[filename.find('A') + 1:filename.find('A') +
                                    4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') +
                                  4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') +
                                 4])

        if benchmark == 'xsub':
            if task == 'ntu60':
                istraining = (subject_id in training_subjects_60)
            else:
                istraining = (subject_id in training_subjects_120)
        elif benchmark == 'xview':
            istraining = (camera_id in training_cameras_60)
        elif benchmark == 'xset':
            istraining = (setup_number in training_setups_120)
        else:
            raise ValueError()

        if istraining:
            train_names.append(filename)
            train_labels.append(action_class - 1)
        else:
            val_names.append(filename)
            val_labels.append(action_class - 1)

    return train_names, train_labels, val_names, val_labels


def gendata(data_path: str,
            out_path: str,
            ignored_sample_path: Optional[str] = None,
            task: str = 'ntu60') -> None:
    split = dict()

    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []

    if task == 'ntu60':
        benchmarks = ['xsub', 'xview']
    else:
        benchmarks = ['xsub', 'xset']

    names = None
    labels = None
    for benchmark in benchmarks:
        train_names, train_labels, val_names, val_labels = \
            get_names_and_labels(data_path, task, benchmark, ignored_samples)
        split[f'{benchmark}_train'] = [osp.splitext(s)[0] for s in train_names]
        split[f'{benchmark}_val'] = [osp.splitext(s)[0] for s in val_names]

        if names is None and labels is None:
            names = train_names + val_names
            labels = train_labels + val_labels

    results = []

    prog_bar = mmengine.ProgressBar(len(names))
    for i, s in enumerate(names):
        ske = read_xyz(
            osp.join(data_path, s),
            max_body=max_body_kinect,
            num_joint=num_joint).astype(np.float16)

        anno = dict()
        anno['frame_dir'] = osp.splitext(s)[0]
        anno['label'] = labels[i]
        anno['keypoint'] = ske
        anno['total_frames'] = ske.shape[1]
        results.append(anno)
        prog_bar.update()

    annotations = {'split': split, 'annotations': results}
    mmengine.dump(annotations, f'{out_path}/{task}_3d.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for NTURGB-D raw skeleton data')
    parser.add_argument(
        '--data-path',
        type=str,
        help='raw skeleton data path',
        default='../../../data/ntu60/nturgb+d_skeletons/')
    parser.add_argument(
        '--ignored-sample-path',
        type=str,
        default='NTU_RGBD_samples_with_missing_skeletons.txt')
    parser.add_argument(
        '--out-folder', type=str, default='../../../data/skeleton/')
    parser.add_argument('--task', type=str, default='ntu60')
    args = parser.parse_args()

    assert args.task in ['ntu60', 'ntu120']

    mmengine.mkdir_or_exist(args.out_folder)

    gendata(args.data_path, args.out_folder, args.ignored_sample_path,
            args.task)
