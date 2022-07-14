# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import os.path as osp

import mmcv
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


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotation_matrix(axis, theta):
    """Return the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians."""
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N C T V M -> N M T V C

    print('pad the null frames with the previous frames')
    prog_bar = mmcv.ProgressBar(len(s))
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp

            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate(
                            [person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break
        prog_bar.update()

    print('sub the center joint #1 (spine joint in ntu and '
          'neck joint in kinetics)')
    prog_bar = mmcv.ProgressBar(len(s))
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask
        prog_bar.update()

    print('parallel the bone between hip(jpt 0) and '
          'spine(jpt 1) of the first person to the z axis')
    prog_bar = mmcv.ProgressBar(len(s))
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)
        prog_bar.update()

    print('parallel the bone between right shoulder(jpt 8) and '
          'left shoulder(jpt 4) of the first person to the x axis')
    prog_bar = mmcv.ProgressBar(len(s))
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)
        prog_bar.update()

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['num_frame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []

        for t in range(skeleton_sequence['num_frame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
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


def get_nonzero_std(s):  # T V C
    index = s.sum(-1).sum(-1) != 0
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :,
                                                    2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton_filter(file)
    # num_frame = seq_info['num_frame']
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
    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            task='ntu60',
            benchmark='xsub',
            part='train',
            pre_norm=True):
    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []

    sample_name = []
    sample_label = []
    total_frames = []
    results = []

    for filename in os.listdir(data_path):
        if filename in ignored_samples:
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
        elif benchmark == 'xsetup':
            istraining = (setup_number in training_setups_120)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true),
                  dtype=np.float32)
    prog_bar = mmcv.ProgressBar(len(sample_name))
    for i, s in enumerate(sample_name):
        data = read_xyz(
            osp.join(data_path, s),
            max_body=max_body_kinect,
            num_joint=num_joint).astype(np.float32)
        fp[i, :, 0:data.shape[1], :, :] = data
        total_frames.append(data.shape[1])
        prog_bar.update()

    if pre_norm:
        fp = pre_normalization(fp)

    prog_bar = mmcv.ProgressBar(len(sample_name))
    for i, s in enumerate(sample_name):
        anno = dict()
        anno['total_frames'] = total_frames[i]
        anno['keypoint'] = fp[i, :, 0:total_frames[i], :, :].transpose(
            3, 1, 2, 0)  # C T V M -> M T V C
        anno['frame_dir'] = osp.splitext(s)[0]
        anno['img_shape'] = (1080, 1920)
        anno['original_shape'] = (1080, 1920)
        anno['label'] = sample_label[i]

        results.append(anno)
        prog_bar.update()

    output_path = '{}/{}.pkl'.format(out_path, part)
    mmcv.dump(results, output_path)
    print(f'{benchmark}-{part} finish~!')


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
        '--out-folder', type=str, default='../../../data/ntu60/')
    parser.add_argument('--task', type=str, default='ntu60')
    args = parser.parse_args()

    assert args.task in ['ntu60', 'ntu120']

    if args.task == 'ntu60':
        benchmark = ['xsub', 'xview']
    else:
        benchmark = ['xsub', 'xsetup']
    part = ['train', 'val']

    for b in benchmark:
        for p in part:
            out_path = osp.join(args.out_folder, b)
            if not osp.exists(out_path):
                os.makedirs(out_path)
            gendata(
                args.data_path,
                out_path,
                args.ignored_sample_path,
                args.task,
                benchmark=b,
                part=p)
