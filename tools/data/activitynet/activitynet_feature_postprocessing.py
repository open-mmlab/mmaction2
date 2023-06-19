# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import multiprocessing
import os
import os.path as osp

import numpy as np
import scipy.interpolate
from mmengine import dump, load

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='ANet Feature Prepare')
    parser.add_argument('--rgb', default='', help='rgb feature root')
    parser.add_argument('--flow', default='', help='flow feature root')
    parser.add_argument('--dest', default='', help='dest root')
    parser.add_argument('--output-format', default='csv')
    args = parser.parse_args()
    return args


def pool_feature(data, num_proposals=100, num_sample_bins=3, pool_type='mean'):
    """Pool features with arbitrary temporal length.

    Args:
        data (list[np.ndarray] | np.ndarray): Features of an untrimmed video,
            with arbitrary temporal length.
        num_proposals (int): The temporal dim of pooled feature. Default: 100.
        num_sample_bins (int): How many points to sample to get the feature
            vector at one timestamp. Default: 3.
        pool_type (str): Type of pooling to pool features. Choices are
            ['mean', 'max']. Default: 'mean'.

    Returns:
        np.ndarray: The pooled feature with shape num_proposals x feature_dim.
    """
    if len(data) == 1:
        return np.concatenate([data] * num_proposals)
    x_range = list(range(len(data)))
    f = scipy.interpolate.interp1d(x_range, data, axis=0)
    eps = 1e-4
    start, end = eps, len(data) - 1 - eps
    anchor_size = (end - start) / num_proposals
    ptr = start
    feature = []
    for _ in range(num_proposals):
        x_new = [
            ptr + i / num_sample_bins * anchor_size
            for i in range(num_sample_bins)
        ]
        y_new = f(x_new)
        if pool_type == 'mean':
            y_new = np.mean(y_new, axis=0)
        elif pool_type == 'max':
            y_new = np.max(y_new, axis=0)
        else:
            raise NotImplementedError('Unsupported pool type')
        feature.append(y_new)
        ptr += anchor_size
    feature = np.stack(feature)
    return feature


def merge_feat(name):
    # concatenate rgb feat and flow feat for a single sample
    rgb_feat = load(osp.join(args.rgb, name))
    flow_feat = load(osp.join(args.flow, name))
    rgb_feat = pool_feature(rgb_feat)
    flow_feat = pool_feature(flow_feat)
    feat = np.concatenate([rgb_feat, flow_feat], axis=-1)
    if not osp.exists(args.dest):
        os.system(f'mkdir -p {args.dest}')
    if args.output_format == 'pkl':
        dump(feat, osp.join(args.dest, name))
    elif args.output_format == 'csv':
        feat = feat.tolist()
        lines = []
        line0 = ','.join([f'f{i}' for i in range(400)])
        lines.append(line0)
        for line in feat:
            lines.append(','.join([f'{x:.4f}' for x in line]))
        with open(osp.join(args.dest, name.replace('.pkl', '.csv')), 'w') as f:
            f.write('\n'.join(lines))


def main():
    global args
    args = parse_args()
    rgb_feat = [file for file in os.listdir(args.rgb) if file.endswith('.pkl')]
    flow_feat = [
        file for file in os.listdir(args.flow) if file.endswith('.pkl')
    ]
    assert set(rgb_feat) == set(flow_feat)
    # for feat in rgb_feat:
    #     merge_feat(feat)
    pool = multiprocessing.Pool(32)
    pool.map(merge_feat, rgb_feat)


if __name__ == '__main__':
    main()
