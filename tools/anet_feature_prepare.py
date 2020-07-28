import argparse
import multiprocessing
import os
import os.path as osp

import numpy as np
import scipy.interpolate
from mmcv import dump, load

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='ANet Feature Prepare')
    parser.add_argument('--rgb', default='', help='rgb feature root')
    parser.add_argument('--flow', default='', help='flow feature root')
    parser.add_argument('--dest', default='', help='dest root')
    parser.add_argument('--output-format', default='pkl', help='clip length')
    args = parser.parse_args()
    return args


def pool_feature(data, num_prop=100, num_sample_bin=3, pool_type='mean'):
    # the range of x will be [0, ..., len]
    if len(data) == 1:
        return np.concatenate([data] * num_prop)
    x_range = list(range(len(data)))
    f = scipy.interpolate.interp1d(x_range, data, axis=0)
    eps = 1e-4
    st, ed = eps, len(data) - 1 - eps
    anchor_size = (ed - st) / num_prop
    ptr = st
    feature = []
    for i in range(num_prop):
        x_new = [
            ptr + i / num_sample_bin * anchor_size
            for i in range(num_sample_bin)
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
    global args
    rgb_feat = load(osp.join(args.rgb, name))
    flow_feat = load(osp.join(args.flow, name))
    rgb_feat = pool_feature(rgb_feat)
    flow_feat = pool_feature(flow_feat)
    feat = np.concatenate([rgb_feat, flow_feat], axis=-1)
    if args.output_format == 'pkl':
        dump(feat, osp.join(args.dest, name))
    elif args.output_format == 'csv':
        feat = feat.tolist()
        lines = []
        line0 = ','.join(['f{}'.format(i) for i in range(400)])
        lines.append(line0)
        for line in feat:
            lines.append(','.join(['{:.4f}'.format(x) for x in line]))
        with open(osp.join(args.dest, name.replace('.pkl', '.csv')), 'w') as f:
            f.write('\n'.join(lines))


def main():
    global args
    args = parse_args()
    rgb_feat = os.listdir(args.rgb)
    flow_feat = os.listdir(args.flow)
    assert set(rgb_feat) == set(flow_feat)
    pool = multiprocessing.Pool(32)
    pool.map(merge_feat, rgb_feat)


if __name__ == '__main__':
    main()
