# Copyright (c) OpenMMLab. All rights reserved.
# In this example, we convert babel120_train to MMAction2 format
# The required files can be downloaded from the homepage of BABEL project
import numpy as np
from mmcv import dump, load


def gen_babel(x, y):
    data = []
    for i, xx in enumerate(x):
        sample = dict()
        sample['keypoint'] = xx.transpose(3, 1, 2, 0).astype(np.float16)
        sample['label'] = y[1][0][i]
        names = [y[0][i], y[1][1][i], y[1][2][i], y[1][3][i]]
        sample['frame_dir'] = '_'.join([str(k) for k in names])
        sample['total_frames'] = 150
        data.append(sample)
    return data


x = np.load('train_ntu_sk_120.npy')
y = load('train_label_120.pkl')

data = gen_babel(x, y)
dump(data, 'babel120_train.pkl')
