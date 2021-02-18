import os.path as osp

import mmcv
import numpy as np
import torch


class LFB(object):

    def __init__(self,
                 lfb_prefix_path,
                 max_num_feat_per_step=5,
                 window_size=60,
                 num_lfb_channels=2048,
                 dataset_modes=['train', 'val']):
        if not osp.exists(lfb_prefix_path):
            raise ValueError(
                f'lfb prefix path {lfb_prefix_path} does not exist!')
        self.lfb_prefix_path = lfb_prefix_path
        self.max_num_feat_per_step = max_num_feat_per_step
        self.window_size = window_size
        self.num_lfb_channels = num_lfb_channels
        self.lfb = {}

        if not isinstance(dataset_modes, list):
            assert isinstance(dataset_modes, str)
            dataset_modes = [dataset_modes]
        for dataset_mode in dataset_modes:
            lfb_path = osp.normpath(
                osp.join(lfb_prefix_path, f'lfb_{dataset_mode}.pkl'))
            print(f'Loading LFB from {lfb_path}...')
            self.lfb.update(mmcv.load(lfb_path))

    @staticmethod
    def sample_long_term_features(self, video_id, timestamp):
        video_features = self.lfb[video_id]
        # sample long term features
        window_size, K = self.window_size, self.max_num_feat_per_step
        start = timestamp - (window_size // 2)
        lt_feats = np.zeros((window_size * K, self.num_lfb_channels))

        for idx, sec in enumerate(range(start, start + window_size)):
            if sec in video_features:
                num_feat = len(video_features[sec])
                num_feat_used = min(num_feat, K)
                random_lfb_indices = np.random.choice(
                    range(num_feat), num_feat_used, replace=False)
                for k, rand_idx in enumerate(random_lfb_indices):
                    lt_feats[idx * K + k] = video_features[sec][rand_idx]

        # [window_size * max_num_feat_per_step, num_lfb_channels]
        return torch.tensor(lt_feats)

    def __getitem__(self, img_key):
        video_id, timestamp = img_key.split(',')
        return self.sample_long_term_features(video_id, int(timestamp))

    def __len__(self):
        return len(self.lfb)
