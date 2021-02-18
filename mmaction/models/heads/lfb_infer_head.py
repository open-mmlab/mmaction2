import os
import os.path as osp

import torch
import torch.nn as nn
from mmcv.runner import get_dist_info

try:
    from mmdet.models.builder import SHARED_HEADS as MMDET_SHARED_HEADS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


class LFBInferHead(nn.Module):

    def __init__(self,
                 lfb_prefix_path,
                 dataset_mode='train',
                 temporal_pool_type='avg',
                 spatial_pool_type='max'):
        super().__init__()
        if not osp.exists(lfb_prefix_path):
            print(f'lfb prefix path {lfb_prefix_path} does not exist, '
                  f'creating the folder...')
            os.makedirs(lfb_prefix_path)
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.lfb_prefix_path = lfb_prefix_path
        self.dataset_mode = dataset_mode

        # Pool by default
        if temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

        self.all_features = []
        self.all_metadata = []

    def init_weights(self, pretrained=None):
        # LFBInferHead has no parameters to be initialized.
        pass

    def forward(self, x, rois, img_metas):
        # feature size [N, C, T, H, W]
        n, c, _, _, _ = x.shape

        # [N, C, 1, 1, 1]
        features = self.temporal_pool(x)
        features = self.spatial_pool(features)

        inds = rois[:, 0].type(torch.int64)
        for ind in inds:
            self.all_metadata.append(img_metas[ind]['img_key'])
        self.all_features += list(features)

        return x

    def __del__(self):
        """Only save LFB at local rank 0."""
        # TODO
        rank, world_size = get_dist_info()
        rank = int(os.environ.get('LOCAL_RANK', rank))
        if rank > 0:
            return

        lfb_file_path = osp.normpath(
            osp.join(self.lfb_prefix_path, f'lfb_{self.dataset_mode}.pkl'))
        print(f'Storing the feature bank in {lfb_file_path}...')
        assert len(self.all_features) == len(self.all_metadata), (
            'features and metadata are not equal in length!')

        lfb = {}
        for feature, metadata in zip(self.all_features, self.all_metadata):
            video_id, timestamp = metadata.split(',')
            timestamp = int(timestamp)

            if video_id not in lfb:
                print(f'Add {video_id} to LFB...')
                lfb[video_id] = {}
            if timestamp not in lfb[video_id]:
                lfb[video_id][timestamp] = []

            lfb[video_id][timestamp].append(torch.squeeze(feature))

        torch.save(lfb, lfb_file_path)

        print(f'LFB constructed! {len(self.all_features)} features from '
              f'{len(lfb)} videos have been stored in total.')


if mmdet_imported:
    MMDET_SHARED_HEADS.register_module()(LFBInferHead)
