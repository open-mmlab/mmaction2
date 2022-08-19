# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine
import torch
import torch.distributed as dist
import torch.nn as nn

from mmaction.registry import MODELS

try:
    from mmdet.registry import MODELS as MMDET_MODELS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

# Note: All these heads take 5D Tensors as input (N, C, T, H, W)


@MODELS.register_module()
class LFBInferHead(nn.Module):
    """Long-Term Feature Bank Infer Head.

    This head is used to derive and save the LFB without affecting the input.
    Args:
        lfb_prefix_path (str): The prefix path to store the lfb.
        dataset_mode (str, optional): Which dataset to be inferred. Choices are
            'train', 'val' or 'test'. Default: 'train'.
        use_half_precision (bool, optional): Whether to store the
            half-precision roi features. Default: True.
        temporal_pool_type (str): The temporal pool type. Choices are 'avg' or
            'max'. Default: 'avg'.
        spatial_pool_type (str): The spatial pool type. Choices are 'avg' or
            'max'. Default: 'max'.
    """

    def __init__(self,
                 lfb_prefix_path,
                 dataset_mode='train',
                 use_half_precision=True,
                 temporal_pool_type='avg',
                 spatial_pool_type='max'):
        super().__init__()
        rank, _ = mmengine.dist.get_dist_info()
        if rank == 0:
            if not osp.exists(lfb_prefix_path):
                print(f'lfb prefix path {lfb_prefix_path} does not exist. '
                      f'Creating the folder...')
                mmengine.mkdir_or_exist(lfb_prefix_path)
            print('\nInferring LFB...')

        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.lfb_prefix_path = lfb_prefix_path
        self.dataset_mode = dataset_mode
        self.use_half_precision = use_half_precision

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
        """LFBInferHead has no parameters to be initialized."""
        pass

    def forward(self, x, rois, img_metas, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The extracted RoI feature.
            rois (torch.Tensor): The regions of interest.
            img_metas (List[dict]): The meta information of the data.

        Returns:
            torch.Tensor: The RoI features that have interacted with context
        """
        # [N, C, 1, 1, 1]
        features = self.temporal_pool(x)
        features = self.spatial_pool(features)
        if self.use_half_precision:
            features = features.half()

        inds = rois[:, 0].type(torch.int64)
        for ind in inds:
            self.all_metadata.append(img_metas[ind]['img_key'])
        self.all_features += list(features)

        # Return the input directly and doesn't affect the input.
        return x

    def __del__(self):
        assert len(self.all_features) == len(self.all_metadata), (
            'features and metadata are not equal in length!')

        rank, world_size = mmengine.dist.get_dist_info()
        if world_size > 1:
            dist.barrier()

        _lfb = {}
        for feature, metadata in zip(self.all_features, self.all_metadata):
            video_id, timestamp = metadata.split(',')
            timestamp = int(timestamp)

            if video_id not in _lfb:
                _lfb[video_id] = {}
            if timestamp not in _lfb[video_id]:
                _lfb[video_id][timestamp] = []

            _lfb[video_id][timestamp].append(torch.squeeze(feature))

        _lfb_file_path = osp.normpath(
            osp.join(self.lfb_prefix_path,
                     f'_lfb_{self.dataset_mode}_{rank}.pkl'))
        torch.save(_lfb, _lfb_file_path)
        print(f'{len(self.all_features)} features from {len(_lfb)} videos '
              f'on GPU {rank} have been stored in {_lfb_file_path}.')

        # Synchronizes all processes to make sure all gpus have stored their
        # roi features
        if world_size > 1:
            dist.barrier()
        if rank > 0:
            return

        print('Gathering all the roi features...')

        lfb = {}
        for rank_id in range(world_size):
            _lfb_file_path = osp.normpath(
                osp.join(self.lfb_prefix_path,
                         f'_lfb_{self.dataset_mode}_{rank_id}.pkl'))

            # Since each frame will only be distributed to one GPU,
            # the roi features on the same timestamp of the same video are all
            # on the same GPU
            _lfb = torch.load(_lfb_file_path)
            for video_id in _lfb:
                if video_id not in lfb:
                    lfb[video_id] = _lfb[video_id]
                else:
                    lfb[video_id].update(_lfb[video_id])

            osp.os.remove(_lfb_file_path)

        lfb_file_path = osp.normpath(
            osp.join(self.lfb_prefix_path, f'lfb_{self.dataset_mode}.pkl'))
        torch.save(lfb, lfb_file_path)
        print(f'LFB has been constructed in {lfb_file_path}!')


if mmdet_imported:
    MMDET_MODELS.register_module()(LFBInferHead)
