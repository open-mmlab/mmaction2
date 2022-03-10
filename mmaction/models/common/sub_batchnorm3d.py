# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.cnn import NORM_LAYERS


@NORM_LAYERS.register_module()
class SubBatchNorm3D(nn.Module):
    """Sub BatchNorm3d splits the batch dimension into N splits, and run BN on
    each of them separately (so that the stats are computed on each subset of
    examples (1/N of batch) independently). During evaluation, it aggregates
    the stats from all splits into one BN.

    Args:
        num_features (int): Dimensions of BatchNorm.
    """

    def __init__(self, num_features, **cfg):
        super(SubBatchNorm3D, self).__init__()

        self.num_features = num_features
        self.cfg_ = deepcopy(cfg)
        self.num_splits = self.cfg_.pop('num_splits', 1)
        self.num_features_split = self.num_features * self.num_splits
        # only keep one set of affine params, not in .bn or .split_bn
        self.cfg_['affine'] = False
        self.bn = nn.BatchNorm3d(num_features, **self.cfg_)
        self.split_bn = nn.BatchNorm3d(self.num_features_split, **self.cfg_)
        self.init_weights(cfg)

    def init_weights(self, cfg):
        if cfg.get('affine', True):
            self.weight = torch.nn.Parameter(torch.ones(self.num_features))
            self.bias = torch.nn.Parameter(torch.zeros(self.num_features))
            self.affine = True
        else:
            self.affine = False

    def _get_aggregated_mean_std(self, means, stds, n):
        mean = means.view(n, -1).sum(0) / n
        std = stds.view(n, -1).sum(0) / n + (
            (means.view(n, -1) - mean)**2).view(n, -1).sum(0) / n
        return mean.detach(), std.detach()

    def aggregate_stats(self):
        """Synchronize running_mean, and running_var to self.bn.

        Call this before eval, then call model.eval(); When eval, forward
        function will call self.bn instead of self.split_bn, During this time
        the running_mean, and running_var of self.bn has been obtained from
        self.split_bn.
        """
        if self.split_bn.track_running_stats:
            aggre_func = self._get_aggregated_mean_std
            self.bn.running_mean.data, self.bn.running_var.data = aggre_func(
                self.split_bn.running_mean, self.split_bn.running_var,
                self.num_splits)
        self.bn.num_batches_tracked = self.split_bn.num_batches_tracked.detach(
        )

    def forward(self, x):
        if self.training:
            n, c, t, h, w = x.shape
            assert n % self.num_splits == 0
            x = x.view(n // self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, t, h, w)
        else:
            x = self.bn(x)
        if self.affine:
            x = x * self.weight.view(-1, 1, 1, 1)
            x = x + self.bias.view(-1, 1, 1, 1)
        return x
