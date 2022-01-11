import torch
import torch.nn as nn
from mmcv.cnn import NORM_LAYERS


@NORM_LAYERS.register_module()
class SubBatchNorm3D(nn.Module):
    """sub batchnorm 3d."""

    def __init__(self, num_features, **args):
        super(SubBatchNorm3D, self).__init__()

        self.num_features = num_features
        if 'num_splits' in args:
            self.num_splits = args['num_splits']
        else:
            self.num_splits = 1
        # print('sub bn num_splits', self.num_splits)
        args['num_features'] = self.num_features
        args['affine'] = False

        self.bn = nn.BatchNorm3d(self.num_features, affine=False)
        self.split_bn = nn.BatchNorm3d(
            self.num_features * self.num_splits, affine=False)
        self.affine = False

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
            aggregate = self._get_aggregated_mean_std
            self.bn.running_mean, self.bn.running_var = aggregate(
                self.split_bn.running_mean, self.split_bn.running_var,
                self.num_splits)
        else:
            # print(' self split_bn not tracking running status---')
            pass
        self.bn.num_batches_tracked = self.split_bn.num_batches_tracked.detach(
        )

    def forward(self, x):
        if self.training:
            n, c, t, h, w = x.shape
            x = x.view(n // self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, t, h, w)
        else:
            x = self.bn(x)
        # if self.affine:
        #     x = x * self.weight.view(-1, 1, 1, 1)
        #     x = x + self.bias.view(-1, 1, 1, 1)
        return x
