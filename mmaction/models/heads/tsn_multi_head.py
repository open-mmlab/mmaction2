import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from torch.autograd import Function

from ..registry import HEADS
from .base import AvgConsensus, BaseHead


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# The additional parameters are all array (we can set different alpha for
# different factors)
# num_classes (be an array)
# grad_rev (if we use grad_rev)
# alpha_max, weight_type (how we change the alpha)
# alpha_max, weight_type are only used when grad_rev = True


@HEADS.register_module()
class TSNMultiHead(BaseHead):
    """Class head for TSN.

    Args:
        in_channels (int): Number of channels in input feature.
        num_classes (list[int]): Number of classes to be classified.
        grad_rev (list[bool]): Use grad rev for the head.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(
            self,
            in_channels,
            num_classes=[739, 117, 291, 69, 1679, 248],
            grad_rev=[False] + [True] * 5,
            alpha_max=[None] + [1] * 5,
            weight_type=[None] + ['fix'] * 5,
            # Usually, the loss weight of HVULoss should be set by users
            loss_cls=dict(type='HVULoss'),
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.4,
            init_std=0.01,
            **kwargs):

        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        self.func_dict = dict(
            fix=lambda x: 1,
            cosine=lambda x: 0.5 * (1 - np.cos(np.pi * x)),
            cosine_rev=lambda x: 0.5 * (1 - np.cos(np.pi * (1 - x))),
            exp=lambda x: float(2 / (1 + np.exp(-10 * x)) - 1),
            exp_rev=lambda x: float(2 / (1 + np.exp(-10 * (1 - x))) - 1))

        self.num_classes = num_classes
        self.grad_rev = grad_rev
        self.alpha_max = alpha_max
        self.weight_type = weight_type
        assert len(num_classes) == len(grad_rev) == len(alpha_max) == len(
            weight_type)
        self.num_heads = len(num_classes)

        consensus_ = consensus.copy()
        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        fcs = [
            nn.Linear(self.in_channels, num_classes)
            for num_classes in self.num_classes
        ]
        self.fcs = nn.ModuleList(fcs)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for fc in self.fcs:
            normal_init(fc, std=self.init_std)

    # we need runner_info to determine
    def forward(self, x, num_segs, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        progress = None
        if True in self.grad_rev:
            assert 'progress' in kwargs
            progress = kwargs['progress']
            assert progress is not None
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
            # [N * num_segs, in_channels, 1, 1]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        x = self.consensus(x)
        # [N, 1, in_channels, 1, 1]
        x = x.squeeze(1)
        # [N, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        x = x.view(x.size(0), -1)
        # [N, in_channels]
        # before we input the feature into different fc, apply grad_rev
        outputs = []
        for i in range(self.num_heads):
            grad_rev = self.grad_rev[i]
            alpha_max = self.alpha_max[i]
            weight_type = self.weight_type[i]
            head = self.fcs[i]
            if not grad_rev:
                ret = head(x)
            else:
                alpha = float(alpha_max)
                alpha *= self.func_dict[weight_type](progress)
                feat = ReverseLayerF.apply(x, alpha)
                ret = head(feat)
            outputs.append(ret)

        cls_score = torch.cat(outputs, dim=1)
        # [N, num_classes]
        return cls_score
