# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmaction.registry import MODELS

try:
    from mmdet.registry import MODELS as MMDET_MODELS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

# Note: All these heads take 5D Tensors as input (N, C, T, H, W)


@MODELS.register_module()
class ACRNHead(nn.Module):
    """ACRN Head: Tile + 1x1 convolution + 3x3 convolution.

    This module is proposed in
    `Actor-Centric Relation Network
    <https://arxiv.org/abs/1807.10982>`_

    Args:
        in_channels (int): The input channel.
        out_channels (int): The output channel.
        stride (int): The spatial stride.
        num_convs (int): The number of 3x3 convolutions in ACRNHead.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        kwargs (dict): Other new arguments, to be compatible with MMDet update.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 num_convs=1,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 **kwargs):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        assert num_convs >= 1
        self.conv2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        convs = []
        for _ in range(num_convs - 1):
            conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

    def init_weights(self, **kwargs):
        """Weight Initialization for ACRNHead."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def forward(self, x, feat, rois, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The extracted RoI feature.
            feat (torch.Tensor): The context feature.
            rois (torch.Tensor): The regions of interest.

        Returns:
            torch.Tensor: The RoI features that have interacted with context
                feature.
        """
        # We use max pooling by default
        x = self.max_pool(x)

        h, w = feat.shape[-2:]
        x_tile = x.repeat(1, 1, 1, h, w)

        roi_inds = rois[:, 0].type(torch.long)
        roi_gfeat = feat[roi_inds]

        new_feat = torch.cat([x_tile, roi_gfeat], dim=1)
        new_feat = self.conv1(new_feat)
        new_feat = self.conv2(new_feat)

        for conv in self.convs:
            new_feat = conv(new_feat)

        return new_feat


if mmdet_imported:
    MMDET_MODELS.register_module()(ACRNHead)
