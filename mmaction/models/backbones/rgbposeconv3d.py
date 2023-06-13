# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.logging import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmaction.registry import MODELS
from .resnet3d_slowfast import ResNet3dPathway


@MODELS.register_module()
class RGBPoseConv3D(BaseModule):
    """RGBPoseConv3D backbone.

    Args:
        pretrained (str): The file path to a pretrained model.
            Defaults to None.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            :math:`\\alpha` in the paper. Defaults to 4.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to :math:`\\beta` in the paper.
            Defaults to 4.
        rgb_detach (bool): Whether to detach the gradients from the pose path.
            Defaults to False.
        pose_detach (bool): Whether to detach the gradients from the rgb path.
            Defaults to False.
        rgb_drop_path (float): The drop rate for dropping the features from
            the pose path. Defaults to 0.
        pose_drop_path (float): The drop rate for dropping the features from
            the rgb path. Defaults to 0.
        rgb_pathway (dict): Configuration of rgb branch. Defaults to
            ``dict(num_stages=4, lateral=True, lateral_infl=1,
            lateral_activate=(0, 0, 1, 1), fusion_kernel=7, base_channels=64,
            conv1_kernel=(1, 7, 7), inflate=(0, 0, 1, 1), with_pool2=False)``.
        pose_pathway (dict): Configuration of pose branch. Defaults to
            ``dict(num_stages=3, stage_blocks=(4, 6, 3), lateral=True,
            lateral_inv=True, lateral_infl=16, lateral_activate=(0, 1, 1),
            fusion_kernel=7, in_channels=17, base_channels=32,
            out_indices=(2, ), conv1_kernel=(1, 7, 7), conv1_stride_s=1,
            conv1_stride_t=1, pool1_stride_s=1, pool1_stride_t=1,
            inflate=(0, 1, 1), spatial_strides=(2, 2, 2),
            temporal_strides=(1, 1, 1), with_pool2=False)``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 speed_ratio: int = 4,
                 channel_ratio: int = 4,
                 rgb_detach: bool = False,
                 pose_detach: bool = False,
                 rgb_drop_path: float = 0,
                 pose_drop_path: float = 0,
                 rgb_pathway: Dict = dict(
                     num_stages=4,
                     lateral=True,
                     lateral_infl=1,
                     lateral_activate=(0, 0, 1, 1),
                     fusion_kernel=7,
                     base_channels=64,
                     conv1_kernel=(1, 7, 7),
                     inflate=(0, 0, 1, 1),
                     with_pool2=False),
                 pose_pathway: Dict = dict(
                     num_stages=3,
                     stage_blocks=(4, 6, 3),
                     lateral=True,
                     lateral_inv=True,
                     lateral_infl=16,
                     lateral_activate=(0, 1, 1),
                     fusion_kernel=7,
                     in_channels=17,
                     base_channels=32,
                     out_indices=(2, ),
                     conv1_kernel=(1, 7, 7),
                     conv1_stride_s=1,
                     conv1_stride_t=1,
                     pool1_stride_s=1,
                     pool1_stride_t=1,
                     inflate=(0, 1, 1),
                     spatial_strides=(2, 2, 2),
                     temporal_strides=(1, 1, 1),
                     dilations=(1, 1, 1),
                     with_pool2=False),
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.pretrained = pretrained
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        if rgb_pathway['lateral']:
            rgb_pathway['speed_ratio'] = speed_ratio
            rgb_pathway['channel_ratio'] = channel_ratio

        if pose_pathway['lateral']:
            pose_pathway['speed_ratio'] = speed_ratio
            pose_pathway['channel_ratio'] = channel_ratio

        self.rgb_path = ResNet3dPathway(**rgb_pathway)
        self.pose_path = ResNet3dPathway(**pose_pathway)
        self.rgb_detach = rgb_detach
        self.pose_detach = pose_detach
        assert 0 <= rgb_drop_path <= 1
        assert 0 <= pose_drop_path <= 1
        self.rgb_drop_path = rgb_drop_path
        self.pose_drop_path = pose_drop_path

    def init_weights(self) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            # Init two branch separately.
            self.rgb_path.init_weights()
            self.pose_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, imgs: torch.Tensor, heatmap_imgs: torch.Tensor) -> tuple:
        """Defines the computation performed at every call.

        Args:
            imgs (torch.Tensor): The input data.
            heatmap_imgs (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input
            samples extracted by the backbone.
        """
        if self.training:
            rgb_drop_path = torch.rand(1) < self.rgb_drop_path
            pose_drop_path = torch.rand(1) < self.pose_drop_path
        else:
            rgb_drop_path, pose_drop_path = False, False
        # We assume base_channel for RGB and Pose are 64 and 32.
        x_rgb = self.rgb_path.conv1(imgs)
        x_rgb = self.rgb_path.maxpool(x_rgb)
        # N x 64 x 8 x 56 x 56
        x_pose = self.pose_path.conv1(heatmap_imgs)
        x_pose = self.pose_path.maxpool(x_pose)

        x_rgb = self.rgb_path.layer1(x_rgb)
        x_rgb = self.rgb_path.layer2(x_rgb)
        x_pose = self.pose_path.layer1(x_pose)

        if hasattr(self.rgb_path, 'layer2_lateral'):
            feat = x_pose.detach() if self.rgb_detach else x_pose
            x_pose_lateral = self.rgb_path.layer2_lateral(feat)
            if rgb_drop_path:
                x_pose_lateral = x_pose_lateral.new_zeros(x_pose_lateral.shape)

        if hasattr(self.pose_path, 'layer1_lateral'):
            feat = x_rgb.detach() if self.pose_detach else x_rgb
            x_rgb_lateral = self.pose_path.layer1_lateral(feat)
            if pose_drop_path:
                x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)

        if hasattr(self.rgb_path, 'layer2_lateral'):
            x_rgb = torch.cat((x_rgb, x_pose_lateral), dim=1)

        if hasattr(self.pose_path, 'layer1_lateral'):
            x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)

        x_rgb = self.rgb_path.layer3(x_rgb)
        x_pose = self.pose_path.layer2(x_pose)

        if hasattr(self.rgb_path, 'layer3_lateral'):
            feat = x_pose.detach() if self.rgb_detach else x_pose
            x_pose_lateral = self.rgb_path.layer3_lateral(feat)
            if rgb_drop_path:
                x_pose_lateral = x_pose_lateral.new_zeros(x_pose_lateral.shape)

        if hasattr(self.pose_path, 'layer2_lateral'):
            feat = x_rgb.detach() if self.pose_detach else x_rgb
            x_rgb_lateral = self.pose_path.layer2_lateral(feat)
            if pose_drop_path:
                x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)

        if hasattr(self.rgb_path, 'layer3_lateral'):
            x_rgb = torch.cat((x_rgb, x_pose_lateral), dim=1)

        if hasattr(self.pose_path, 'layer2_lateral'):
            x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)

        x_rgb = self.rgb_path.layer4(x_rgb)
        x_pose = self.pose_path.layer3(x_pose)

        return x_rgb, x_pose
