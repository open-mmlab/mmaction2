import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import load_checkpoint
from mmcv.utils import print_log

from ...utils import get_root_logger
from ..registry import BACKBONES
from .resnet3d_slowfast import ResNet3dPathway
from .resnet_audio import ResNetAudio


class ResNetAudioPathway(ResNetAudio):
    """A pathway of AVSlowfast based on ResNetAudio.

    Args:
        *args (arguments): Arguments same as :class:``ResNet3d``.
        lateral (bool): Determines whether to enable the lateral connection
            to another pathway. Default: True.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            ``alpha`` in the paper. Default: 32.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to ``beta`` in the paper.
            Default: 2.
        fusion_kernel (int): The kernel size of lateral fusion.
            Default: 9.
        **kwargs (keyword arguments): Keywords arguments for ResNet3d.
    """

    def __init__(self,
                 *args,
                 lateral=True,
                 speed_ratio=32,
                 channel_ratio=2,
                 fusion_kernel=9,
                 **kwargs):
        self.lateral = lateral
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        super().__init__(*args, **kwargs)
        self.inplanes = self.base_channels
        if self.lateral:
            self.conv1_lateral = ConvModule(
                self.inplanes,
                self.inplanes * self.channel_ratio,
                kernel_size=(fusion_kernel, 1),
                stride=(self.speed_ratio, 1),
                padding=((fusion_kernel - 1) // 2, 0),
                bias=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=None)

        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2**i
            self.inplanes = planes * self.block.expansion

            if lateral and i != self.num_stages - 1:
                # no lateral connection needed in final stage
                lateral_name = f'layer{(i + 1)}_lateral'
                setattr(
                    self, lateral_name,
                    ConvModule(
                        self.inplanes,
                        self.inplanes * self.channel_ratio,
                        kernel_size=fusion_kernel,
                        stride=self.speed_ratio,
                        bias=False,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=None))
                self.lateral_connections.append(lateral_name)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        `self.frozen_stages`."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            # Freeze those in the lateral connections as well.
            if (i != len(self.res_layers) and self.lateral):
                # No fusion needed in the final stage
                lateral_name = self.lateral_connections[i - 1]
                conv_lateral = getattr(self, lateral_name)
                conv_lateral.eval()
                for param in conv_lateral.parameters():
                    param.requires_grad = False

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        super().init_weights()
        # Init those in the lateral connections as well.
        for module_name in self.lateral_connections:
            layer = getattr(self, module_name)
            for m in layer.modules():
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    kaiming_init(m)


pathway_cfg = {
    'resnet_audio': ResNetAudioPathway,
    'resnet3d': ResNet3dPathway
    # TODO: BNInceptionPathway
}


def build_pathway(cfg, *args, **kwargs):
    """Build pathway.

    Args:
        cfg (None or dict): cfg should contain:
            - type (str): identify conv layer type.

    Returns:
        nn.Module: Created pathway.
    """
    if not (isinstance(cfg, dict) and 'type' in cfg):
        raise TypeError('cfg must be a dict containing the key "type"')
    cfg_ = cfg.copy()

    pathway_type = cfg_.pop('type')
    if pathway_type not in pathway_cfg:
        raise KeyError(f'Unrecognized pathway type {pathway_type}')
    else:
        pathway_cls = pathway_cfg[pathway_type]
    pathway = pathway_cls(*args, **kwargs, **cfg_)

    return pathway


@BACKBONES.register_module()
class AVResNet3dSlowFast(nn.Module):
    """Audio Visual Slowfast backbone.

    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_

    Args:
        pretrained (str): The file path to a pretrained model.
        resample_rate (int): A large temporal stride ``resample_rate``
            on input frames, corresponding to the :math:`\\tau` in the paper.
            i.e., it processes only one out of ``resample_rate`` frames.
            Default: 16.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            :math:`\\alpha` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to :math:`\\beta` in the paper.
            Default: 8.
        slow_pathway (dict): Configuration of slow branch, should contain
            necessary arguments for building the specific type of pathway
            and:
            type (str): type of backbone the pathway bases on.
            lateral (bool): determine whether to build lateral connection
            for the pathway.Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=True, depth=50, pretrained=None,
                conv1_kernel=(1, 7, 7), dilations=(1, 1, 1, 1),
                conv1_stride_t=1, pool1_stride_t=1, inflate=(0, 0, 1, 1))

        fast_pathway (dict): Configuration of fast branch, similar to
            `slow_pathway`. Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=False, depth=50, pretrained=None, base_channels=8,
                conv1_kernel=(5, 7, 7), conv1_stride_t=1, pool1_stride_t=1)
    """

    def __init__(self,
                 pretrained,
                 resample_rate_fast=8,
                 speed_ratio_fast=8,
                 channel_ratio_fast=8,
                 speed_ratio_audio=32,
                 channel_ratio_audio=2,
                 drop_out_ratio=0.5,
                 slow_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=True,
                     fusion_kernel=5,
                     conv1_kernel=(1, 7, 7),
                     dilations=(1, 1, 1, 1),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                     inflate=(0, 0, 1, 1)),
                 fast_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=False,
                     base_channels=8,
                     conv1_kernel=(5, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1),
                 audio_pathway=dict(
                     type='resnet_audio',
                     depth=50,
                     pretrained=None,
                     strides=(2, 2, 2, 2),
                     lateral=True)):
        super().__init__()
        self.pretrained = pretrained
        self.resample_rate_fast = resample_rate_fast
        self.speed_ratio_fast = speed_ratio_fast
        self.channel_ratio_fast = channel_ratio_fast
        self.speed_ratio_audio = speed_ratio_audio
        self.channel_ratio_audio = channel_ratio_audio
        self.drop_out_ratio = drop_out_ratio

        if slow_pathway['lateral']:
            slow_pathway['speed_ratio'] = speed_ratio_fast
            slow_pathway['channel_ratio'] = channel_ratio_fast
        if audio_pathway['lateral']:
            audio_pathway['speed_ratio'] = speed_ratio_audio
            audio_pathway['channel_ratio'] = channel_ratio_audio
        random.seed(100)
        # set the random seed to avoid different
        # graphs in distributed env
        self.slow_path = build_pathway(slow_pathway)
        self.fast_path = build_pathway(fast_pathway)
        self.audio_path = build_pathway(audio_pathway)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            # Directly load 3D model.
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            # Init three branch seperately.
            self.fast_path.init_weights()
            self.slow_path.init_weights()
            self.audio_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, a):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            a (torch.Tensor): The input mel-spectrogram.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples extracted
                by the backbone.
        """
        use_audio = random.random() > self.drop_out_ratio
        # stem
        x_slow = nn.functional.interpolate(
            x,
            mode='nearest',
            scale_factor=(1.0 / self.resample_rate_fast, 1.0, 1.0))
        x_slow = self.slow_path.conv1(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)

        x_fast = nn.functional.interpolate(
            x,
            mode='nearest',
            scale_factor=(1.0 /
                          (self.resample_rate_fast // self.speed_ratio_fast),
                          1.0, 1.0))
        x_fast = self.fast_path.conv1(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)

        x_audio = a
        x_audio = self.audio_path.conv1(x_audio)
        if self.slow_path.lateral:
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast)
            x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        if use_audio and self.audio_path.lateral:
            x_audio_lateral = self.audio_path.conv1_lateral(x_audio)
            x_audio_lateral = x_audio_lateral.unsqueeze(4)
            # use t-pool rather than t-conv
            x_audio_lateral_pooled = F.adaptive_avg_pool3d(
                x_audio_lateral,
                x_slow.size()[2:])
            x_slow = x_slow + x_audio_lateral_pooled

        # res-stages
        for i, layer_name in enumerate(self.slow_path.res_layers):
            res_layer = getattr(self.slow_path, layer_name)
            x_slow = res_layer(x_slow)
            res_layer_fast = getattr(self.fast_path, layer_name)
            x_fast = res_layer_fast(x_fast)
            res_layer_audio = getattr(self.audio_path, layer_name)
            x_audio = res_layer_audio(x_audio)

            if (i != len(self.slow_path.res_layers) - 1
                    and self.slow_path.lateral and self.audio_path.lateral):
                # No fusion needed in the final stage
                lateral_name = self.slow_path.lateral_connections[i]
                conv_lateral = getattr(self.slow_path, lateral_name)
                x_fast_lateral = conv_lateral(x_fast)
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
                if use_audio:
                    lateral_name = self.audio_path.lateral_connections[i]
                    conv_lateral = getattr(self.audio_path, lateral_name)
                    x_audio_lateral = conv_lateral(x_audio)
                    # NCTF -> NCTHW
                    x_audio_lateral = x_audio_lateral.unsqueeze(4)
                    x_audio_lateral_pooled = F.adaptive_avg_pool3d(
                        x_audio_lateral,
                        x_slow.size()[2:])
                    x_slow = x_slow + x_audio_lateral_pooled
                else:
                    x_audio = torch.zeros_like(x_audio, requires_grad=True)

        out = (x_slow, x_fast, x_audio)

        return out
