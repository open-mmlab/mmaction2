import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.utils import _BatchNorm

from ..builder import BACKBONES
from .resnet3d import Bottleneck3d, ResNet3d


class CSNBottleneck3d(Bottleneck3d):
    """Channel-Separated Bottleneck Block.

    This module is proposed in
    "Video Classification with Channel-Separated Convolutional Networks"
    Link: https://arxiv.org/pdf/1711.11248.pdf

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        bottleneck_mode (str): Determine which ways to factorize a 3D
            bottleneck block using channel-separated convolutional networks.
                If set to 'ip', it will replace the 3x3x3 conv2 layer with a
                1x1x1 traditional convolution and a 3x3x3 depthwise
                convolution, i.e., Interaction-preserved channel-separated
                bottleneck block.
                If set to 'ir', it will replace the 3x3x3 conv2 layer with a
                3x3x3 depthwise convolution, which is derived from preserved
                bottleneck block by removing the extra 1x1x1 convolution,
                i.e., Interaction-reduced channel-separated bottleneck block.
            Default: 'ir'.
        args (position arguments): Position arguments for Bottleneck.
        kwargs (dict, optional): Keyword arguments for Bottleneck.
    """

    def __init__(self,
                 inplanes,
                 planes,
                 *args,
                 bottleneck_mode='ir',
                 **kwargs):
        super(CSNBottleneck3d, self).__init__(inplanes, planes, *args,
                                              **kwargs)
        self.bottleneck_mode = bottleneck_mode
        conv2 = []
        if self.bottleneck_mode == 'ip':
            conv2.append(
                nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False))
        conv2_kernel_size = self.conv2.conv.kernel_size
        conv2_stride = self.conv2.conv.stride
        conv2_padding = self.conv2.conv.padding
        conv2_dilation = self.conv2.conv.dilation
        conv2_bias = bool(self.conv2.conv.bias)
        self.conv2 = ConvModule(
            planes,
            planes,
            conv2_kernel_size,
            stride=conv2_stride,
            padding=conv2_padding,
            dilation=conv2_dilation,
            bias=conv2_bias,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            groups=planes)
        conv2.append(self.conv2)
        self.conv2 = nn.Sequential(*conv2)


@BACKBONES.register_module()
class ResNet3dCSN(ResNet3d):
    """ResNet backbone for CSN.

    Args:
        depth (int): Depth of ResNetCSN, from {18, 34, 50, 101, 152}.
        pretrained (str | None): Name of pretrained model.
        temporal_strides (tuple[int]):
            Temporal strides of residual blocks of each stage.
            Default: (1, 2, 2, 2).
        conv1_kernel (tuple[int]): Kernel size of the first conv layer.
            Default: (3, 7, 7).
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Default: 1.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Default: 1.
        norm_cfg (dict): Config for norm layers. required keys are `type` and
            `requires_grad`.
            Default: dict(type='BN3d', requires_grad=True, eps=1e-3).
        inflate_style (str): `3x1x1` or `3x3x3`. which determines the kernel
            sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x3x3'.
        bottleneck_mode (str): Determine which ways to factorize a 3D
            bottleneck block using channel-separated convolutional networks.
                If set to 'ip', it will replace the 3x3x3 conv2 layer with a
                1x1x1 traditional convolution and a 3x3x3 depthwise
                convolution, i.e., Interaction-preserved channel-separated
                bottleneck block.
                If set to 'ir', it will replace the 3x3x3 conv2 layer with a
                3x3x3 depthwise convolution, which is derived from preserved
                bottleneck block by removing the extra 1x1x1 convolution,
                i.e., Interaction-reduced channel-separated bottleneck block.
            Default: 'ip'.
        kwargs (dict, optional): Key arguments for "make_res_layer".
    """

    def __init__(self,
                 depth,
                 pretrained,
                 temporal_strides=(1, 2, 2, 2),
                 conv1_kernel=(3, 7, 7),
                 conv1_stride_t=1,
                 pool1_stride_t=1,
                 norm_cfg=dict(type='BN3d', requires_grad=True, eps=1e-3),
                 inflate_style='3x3x3',
                 bottleneck_mode='ir',
                 bn_frozen=False,
                 **kwargs):
        self.arch_settings = {
            # 18: (BasicBlock3d, (2, 2, 2, 2)),
            # 34: (BasicBlock3d, (3, 4, 6, 3)),
            50: (CSNBottleneck3d, (3, 4, 6, 3)),
            101: (CSNBottleneck3d, (3, 4, 23, 3)),
            152: (CSNBottleneck3d, (3, 8, 36, 3))
        }
        self.bn_frozen = bn_frozen
        if bottleneck_mode not in ['ip', 'ir']:
            raise ValueError(f'Bottleneck mode must be "ip" or "ir",'
                             f'but got {bottleneck_mode}.')
        super(ResNet3dCSN, self).__init__(
            depth,
            pretrained,
            temporal_strides=temporal_strides,
            conv1_kernel=conv1_kernel,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_t=pool1_stride_t,
            norm_cfg=norm_cfg,
            inflate_style=inflate_style,
            bottleneck_mode=bottleneck_mode,
            **kwargs)

    def train(self, mode=True):
        super(ResNet3d, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                    if self.bn_frozen:
                        for param in m.parameters():
                            param.requires_grad = False
