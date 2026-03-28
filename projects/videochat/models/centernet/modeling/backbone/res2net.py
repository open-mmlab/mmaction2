# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved This
# file is modified from https://github.com/Res2Net/Res2Net-detectron2/blob
# /master/detectron2/modeling/backbone/resnet.py The original file is under
# Apache-2.0 License
import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.layers import (CNNBlockBase, Conv2d, DeformConv,
                               ModulatedDeformConv, ShapeSpec, get_norm)
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN
from torch import nn

from .bifpn import BiFPN
from .fpn_p5 import LastLevelP6P7_P5

__all__ = [
    'ResNetBlockBase',
    'BasicBlock',
    'BottleneckBlock',
    'DeformBottleneckBlock',
    'BasicStem',
    'ResNet',
    'make_stage',
    'build_res2net_backbone',
]

ResNetBlockBase = CNNBlockBase
"""
Alias for backward compatibiltiy.
"""


class BasicBlock(CNNBlockBase):
    """The basic residual block for ResNet-18 and ResNet-34, with two 3x3 conv
    layers and a projection shortcut if needed."""

    def __init__(self, in_channels, out_channels, *, stride=1, norm='BN'):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock(CNNBlockBase):
    """The standard bottle2neck residual block used by Res2Net-50, 101 and
    152."""

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm='BN',
        stride_in_1x1=False,
        dilation=1,
        basewidth=26,
        scale=4,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False),
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    norm=get_norm(norm, out_channels),
                ))
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t
        # implementations have stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        width = bottleneck_channels // scale

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if self.in_channels != self.out_channels and stride_3x3 != 2:
            self.pool = nn.AvgPool2d(
                kernel_size=3, stride=stride_3x3, padding=1)

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(
                    width,
                    width,
                    kernel_size=3,
                    stride=stride_3x3,
                    padding=1 * dilation,
                    bias=False,
                    groups=num_groups,
                    dilation=dilation,
                ))
            bns.append(get_norm(norm, width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        self.scale = scale
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride_3x3 = stride_3x3
        for layer in [self.conv1, self.conv3]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)
        if self.shortcut is not None:
            for layer in self.shortcut.modules():
                if isinstance(layer, Conv2d):
                    weight_init.c2_msra_fill(layer)

        for layer in self.convs:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity. See Sec 5.1 in
        # "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour": "For
        # BN layers, the learnable scaling coefficient γ is initialized to
        # be 1, except for each residual block's last BN where γ is
        # initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0) Add it as an option
        # when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.in_channels != self.out_channels:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = F.relu_(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stride_3x3 == 1:
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stride_3x3 == 2:
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class DeformBottleneckBlock(ResNetBlockBase):
    """Not implemented for res2net yet.

    Similar to :class:`BottleneckBlock`, but with deformable conv in the 3x3
    convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm='BN',
        stride_in_1x1=False,
        dilation=1,
        deform_modulated=False,
        deform_num_groups=1,
        basewidth=26,
        scale=4,
    ):
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated

        if in_channels != out_channels:
            # self.shortcut = Conv2d(
            #     in_channels,
            #     out_channels,
            #     kernel_size=1,
            #     stride=stride,
            #     bias=False,
            #     norm=get_norm(norm, out_channels),
            # )
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False),
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    norm=get_norm(norm, out_channels),
                ))
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        width = bottleneck_channels // scale

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if self.in_channels != self.out_channels and stride_3x3 != 2:
            self.pool = nn.AvgPool2d(
                kernel_size=3, stride=stride_3x3, padding=1)

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size *
            # kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        # self.conv2_offset = Conv2d(
        #     bottleneck_channels,
        #     offset_channels * deform_num_groups,
        #     kernel_size=3,
        #     stride=stride_3x3,
        #     padding=1 * dilation,
        #     dilation=dilation,
        # )
        # self.conv2 = deform_conv_op(
        #     bottleneck_channels,
        #     bottleneck_channels,
        #     kernel_size=3,
        #     stride=stride_3x3,
        #     padding=1 * dilation,
        #     bias=False,
        #     groups=num_groups,
        #     dilation=dilation,
        #     deformable_groups=deform_num_groups,
        #     norm=get_norm(norm, bottleneck_channels),
        # )

        conv2_offsets = []
        convs = []
        bns = []
        for i in range(self.nums):
            conv2_offsets.append(
                Conv2d(
                    width,
                    offset_channels * deform_num_groups,
                    kernel_size=3,
                    stride=stride_3x3,
                    padding=1 * dilation,
                    bias=False,
                    groups=num_groups,
                    dilation=dilation,
                ))
            convs.append(
                deform_conv_op(
                    width,
                    width,
                    kernel_size=3,
                    stride=stride_3x3,
                    padding=1 * dilation,
                    bias=False,
                    groups=num_groups,
                    dilation=dilation,
                    deformable_groups=deform_num_groups,
                ))
            bns.append(get_norm(norm, width))
        self.conv2_offsets = nn.ModuleList(conv2_offsets)
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        self.scale = scale
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride_3x3 = stride_3x3
        # for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
        #     if layer is not None:  # shortcut can be None
        #         weight_init.c2_msra_fill(layer)

        # nn.init.constant_(self.conv2_offset.weight, 0)
        # nn.init.constant_(self.conv2_offset.bias, 0)
        for layer in [self.conv1, self.conv3]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)
        if self.shortcut is not None:
            for layer in self.shortcut.modules():
                if isinstance(layer, Conv2d):
                    weight_init.c2_msra_fill(layer)

        for layer in self.convs:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        for layer in self.conv2_offsets:
            if layer.weight is not None:
                nn.init.constant_(layer.weight, 0)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        # if self.deform_modulated:
        #     offset_mask = self.conv2_offset(out)
        #     offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        #     offset = torch.cat((offset_x, offset_y), dim=1)
        #     mask = mask.sigmoid()
        #     out = self.conv2(out, offset, mask)
        # else:
        #     offset = self.conv2_offset(out)
        #     out = self.conv2(out, offset)
        # out = F.relu_(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.in_channels != self.out_channels:
                sp = spx[i].contiguous()
            else:
                sp = sp + spx[i].contiguous()

            # sp = self.convs[i](sp)
            if self.deform_modulated:
                offset_mask = self.conv2_offsets[i](sp)
                offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
                offset = torch.cat((offset_x, offset_y), dim=1)
                mask = mask.sigmoid()
                sp = self.convs[i](sp, offset, mask)
            else:
                offset = self.conv2_offsets[i](sp)
                sp = self.convs[i](sp, offset)
            sp = F.relu_(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stride_3x3 == 1:
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stride_3x3 == 2:
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


def make_stage(block_class, num_blocks, first_stride, *, in_channels,
               out_channels, **kwargs):
    """Create a list of blocks just like those in a ResNet stage.

    Args: block_class (type): a subclass of ResNetBlockBase num_blocks (
    int): first_stride (int): the stride of the first block. The other
    blocks will have stride=1. in_channels (int): input channels of the
    entire stage. out_channels (int): output channels of **every block** in
    the stage. kwargs: other arguments passed to the constructor of every
    block. Returns: list[nn.Module]: a list of block module.
    """
    assert 'stride' not in kwargs, 'Stride of blocks in make_stage ' \
                                   'cannot be changed. '
    blocks = []
    for i in range(num_blocks):
        blocks.append(
            block_class(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=first_stride if i == 0 else 1,
                **kwargs,
            ))
        in_channels = out_channels
    return blocks


class BasicStem(CNNBlockBase):
    """The standard ResNet stem (layers before the first residual block)."""

    def __init__(self, in_channels=3, out_channels=64, norm='BN'):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            get_norm(norm, 32),
            nn.ReLU(inplace=True),
            Conv2d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            get_norm(norm, 32),
            nn.ReLU(inplace=True),
            Conv2d(
                32,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.bn1 = get_norm(norm, out_channels)

        for layer in self.conv1:
            if isinstance(layer, Conv2d):
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class ResNet(Backbone):

    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args: stem (nn.Module): a stem module stages (list[list[
        CNNBlockBase]]): several (typically 4) stages, each contains
        multiple :class:`CNNBlockBase`. num_classes (None or int): if None,
        will not perform classification. Otherwise, will create a linear
        layer. out_features (list[str]): name of the layers whose outputs
        should be returned in forward. Can be anything in "stem", "linear",
        or "res2" ... If None, will return the output of the last layer.
        """
        super(ResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {'stem': current_stride}
        self._out_feature_channels = {'stem': self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = 'res' + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks]))
            self._out_feature_channels[name] = curr_channels = blocks[
                -1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet
            # in 1 Hour": "The 1000-way fully-connected layer is initialized
            # by drawing weights from a zero-mean Gaussian with standard
            # deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = 'linear'

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, 'Available children: {}'.format(
                ', '.join(children))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if 'stem' in self._out_features:
            outputs['stem'] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if 'linear' in self._out_features:
                outputs['linear'] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name])
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """Freeze the first several stages of the ResNet.

        Commonly used in
        fine-tuning.
        Args:
            freeze_at (int): number of stem and stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                the first stage, etc.
        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, (stage, _) in enumerate(self.stages_and_names, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self


@BACKBONE_REGISTRY.register()
def build_res2net_backbone(cfg, input_shape):
    """Create a Res2Net instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    depth = cfg.MODEL.RESNETS.DEPTH
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    scale = 4
    bottleneck_channels = num_groups * width_per_group * scale
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {
        1, 2
    }, 'res5_dilation cannot be {}.'.format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, \
            'Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34'
        assert not any(
            deform_on_per_stage
        ), 'MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34'
        assert res5_dilation == 1, \
            'Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34'
        assert num_groups == 1, \
            'Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34'

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{
        'res2': 2,
        'res3': 3,
        'res4': 4,
        'res5': 5
    }[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5
                                         and dilation == 2) else 2
        stage_kargs = {
            'num_blocks': num_blocks_per_stage[idx],
            'first_stride': first_stride,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'norm': norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs['block_class'] = BasicBlock
        else:
            stage_kargs['bottleneck_channels'] = bottleneck_channels
            stage_kargs['stride_in_1x1'] = stride_in_1x1
            stage_kargs['dilation'] = dilation
            stage_kargs['num_groups'] = num_groups
            stage_kargs['scale'] = scale

            if deform_on_per_stage[idx]:
                stage_kargs['block_class'] = DeformBottleneckBlock
                stage_kargs['deform_modulated'] = deform_modulated
                stage_kargs['deform_num_groups'] = deform_num_groups
            else:
                stage_kargs['block_class'] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features).freeze(freeze_at)


@BACKBONE_REGISTRY.register()
def build_p67_res2net_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns: backbone (Backbone): backbone module, must be a subclass of
    :class:`Backbone`.
    """
    bottom_up = build_res2net_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_res2net_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns: backbone (Backbone): backbone module, must be a subclass of
    :class:`Backbone`.
    """
    bottom_up = build_res2net_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    backbone = BiFPN(
        cfg=cfg,
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=cfg.MODEL.BIFPN.OUT_CHANNELS,
        norm=cfg.MODEL.BIFPN.NORM,
        num_levels=cfg.MODEL.BIFPN.NUM_LEVELS,
        num_bifpn=cfg.MODEL.BIFPN.NUM_BIFPN,
        separable_conv=cfg.MODEL.BIFPN.SEPARABLE_CONV,
    )
    return backbone
