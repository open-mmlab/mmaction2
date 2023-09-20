# Modified from https://github.com/rwightman/efficientdet-pytorch/blob
# /master/effdet/efficientdet.py The original file is under Apache-2.0 License
import math
from collections import OrderedDict

import torch
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.layers.batch_norm import get_norm
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from torch import nn

from .dlafpn import dla34


def get_fpn_config(base_reduction=8):
    """BiFPN config with sum."""
    p = {
        'nodes': [
            {
                'reduction': base_reduction << 3,
                'inputs_offsets': [3, 4]
            },
            {
                'reduction': base_reduction << 2,
                'inputs_offsets': [2, 5]
            },
            {
                'reduction': base_reduction << 1,
                'inputs_offsets': [1, 6]
            },
            {
                'reduction': base_reduction,
                'inputs_offsets': [0, 7]
            },
            {
                'reduction': base_reduction << 1,
                'inputs_offsets': [1, 7, 8]
            },
            {
                'reduction': base_reduction << 2,
                'inputs_offsets': [2, 6, 9]
            },
            {
                'reduction': base_reduction << 3,
                'inputs_offsets': [3, 5, 10]
            },
            {
                'reduction': base_reduction << 4,
                'inputs_offsets': [4, 11]
            },
        ],
        'weight_method':
        'fastattn',
    }
    return p


def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):

    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


class SequentialAppend(nn.Sequential):

    def __init__(self, *args):
        super(SequentialAppend, self).__init__(*args)

    def forward(self, x):
        for module in self:
            x.append(module(x))
        return x


class SequentialAppendLast(nn.Sequential):

    def __init__(self, *args):
        super(SequentialAppendLast, self).__init__(*args)

    # def forward(self, x: List[torch.Tensor]):
    def forward(self, x):
        for module in self:
            x.append(module(x[-1]))
        return x


class ConvBnAct2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 padding='',
                 bias=False,
                 norm='',
                 act_layer=Swish):
        super(ConvBnAct2d, self).__init__()
        # self.conv = create_conv2d( in_channels, out_channels, kernel_size,
        # stride=stride, dilation=dilation, padding=padding, bias=bias)
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=(norm == ''))
        self.bn = get_norm(norm, out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):
    """Separable Conv."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='',
                 bias=False,
                 channel_multiplier=1.0,
                 pw_kernel_size=1,
                 act_layer=Swish,
                 norm=''):
        super(SeparableConv2d, self).__init__()

        # self.conv_dw = create_conv2d( in_channels, int(in_channels *
        # channel_multiplier), kernel_size, stride=stride,
        # dilation=dilation, padding=padding, depthwise=True)

        self.conv_dw = Conv2d(
            in_channels,
            int(in_channels * channel_multiplier),
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias,
            groups=out_channels)
        # print('conv_dw', kernel_size, stride) self.conv_pw =
        # create_conv2d( int(in_channels * channel_multiplier),
        # out_channels, pw_kernel_size, padding=padding, bias=bias)

        self.conv_pw = Conv2d(
            int(in_channels * channel_multiplier),
            out_channels,
            kernel_size=pw_kernel_size,
            padding=pw_kernel_size // 2,
            bias=(norm == ''))
        # print('conv_pw', pw_kernel_size)

        self.bn = get_norm(norm, out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResampleFeatureMap(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction_ratio=1.,
                 pad_type='',
                 pooling_type='max',
                 norm='',
                 apply_bn=False,
                 conv_after_downsample=False,
                 redundant_bias=False):
        super(ResampleFeatureMap, self).__init__()
        pooling_type = pooling_type or 'max'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.conv_after_downsample = conv_after_downsample

        conv = None
        if in_channels != out_channels:
            conv = ConvBnAct2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=pad_type,
                norm=norm if apply_bn else '',
                bias=not apply_bn or redundant_bias,
                act_layer=None)

        if reduction_ratio > 1:
            stride_size = int(reduction_ratio)
            if conv is not None and not self.conv_after_downsample:
                self.add_module('conv', conv)
            self.add_module(
                'downsample',
                # create_pool2d( pooling_type, kernel_size=stride_size + 1,
                # stride=stride_size, padding=pad_type) nn.MaxPool2d(
                # kernel_size=stride_size + 1, stride=stride_size,
                # padding=pad_type)
                nn.MaxPool2d(kernel_size=stride_size, stride=stride_size))
            if conv is not None and self.conv_after_downsample:
                self.add_module('conv', conv)
        else:
            if conv is not None:
                self.add_module('conv', conv)
            if reduction_ratio < 1:
                scale = int(1 // reduction_ratio)
                self.add_module('upsample',
                                nn.UpsamplingNearest2d(scale_factor=scale))


class FpnCombine(nn.Module):

    def __init__(self,
                 feature_info,
                 fpn_config,
                 fpn_channels,
                 inputs_offsets,
                 target_reduction,
                 pad_type='',
                 pooling_type='max',
                 norm='',
                 apply_bn_for_resampling=False,
                 conv_after_downsample=False,
                 redundant_bias=False,
                 weight_method='attn'):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            in_channels = fpn_channels
            if offset < len(feature_info):
                in_channels = feature_info[offset]['num_chs']
                input_reduction = feature_info[offset]['reduction']
            else:
                node_idx = offset - len(feature_info)
                # print('node_idx, len', node_idx, len(fpn_config['nodes']))
                input_reduction = fpn_config['nodes'][node_idx]['reduction']
            reduction_ratio = target_reduction / input_reduction
            self.resample[str(offset)] = ResampleFeatureMap(
                in_channels,
                fpn_channels,
                reduction_ratio=reduction_ratio,
                pad_type=pad_type,
                pooling_type=pooling_type,
                norm=norm,
                apply_bn=apply_bn_for_resampling,
                conv_after_downsample=conv_after_downsample,
                redundant_bias=redundant_bias)

        if weight_method == 'attn' or weight_method == 'fastattn':
            # WSM
            self.edge_weights = nn.Parameter(
                torch.ones(len(inputs_offsets)), requires_grad=True)
        else:
            self.edge_weights = None

    def forward(self, x):
        dtype = x[0].dtype
        nodes = []
        for offset in self.inputs_offsets:
            input_node = x[offset]
            input_node = self.resample[str(offset)](input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(
                self.edge_weights.type(dtype), dim=0)
            x = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.type(dtype))
            weights_sum = torch.sum(edge_weights)
            x = torch.stack([(nodes[i] * edge_weights[i]) /
                             (weights_sum + 0.0001)
                             for i in range(len(nodes))],
                            dim=-1)
        elif self.weight_method == 'sum':
            x = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(
                self.weight_method))
        x = torch.sum(x, dim=-1)
        return x


class BiFpnLayer(nn.Module):

    def __init__(self,
                 feature_info,
                 fpn_config,
                 fpn_channels,
                 num_levels=5,
                 pad_type='',
                 pooling_type='max',
                 norm='',
                 act_layer=Swish,
                 apply_bn_for_resampling=False,
                 conv_after_downsample=True,
                 conv_bn_relu_pattern=False,
                 separable_conv=True,
                 redundant_bias=False):
        super(BiFpnLayer, self).__init__()
        self.fpn_config = fpn_config
        self.num_levels = num_levels
        self.conv_bn_relu_pattern = False

        self.feature_info = []
        self.fnode = SequentialAppend()
        for i, fnode_cfg in enumerate(fpn_config['nodes']):
            # logging.debug('fnode {} : {}'.format(i, fnode_cfg))
            # print('fnode {} : {}'.format(i, fnode_cfg))
            fnode_layers = OrderedDict()

            # combine features
            reduction = fnode_cfg['reduction']
            fnode_layers['combine'] = FpnCombine(
                feature_info,
                fpn_config,
                fpn_channels,
                fnode_cfg['inputs_offsets'],
                target_reduction=reduction,
                pad_type=pad_type,
                pooling_type=pooling_type,
                norm=norm,
                apply_bn_for_resampling=apply_bn_for_resampling,
                conv_after_downsample=conv_after_downsample,
                redundant_bias=redundant_bias,
                weight_method=fpn_config['weight_method'])
            self.feature_info.append(
                dict(num_chs=fpn_channels, reduction=reduction))

            # after combine ops
            after_combine = OrderedDict()
            if not conv_bn_relu_pattern:
                after_combine['act'] = act_layer(inplace=True)
                conv_bias = redundant_bias
                conv_act = None
            else:
                conv_bias = False
                conv_act = act_layer
            conv_kwargs = dict(
                in_channels=fpn_channels,
                out_channels=fpn_channels,
                kernel_size=3,
                padding=pad_type,
                bias=conv_bias,
                norm=norm,
                act_layer=conv_act)
            after_combine['conv'] = SeparableConv2d(
                **conv_kwargs) if separable_conv else ConvBnAct2d(
                    **conv_kwargs)
            fnode_layers['after_combine'] = nn.Sequential(after_combine)

            self.fnode.add_module(str(i), nn.Sequential(fnode_layers))

        self.feature_info = self.feature_info[-num_levels::]

    def forward(self, x):
        x = self.fnode(x)
        return x[-self.num_levels::]


class BiFPN(Backbone):

    def __init__(
        self,
        cfg,
        bottom_up,
        in_features,
        out_channels,
        norm='',
        num_levels=5,
        num_bifpn=4,
        separable_conv=False,
    ):
        super(BiFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g.
        # ResNet)
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        self.num_levels = num_levels
        self.num_bifpn = num_bifpn
        self.bottom_up = bottom_up
        self.in_features = in_features
        self._size_divisibility = 128
        levels = [int(math.log2(s)) for s in in_strides]
        self._out_feature_strides = {
            'p{}'.format(int(math.log2(s))): s
            for s in in_strides
        }
        if len(in_features) < num_levels:
            for ll in range(num_levels - len(in_features)):
                s = ll + levels[-1]
                self._out_feature_strides['p{}'.format(s + 1)] = 2**(s + 1)
        self._out_features = list(sorted(self._out_feature_strides.keys()))
        self._out_feature_channels = {
            k: out_channels
            for k in self._out_features
        }

        # print('self._out_feature_strides', self._out_feature_strides)
        # print('self._out_feature_channels', self._out_feature_channels)

        feature_info = [{
            'num_chs': in_channels[level],
            'reduction': in_strides[level]
        } for level in range(len(self.in_features))]
        # self.config = config
        fpn_config = get_fpn_config()
        self.resample = SequentialAppendLast()
        for level in range(num_levels):
            if level < len(feature_info):
                in_chs = in_channels[level]  # feature_info[level]['num_chs']
                reduction = in_strides[
                    level]  # feature_info[level]['reduction']
            else:
                # Adds a coarser level by downsampling the last feature map
                reduction_ratio = 2
                self.resample.add_module(
                    str(level),
                    ResampleFeatureMap(
                        in_channels=in_chs,
                        out_channels=out_channels,
                        pad_type='same',
                        pooling_type=None,
                        norm=norm,
                        reduction_ratio=reduction_ratio,
                        apply_bn=True,
                        conv_after_downsample=False,
                        redundant_bias=False,
                    ))
                in_chs = out_channels
                reduction = int(reduction * reduction_ratio)
                feature_info.append(dict(num_chs=in_chs, reduction=reduction))

        self.cell = nn.Sequential()
        for rep in range(self.num_bifpn):
            # logging.debug('building cell {}'.format(rep))
            # print('building cell {}'.format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                fpn_config=fpn_config,
                fpn_channels=out_channels,
                num_levels=self.num_levels,
                pad_type='same',
                pooling_type=None,
                norm=norm,
                act_layer=Swish,
                separable_conv=separable_conv,
                apply_bn_for_resampling=True,
                conv_after_downsample=False,
                conv_bn_relu_pattern=False,
                redundant_bias=False,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info
        # import pdb; pdb.set_trace()

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        # print('input shapes', x.shape)
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features]
        assert len(self.resample) == self.num_levels - len(x)
        x = self.resample(x)
        # shapes = [xx.shape for xx in x]
        # print('resample shapes', shapes)
        x = self.cell(x)
        out = {f: xx for f, xx in zip(self._out_features, x)}
        # import pdb; pdb.set_trace()
        return out


@BACKBONE_REGISTRY.register()
def build_resnet_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns: backbone (Backbone): backbone module, must be a subclass of
    :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
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


@BACKBONE_REGISTRY.register()
def build_p37_dla_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args: cfg: a detectron2 CfgNode Returns: backbone (Backbone): backbone
    module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = dla34(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    assert cfg.MODEL.BIFPN.NUM_LEVELS == 5

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
