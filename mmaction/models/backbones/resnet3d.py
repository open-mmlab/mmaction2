# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple, Union

import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, NonLocal3d, build_activation_layer
from mmengine.logging import MMLogger
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch import Tensor, nn
from torch.nn.modules.utils import _ntuple, _triple

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, OptConfigType


class BasicBlock3d(nn.Module):
    """BasicBlock 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer.
            Defaults to 1.
        temporal_stride (int): Temporal stride in the conv3d layer.
            Defaults to 1.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        downsample (nn.Module or None): Downsample layer. Defaults to None.
        style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``, the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``pytorch``.
        inflate (bool): Whether to inflate kernel. Defaults to True.
        non_local (bool): Determine whether to apply non-local module in this
            block. Defaults to False.
        non_local_cfg (dict or ConfigDict): Config for non-local module.
            Defaults to ``dict()``.
        conv_cfg (dict or ConfigDict): Config dict for convolution layer.
            Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict or ConfigDict): Config for norm layers.
            Required keys are ``type``. Defaults to ``dict(type='BN3d')``.
        act_cfg (dict or ConfigDict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 spatial_stride: int = 1,
                 temporal_stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 style: str = 'pytorch',
                 inflate: bool = True,
                 non_local: bool = False,
                 non_local_cfg: ConfigType = dict(),
                 conv_cfg: ConfigType = dict(type='Conv3d'),
                 norm_cfg: ConfigType = dict(type='BN3d'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 with_cp: bool = False,
                 **kwargs) -> None:
        super().__init__()
        assert style in ['pytorch', 'caffe']
        # make sure that only ``inflate_style`` is passed into kwargs
        assert set(kwargs).issubset(['inflate_style'])

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        self.conv1_stride_s = spatial_stride
        self.conv2_stride_s = 1
        self.conv1_stride_t = temporal_stride
        self.conv2_stride_t = 1

        if self.inflate:
            conv1_kernel_size = (3, 3, 3)
            conv1_padding = (1, dilation, dilation)
            conv2_kernel_size = (3, 3, 3)
            conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 3, 3)
            conv1_padding = (0, dilation, dilation)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=conv1_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes * self.expansion,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s),
            padding=conv2_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

        if self.non_local:
            self.non_local_block = NonLocal3d(self.conv2.norm.num_features,
                                              **self.non_local_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        if self.non_local:
            out = self.non_local_block(out)

        return out


class Bottleneck3d(nn.Module):
    """Bottleneck 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer.
            Defaults to 1.
        temporal_stride (int): Temporal stride in the conv3d layer.
            Defaults to 1.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        downsample (nn.Module, optional): Downsample layer. Defaults to None.
        style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``, the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``pytorch``.
        inflate (bool): Whether to inflate kernel. Defaults to True.
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Defaults to ``3x1x1``.
        non_local (bool): Determine whether to apply non-local module in this
            block. Defaults to False.
        non_local_cfg (dict or ConfigDict): Config for non-local module.
            Defaults to ``dict()``.
        conv_cfg (dict or ConfigDict): Config dict for convolution layer.
            Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict or ConfigDict): Config for norm layers. required
            keys are ``type``. Defaults to ``dict(type='BN3d')``.
        act_cfg (dict or ConfigDict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 spatial_stride: int = 1,
                 temporal_stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 style: str = 'pytorch',
                 inflate: bool = True,
                 inflate_style: str = '3x1x1',
                 non_local: bool = False,
                 non_local_cfg: ConfigType = dict(),
                 conv_cfg: ConfigType = dict(type='Conv3d'),
                 norm_cfg: ConfigType = dict(type='BN3d'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 with_cp: bool = False) -> None:
        super().__init__()
        assert style in ['pytorch', 'caffe']
        assert inflate_style in ['3x1x1', '3x3x3']

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        if self.style == 'pytorch':
            self.conv1_stride_s = 1
            self.conv2_stride_s = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride_s = spatial_stride
            self.conv2_stride_s = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1

        if self.inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=conv1_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s),
            padding=conv2_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            # No activation in the third ConvModule for bottleneck
            act_cfg=None)

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

        if self.non_local:
            self.non_local_block = NonLocal3d(self.conv3.norm.num_features,
                                              **self.non_local_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        if self.non_local:
            out = self.non_local_block(out)

        return out


@MODELS.register_module()
class ResNet3d(nn.Module):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from
            {``18``, ``34``, ``50``, ``101``, ``152``}.
        pretrained (str, optional): Name of pretrained model. Defaults to None.
        stage_blocks (tuple, optional): Set number of stages for each res
            layer. Defaults to None.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Defaults to True.
        in_channels (int): Channel num of input features. Defaults to 3.
        base_channels (int): Channel num of stem output features.
            Defaults to 64.
        out_indices (Sequence[int]): Indices of output feature.
            Defaults to ```(3, )``.
        num_stages (int): Resnet stages. Defaults to 4.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
            Defaults to ``(1, 2, 2, 2)``.
        temporal_strides (Sequence[int]):
            Temporal strides of residual blocks of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        dilations (Sequence[int]): Dilation of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Defaults to ``(3, 7, 7)``.
        conv1_stride_s (int): Spatial stride of the first conv layer.
            Defaults to 2.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Defaults to 1.
        pool1_stride_s (int): Spatial stride of the first pooling layer.
            Defaults to 2.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Defaults to 1.
        with_pool2 (bool): Whether to use pool2. Defaults to True.
        style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``, the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``pytorch``.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Defaults to -1.
        inflate (Sequence[int]): Inflate Dims of each block.
            Defaults to ``(1, 1, 1, 1)``.
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Defaults to ``3x1x1``.
        conv_cfg (dict or ConfigDict): Config for conv layers.
            Required keys are ``type``. Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict or ConfigDict): Config for norm layers.
            Required keys are ``type`` and ``requires_grad``.
            Defaults to ``dict(type='BN3d', requires_grad=True)``.
        act_cfg (dict or ConfigDict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU', inplace=True)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (``mean`` and ``var``). Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        non_local (Sequence[int]): Determine whether to apply non-local module
            in the corresponding block of each stages.
            Defaults to ``(0, 0, 0, 0)``.
        non_local_cfg (dict or ConfigDict): Config for non-local module.
            Defaults to ``dict()``.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Defaults to True.
    """

    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int,
                 pretrained: Optional[str] = None,
                 stage_blocks: Optional[Tuple] = None,
                 pretrained2d: bool = True,
                 in_channels: int = 3,
                 num_stages: int = 4,
                 base_channels: int = 64,
                 out_indices: Sequence[int] = (3, ),
                 spatial_strides: Sequence[int] = (1, 2, 2, 2),
                 temporal_strides: Sequence[int] = (1, 1, 1, 1),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 conv1_kernel: Sequence[int] = (3, 7, 7),
                 conv1_stride_s: int = 2,
                 conv1_stride_t: int = 1,
                 pool1_stride_s: int = 2,
                 pool1_stride_t: int = 1,
                 with_pool1: bool = True,
                 with_pool2: bool = True,
                 style: str = 'pytorch',
                 frozen_stages: int = -1,
                 inflate: Sequence[int] = (1, 1, 1, 1),
                 inflate_style: str = '3x1x1',
                 conv_cfg: ConfigType = dict(type='Conv3d'),
                 norm_cfg: ConfigType = dict(type='BN3d', requires_grad=True),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 non_local: Sequence[int] = (0, 0, 0, 0),
                 non_local_cfg: ConfigType = dict(),
                 zero_init_residual: bool = True,
                 **kwargs) -> None:
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.stage_blocks = stage_blocks
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(
            dilations) == num_stages
        if self.stage_blocks is not None:
            assert len(self.stage_blocks) == num_stages

        self.conv1_kernel = conv1_kernel
        self.conv1_stride_s = conv1_stride_s
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_s = pool1_stride_s
        self.pool1_stride_t = pool1_stride_t
        self.with_pool1 = with_pool1
        self.with_pool2 = with_pool2
        self.style = style
        self.frozen_stages = frozen_stages
        self.stage_inflations = _ntuple(num_stages)(inflate)
        self.non_local_stages = _ntuple(num_stages)(non_local)
        self.inflate_style = inflate_style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]

        if self.stage_blocks is None:
            self.stage_blocks = stage_blocks[:num_stages]

        self.inplanes = self.base_channels

        self.non_local_cfg = non_local_cfg

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                non_local=self.non_local_stages[i],
                non_local_cfg=self.non_local_cfg,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                with_cp=with_cp,
                **kwargs)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * self.base_channels * 2**(
            len(self.stage_blocks) - 1)

    @staticmethod
    def make_res_layer(block: nn.Module,
                       inplanes: int,
                       planes: int,
                       blocks: int,
                       spatial_stride: Union[int, Sequence[int]] = 1,
                       temporal_stride: Union[int, Sequence[int]] = 1,
                       dilation: int = 1,
                       style: str = 'pytorch',
                       inflate: Union[int, Sequence[int]] = 1,
                       inflate_style: str = '3x1x1',
                       non_local: Union[int, Sequence[int]] = 0,
                       non_local_cfg: ConfigType = dict(),
                       norm_cfg: OptConfigType = None,
                       act_cfg: OptConfigType = None,
                       conv_cfg: OptConfigType = None,
                       with_cp: Optional[bool] = False,
                       **kwargs) -> nn.Module:
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature
                in each block.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides in
                residual and conv layers. Defaults to 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Defaults to 1.
            dilation (int): Spacing between kernel elements. Defaults to 1.
            style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``,
                the stride-two layer is the 3x3 conv layer, otherwise
                the stride-two layer is the first 1x1 conv layer.
                Default: ``pytorch``.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Defaults to 1.
            inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines
                the kernel sizes and padding strides for conv1 and conv2
                in each block. Default: ``3x1x1``.
            non_local (int | Sequence[int]): Determine whether to apply
                non-local module in the corresponding block of each stages.
                Defaults to 0.
            non_local_cfg (dict): Config for non-local module.
                Defaults to ``dict()``.
            conv_cfg (dict or ConfigDict, optional): Config for conv layers.
                Defaults to None.
            norm_cfg (dict or ConfigDict, optional): Config for norm layers.
                Defaults to None.
            act_cfg (dict or ConfigDict, optional): Config for activate layers.
                Defaults to None.
            with_cp (bool, optional): Use checkpoint or not. Using checkpoint
                will save some memory while slowing down the training speed.
                Defaults to False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks
        non_local = non_local if not isinstance(
            non_local, int) else (non_local, ) * blocks
        assert len(inflate) == blocks and len(non_local) == blocks
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                style=style,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp,
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    style=style,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                    **kwargs))

        return nn.Sequential(*layers)

    @staticmethod
    def _inflate_conv_params(conv3d: nn.Module, state_dict_2d: OrderedDict,
                             module_name_2d: str,
                             inflated_param_names: List[str]) -> None:
        """Inflate a conv module from 2d to 3d.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the
                2d model.
            inflated_param_names (List[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + '.weight'

        conv2d_weight = state_dict_2d[weight_2d_name]
        kernel_t = conv3d.weight.data.shape[2]

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
            conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    @staticmethod
    def _inflate_bn_params(bn3d: nn.Module, state_dict_2d: OrderedDict,
                           module_name_2d: str,
                           inflated_param_names: List[str]) -> None:
        """Inflate a norm module from 2d to 3d.

        Args:
            bn3d (nn.Module): The destination bn3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding bn module in the
                2d model.
            inflated_param_names (List[str]): List of parameters that have been
                inflated.
        """
        for param_name, param in bn3d.named_parameters():
            param_2d_name = f'{module_name_2d}.{param_name}'
            param_2d = state_dict_2d[param_2d_name]
            if param.data.shape != param_2d.shape:
                warnings.warn(f'The parameter of {module_name_2d} is not'
                              'loaded due to incompatible shapes. ')
                return

            param.data.copy_(param_2d)
            inflated_param_names.append(param_2d_name)

        for param_name, param in bn3d.named_buffers():
            param_2d_name = f'{module_name_2d}.{param_name}'
            # some buffers like num_batches_tracked may not exist in old
            # checkpoints
            if param_2d_name in state_dict_2d:
                param_2d = state_dict_2d[param_2d_name]
                param.data.copy_(param_2d)
                inflated_param_names.append(param_2d_name)

    @staticmethod
    def _inflate_weights(self, logger: MMLogger) -> None:
        """Inflate the resnet2d parameters to resnet3d.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (MMLogger): The logger used to print
                debugging information.
        """

        state_dict_r2d = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']

        inflated_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if 'downsample' in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace('conv', 'bn')
                if original_conv_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_conv_name}')
                else:
                    shape_2d = state_dict_r2d[original_conv_name +
                                              '.weight'].shape
                    shape_3d = module.conv.weight.data.shape
                    if shape_2d != shape_3d[:2] + shape_3d[3:]:
                        logger.warning(f'Weight shape mismatch for '
                                       f': {original_conv_name} : '
                                       f'3d weight shape: {shape_3d}; '
                                       f'2d weight shape: {shape_2d}. ')
                    else:
                        self._inflate_conv_params(module.conv, state_dict_r2d,
                                                  original_conv_name,
                                                  inflated_param_names)

                if original_bn_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_bn_name}')
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d,
                                            original_bn_name,
                                            inflated_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded'
                        f': {remaining_names}')

    def inflate_weights(self, logger: MMLogger) -> None:
        self._inflate_weights(self, logger)

    def _make_stem_layer(self) -> None:
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=self.conv1_kernel,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=tuple([(k - 1) // 2 for k in _triple(self.conv1_kernel)]),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(self.pool1_stride_t, self.pool1_stride_s,
                    self.pool1_stride_s),
            padding=(0, 1, 1))

        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @staticmethod
    def _init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch.

        Args:
            pretrained (str | None): The path of the pretrained weight. Will
                override the original `pretrained` if set. The arg is added to
                be compatible with mmdet. Default: None.
        """
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)

            else:
                # Directly load 3D model.
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck3d):
                        constant_init(m.conv3.bn, 0)
                    elif isinstance(m, BasicBlock3d):
                        constant_init(m.conv2.bn, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        self._init_weights(self, pretrained)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor or Tuple[Tensor]: The feature of the input
                samples extracted by the backbone.
        """
        x = self.conv1(x)
        if self.with_pool1:
            x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def train(self, mode: bool = True) -> None:
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class ResNet3dLayer(nn.Module):
    """ResNet 3d Layer.

    Args:
        depth (int): Depth of resnet,
            from {``18``, ``34``, ``50``, ``101``, ``152``}.
        pretrained (str, optional): Name of pretrained model. Defaults to None.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Defaults to True.
        stage (int): The index of Resnet stage. Defaults to 3.
        base_channels (int): Channel num of stem output features.
            Defaults to 64.
        spatial_stride (int): The 1st res block's spatial stride.
            Defaults to 2.
        temporal_stride (int): The 1st res block's temporal stride.
            Defaults to 1.
        dilation (int): The dilation. Defaults to 1.
        style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``, the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``pytorch``.
        all_frozen (bool): Frozen all modules in the layer. Defaults to False.
        inflate (int): Inflate dims of each block. Defaults to 1.
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Defaults to ``3x1x1``.
        conv_cfg (dict or ConfigDict): Config for conv layers.
            Required keys are ``type``. Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict or ConfigDict): Config for norm layers.
            Required keys are ``type`` and ``requires_grad``.
            Defaults to ``dict(type='BN3d', requires_grad=True)``.
        act_cfg (dict or ConfigDict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU', inplace=True)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (``mean`` and ``var``). Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Defaults to True.
    """

    def __init__(self,
                 depth: int,
                 pretrained: Optional[str] = None,
                 pretrained2d: bool = True,
                 stage: int = 3,
                 base_channels: int = 64,
                 spatial_stride: int = 2,
                 temporal_stride: int = 1,
                 dilation: int = 1,
                 style: str = 'pytorch',
                 all_frozen: bool = False,
                 inflate: int = 1,
                 inflate_style: str = '3x1x1',
                 conv_cfg: ConfigType = dict(type='Conv3d'),
                 norm_cfg: ConfigType = dict(type='BN3d', requires_grad=True),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 zero_init_residual: bool = True,
                 **kwargs) -> None:
        super().__init__()
        self.arch_settings = ResNet3d.arch_settings
        assert depth in self.arch_settings

        self.make_res_layer = ResNet3d.make_res_layer
        self._inflate_conv_params = ResNet3d._inflate_conv_params
        self._inflate_bn_params = ResNet3d._inflate_bn_params
        self._inflate_weights = ResNet3d._inflate_weights
        self._init_weights = ResNet3d._init_weights

        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.stage = stage
        # stage index is 0 based
        assert 0 <= stage <= 3
        self.base_channels = base_channels

        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation

        self.style = style
        self.all_frozen = all_frozen

        self.stage_inflation = inflate
        self.inflate_style = inflate_style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        block, stage_blocks = self.arch_settings[depth]
        stage_block = stage_blocks[stage]
        planes = 64 * 2**stage
        inplanes = 64 * 2**(stage - 1) * block.expansion

        res_layer = self.make_res_layer(
            block,
            inplanes,
            planes,
            stage_block,
            spatial_stride=spatial_stride,
            temporal_stride=temporal_stride,
            dilation=dilation,
            style=self.style,
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
            act_cfg=self.act_cfg,
            inflate=self.stage_inflation,
            inflate_style=self.inflate_style,
            with_cp=with_cp,
            **kwargs)

        self.layer_name = f'layer{stage + 1}'
        self.add_module(self.layer_name, res_layer)

    def inflate_weights(self, logger: MMLogger) -> None:
        self._inflate_weights(self, logger)

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.all_frozen:
            layer = getattr(self, self.layer_name)
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        self._init_weights(self, pretrained)

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The feature of the input
                samples extracted by the resisual layer.
        """
        res_layer = getattr(self, self.layer_name)
        out = res_layer(x)
        return out

    def train(self, mode: bool = True) -> None:
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
