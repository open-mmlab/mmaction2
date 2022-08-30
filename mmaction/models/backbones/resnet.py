# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple, Union

import mmengine
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch.utils import checkpoint as cp

from mmaction.registry import MODELS
from mmaction.utils import ConfigType


class BasicBlock(nn.Module):
    """Basic block for ResNet.

    Args:
        inplanes (int): Number of channels for the input in first conv2d layer.
        planes (int): Number of channels produced by some norm/conv2d layers.
        stride (int): Stride in the conv layer. Defaults to 1.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        downsample (nn.Module, optional): Downsample layer. Defaults to None.
        style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``, the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``pytorch``.
        conv_cfg (Union[dict, ConfigDict]): Config for norm layers.
            Defaults to ``dict(type='Conv')``.
        norm_cfg (Union[dict, ConfigDict]): Config for norm layers. required
            keys are ``type`` and ``requires_grad``.
            Defaults to ``dict(type='BN2d', requires_grad=True)``.
        act_cfg (Union[dict, ConfigDict]): Config for activate layers.
            Defaults to ``dict(type='ReLU', inplace=True)``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 style: str = 'pytorch',
                 conv_cfg: ConfigType = dict(type='Conv'),
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 with_cp: bool = False) -> None:
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.style = style
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        assert not with_cp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        inplanes (int):
            Number of channels for the input feature in first conv layer.
        planes (int):
            Number of channels produced by some norm layes and conv layers.
        stride (int): Spatial stride in the conv layer. Defaults to 1.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        downsample (nn.Module, optional): Downsample layer. Defaults to None.
        style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``, the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``pytorch``.
        conv_cfg (Union[dict, ConfigDict]): Config for norm layers.
            Defaults to ``dict(type='Conv')``.
        norm_cfg (Union[dict, ConfigDict]): Config for norm layers. required
            keys are ``type`` and ``requires_grad``.
            Defaults to ``dict(type='BN2d', requires_grad=True)``.
        act_cfg (Union[dict, ConfigDict]): Config for activate layers.
            Defaults to ``dict(type='ReLU', inplace=True)``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """

    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 style: str = 'pytorch',
                 conv_cfg: ConfigType = dict(type='Conv'),
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 with_cp: bool = False) -> None:
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """

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

        return out


def make_res_layer(block: nn.Module,
                   inplanes: int,
                   planes: int,
                   blocks: int,
                   stride: int = 1,
                   dilation: int = 1,
                   style: str = 'pytorch',
                   conv_cfg: Optional[ConfigType] = None,
                   norm_cfg: Optional[ConfigType] = None,
                   act_cfg: Optional[ConfigType] = None,
                   with_cp: bool = False) -> nn.Module:
    """Build residual layer for ResNet.

    Args:
        block: (nn.Module): Residual module to be built.
        inplanes (int): Number of channels for the input feature in each block.
        planes (int): Number of channels for the output feature in each block.
        blocks (int): Number of residual blocks.
        stride (int): Stride in the conv layer. Defaults to 1.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``, the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``pytorch``.
        conv_cfg (Union[dict, ConfigDict], optional): Config for norm layers.
            Defaults to None.
        norm_cfg (Union[dict, ConfigDict], optional): Config for norm layers.
            Defaults to None.
        act_cfg (Union[dict, ConfigDict], optional): Config for activate
            layers. Defaults to None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.

    Returns:
        nn.Module: A residual layer for the given config.
    """
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = ConvModule(
            inplanes,
            planes * block.expansion,
            kernel_size=1,
            stride=stride,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_cp=with_cp))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation,
                style=style,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp))

    return nn.Sequential(*layers)


@MODELS.register_module()
class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from ``{18, 34, 50, 101, 152}``.
        pretrained (str, optional): Name of pretrained model. Defaults to None.
        torchvision_pretrain (bool): Whether to load pretrained model from
            torchvision. Defaults to True.
        in_channels (int): Channel num of input features. Defaults to 3.
        num_stages (int): Resnet stages. Defaults to 4.
        out_indices (Sequence[int]): Indices of output feature.
            Defaults to (3, ).
        strides (Sequence[int]): Strides of the first block of each stage.
            Defaults to ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``, the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``pytorch``.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Defaults to -1.
        conv_cfg (dict or ConfigDict): Config for norm layers.
            Defaults ``dict(type='Conv')``.
        norm_cfg (Union[dict, ConfigDict]): Config for norm layers. required
            keys are ``type`` and ``requires_grad``.
            Defaults to ``dict(type='BN2d', requires_grad=True)``.
        act_cfg (Union[dict, ConfigDict]): Config for activate layers.
            Defaults to ``dict(type='ReLU', inplace=True)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Defaults to False.
        partial_bn (bool): Whether to use partial bn. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int,
                 pretrained: Optional[str] = None,
                 torchvision_pretrain: bool = True,
                 in_channels: int = 3,
                 num_stages: int = 4,
                 out_indices: Sequence[int] = (3, ),
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 style: str = 'pytorch',
                 frozen_stages: int = -1,
                 conv_cfg: ConfigType = dict(type='Conv'),
                 norm_cfg: ConfigType = dict(type='BN2d', requires_grad=True),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 norm_eval: bool = False,
                 partial_bn: bool = False,
                 with_cp: bool = False) -> None:
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.torchvision_pretrain = torchvision_pretrain
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.partial_bn = partial_bn
        self.with_cp = with_cp

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    def _make_stem_layer(self) -> None:
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    @staticmethod
    def _load_conv_params(conv: nn.Module, state_dict_tv: OrderedDict,
                          module_name_tv: str,
                          loaded_param_names: List[str]) -> None:
        """Load the conv parameters of resnet from torchvision.

        Args:
            conv (nn.Module): The destination conv module.
            state_dict_tv (OrderedDict): The state dict of pretrained
                torchvision model.
            module_name_tv (str): The name of corresponding conv module in the
                torchvision model.
            loaded_param_names (list[str]): List of parameters that have been
                loaded.
        """

        weight_tv_name = module_name_tv + '.weight'
        if conv.weight.data.shape == state_dict_tv[weight_tv_name].shape:
            conv.weight.data.copy_(state_dict_tv[weight_tv_name])
            loaded_param_names.append(weight_tv_name)

        if getattr(conv, 'bias') is not None:
            bias_tv_name = module_name_tv + '.bias'
            if conv.bias.data.shape == state_dict_tv[bias_tv_name].shape:
                conv.bias.data.copy_(state_dict_tv[bias_tv_name])
                loaded_param_names.append(bias_tv_name)

    @staticmethod
    def _load_bn_params(bn: nn.Module, state_dict_tv: OrderedDict,
                        module_name_tv: str,
                        loaded_param_names: List[str]) -> None:
        """Load the bn parameters of resnet from torchvision.

        Args:
            bn (nn.Module): The destination bn module.
            state_dict_tv (OrderedDict): The state dict of pretrained
                torchvision model.
            module_name_tv (str): The name of corresponding bn module in the
                torchvision model.
            loaded_param_names (list[str]): List of parameters that have been
                loaded.
        """

        for param_name, param in bn.named_parameters():
            param_tv_name = f'{module_name_tv}.{param_name}'
            param_tv = state_dict_tv[param_tv_name]
            if param.data.shape == param_tv.shape:
                param.data.copy_(param_tv)
                loaded_param_names.append(param_tv_name)

        for param_name, param in bn.named_buffers():
            param_tv_name = f'{module_name_tv}.{param_name}'
            # some buffers like num_batches_tracked may not exist
            if param_tv_name in state_dict_tv:
                param_tv = state_dict_tv[param_tv_name]
                if param.data.shape == param_tv.shape:
                    param.data.copy_(param_tv)
                    loaded_param_names.append(param_tv_name)

    def _load_torchvision_checkpoint(self,
                                     logger: mmengine.MMLogger = None) -> None:
        """Initiate the parameters from torchvision pretrained checkpoint."""
        state_dict_torchvision = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_torchvision:
            state_dict_torchvision = state_dict_torchvision['state_dict']

        loaded_param_names = []
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
                self._load_conv_params(module.conv, state_dict_torchvision,
                                       original_conv_name, loaded_param_names)
                self._load_bn_params(module.bn, state_dict_torchvision,
                                     original_bn_name, loaded_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_torchvision.keys()) - set(loaded_param_names)
        if remaining_names:
            logger.info(
                f'These parameters in pretrained checkpoint are not loaded'
                f': {remaining_names}')

    def init_weights(self) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            if self.torchvision_pretrain:
                # torchvision's
                self._load_torchvision_checkpoint(logger)
            else:
                # ours
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: torch.Tensor) \
            -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            Union[torch.Tensor or Tuple[torch.Tensor]]: The feature of the
                input samples extracted by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.bn.eval()
            for m in self.conv1.modules():
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _partial_bn(self) -> None:
        logger = MMLogger.get_current_instance()
        logger.info('Freezing BatchNorm2D except the first one.')
        count_bn = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                count_bn += 1
                if count_bn >= 2:
                    m.eval()
                    # shutdown update in frozen mode
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def train(self, mode: bool = True) -> None:
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        if mode and self.partial_bn:
            self._partial_bn()
