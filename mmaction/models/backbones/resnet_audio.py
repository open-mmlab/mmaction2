# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.runner import load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch.nn.modules.utils import _ntuple

from mmaction.registry import MODELS
from mmaction.utils import ConfigType


class Bottleneck2dAudio(nn.Module):
    """Bottleneck2D block for ResNet2D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        stride (int): Stride in the conv layer. Defaults to 2.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        downsample (nn.Module, optional): Downsample layer. Defaults to None.
        factorize (bool): Whether to factorize kernel. Defaults to True.
        norm_cfg (dict): Config for norm layers. required keys are ``type`` and
            ``requires_grad``. Defaults to None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the trgaining speed. Defaults to False.
    """
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 2,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 factorize: bool = True,
                 norm_cfg: ConfigType = None,
                 with_cp: bool = False) -> None:
        super().__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.factorize = factorize
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

        self.conv1_stride = 1
        self.conv2_stride = stride

        conv1_kernel_size = (1, 1)
        conv1_padding = 0
        conv2_kernel_size = (3, 3)
        conv2_padding = (dilation, dilation)
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=conv1_kernel_size,
            padding=conv1_padding,
            dilation=dilation,
            norm_cfg=self.norm_cfg,
            bias=False)
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=conv2_kernel_size,
            stride=stride,
            padding=conv2_padding,
            dilation=dilation,
            bias=False,
            conv_cfg=dict(type='ConvAudio') if factorize else dict(
                type='Conv'),
            norm_cfg=None,
            act_cfg=None)
        self.conv3 = ConvModule(
            2 * planes if factorize else planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@MODELS.register_module()
class ResNetAudio(nn.Module):
    """ResNet 2d audio backbone. Reference:

        <https://arxiv.org/abs/2001.08740>`_.

    Args:
        depth (int): Depth of resnet, from ``{50, 101, 152}``.
        pretrained (str, optional): Name of pretrained model. Defaults to None.
        in_channels (int): Channel num of input features. Defaults to 1.
        base_channels (int): Channel num of stem output features.
            Defaults to 32.
        num_stages (int): Resnet stages. Defaults to 4.
        strides (Sequence[int]): Strides of residual blocks of each stage.
            Defaults to ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        conv1_kernel (int): Kernel size of the first conv layer. Defaults to 9.
        conv1_stride (Union[int, Tuple[int]]): Stride of the first conv layer.
            Defaults to 1.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Defaults to -1.
        factorize (Sequence[int]): factorize Dims of each block for audio.
            Defaults to ``(1, 1, 0, 0)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        conv_cfg (Union[dict, ConfigDict]): Config for norm layers.
            Defaults to ``dict(type='Conv')``.
        norm_cfg (Union[dict, ConfigDict]): Config for norm layers. required
            keys are ``type`` and ``requires_grad``.
            Defaults to ``dict(type='BN2d', requires_grad=True)``.
        act_cfg (Union[dict, ConfigDict]): Config for activate layers.
            Defaults to ``dict(type='ReLU', inplace=True)``.
        zero_init_residual (bool): Whether to use zero initialization
            for residual block. Defaults to True.
    """

    arch_settings = {
        # 18: (BasicBlock2dAudio, (2, 2, 2, 2)),
        # 34: (BasicBlock2dAudio, (3, 4, 6, 3)),
        50: (Bottleneck2dAudio, (3, 4, 6, 3)),
        101: (Bottleneck2dAudio, (3, 4, 23, 3)),
        152: (Bottleneck2dAudio, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int,
                 pretrained: str = None,
                 in_channels: int = 1,
                 num_stages: int = 4,
                 base_channels: int = 32,
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 conv1_kernel: int = 9,
                 conv1_stride: int = 1,
                 frozen_stages: int = -1,
                 factorize: Sequence[int] = (1, 1, 0, 0),
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 conv_cfg: ConfigType = dict(type='Conv'),
                 norm_cfg: ConfigType = dict(type='BN2d', requires_grad=True),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 zero_init_residual: bool = True) -> None:
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.dilations = dilations
        self.conv1_kernel = conv1_kernel
        self.conv1_stride = conv1_stride
        self.frozen_stages = frozen_stages
        self.stage_factorization = _ntuple(num_stages)(factorize)
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = self.base_channels

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                factorize=self.stage_factorization[i],
                norm_cfg=self.norm_cfg,
                with_cp=with_cp)
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
                       stride: int = 1,
                       dilation: int = 1,
                       factorize: int = 1,
                       norm_cfg: Optional[ConfigType] = None,
                       with_cp: bool = False) -> nn.Module:
        """Build residual layer for ResNetAudio.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature
                in each block.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            stride (int): Strides of residual blocks of each stage.
                Defaults to  1.
            dilation (int): Spacing between kernel elements. Defaults to 1.
            factorize (Uninon[int, Sequence[int]]): Determine whether to
                factorize for each block. Defaults to 1.
            norm_cfg (Union[dict, ConfigDict], optional): Config for norm
                layers. Defaults to None.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save
                some memory while slowing down the training speed.
                Defaults to False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        factorize = factorize if not isinstance(
            factorize, int) else (factorize, ) * blocks
        assert len(factorize) == blocks
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
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
                factorize=(factorize[0] == 1),
                norm_cfg=norm_cfg,
                with_cp=with_cp))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    1,
                    dilation,
                    factorize=(factorize[i] == 1),
                    norm_cfg=norm_cfg,
                    with_cp=with_cp))

        return nn.Sequential(*layers)

    def _make_stem_layer(self) -> None:
        """Construct the stem layers consists of a ``conv+norm+act`` module and
        a pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=self.conv1_kernel,
            stride=self.conv1_stride,
            bias=False,
            conv_cfg=dict(type='ConvAudio', op='sum'),
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.bn.eval()
            for m in [self.conv1.conv, self.conv1.bn]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck2dAudio):
                        constant_init(m.conv3.bn, 0)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input samples extracted
                by the backbone.
        """
        x = self.conv1(x)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        return x

    def train(self, mode: bool = True) -> None:
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
