import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _ntuple, _triple

from mmaction.models.common import build_norm_layer
from mmaction.models.registry import BACKBONES
from mmaction.utils import get_root_logger


def get_layer_names(layer_params):
    """get layer name when loading checkpoint."""
    layer_names = list()

    for layer_param in layer_params:
        pos = layer_param.rfind('.')
        layer = layer_param[:pos]
        if layer not in layer_names:
            layer_names.append(layer)

    return layer_names


class Bottleneck3d(nn.Module):
    """Bottleneck3D block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module): Downsample layer. Default: None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        inflate (bool): Whether to inflate kernel. Default: True.
        inflate_style (str): `3x1x1` or `1x1x1`. which determines the kernel
            sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        norm_cfg (dict): Config for norm layers. required keys are `type`,
            Default: dict(type='BN3d').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 inflate=True,
                 inflate_style='3x1x1',
                 norm_cfg=dict(type='BN3d'),
                 with_cp=False):
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
        self.with_cp = with_cp

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

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        if self.inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = 0
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
            self.conv1 = nn.Conv3d(
                inplanes,
                planes,
                kernel_size=conv1_kernel_size,
                stride=(self.conv1_stride_t, self.conv1_stride_s,
                        self.conv1_stride_s),
                padding=conv1_padding,
                bias=False)
            self.conv2 = nn.Conv3d(
                planes,
                planes,
                kernel_size=conv2_kernel_size,
                stride=(self.conv2_stride_t, self.conv2_stride_s,
                        self.conv2_stride_s),
                padding=conv2_padding,
                dilation=(1, dilation, dilation),
                bias=False)
        else:
            self.conv1 = nn.Conv3d(
                inplanes,
                planes,
                kernel_size=1,
                stride=(1, self.conv1_stride_s, self.conv1_stride_s),
                bias=False)
            self.conv2 = nn.Conv3d(
                planes,
                planes,
                kernel_size=(1, 3, 3),
                stride=(1, self.conv2_stride_s, self.conv2_stride_s),
                padding=(0, dilation, dilation),
                dilation=(1, dilation, dilation),
                bias=False)

        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)

        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

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


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   spatial_stride=1,
                   temporal_stride=1,
                   dilation=1,
                   style='pytorch',
                   inflate=1,
                   inflate_style='3x1x1',
                   norm_cfg=None,
                   with_cp=False):
    """Build residual layer for ResNet3D.

    Args:
        block (nn.Module): Residual module to be built.
        inplanes (int): Number of channels for the input feature in each block.
        planes (int): Number of channels for the output feature in each block.
        blocks (int): Number of residual blocks.
        spatial_stride (int | Sequence[int]): Spatial strides in residual and
            conv layers. Default: 1.
        temporal_stride (int | Sequence[int]): Temporal strides in residual and
            conv layers. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        inflate (int | Sequence[int]): Determine whether to inflate for each
            block. Default: 1.
        inflate_style (str): `3x1x1` or `1x1x1`. which determines the kernel
            sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        norm_cfg (dict): Config for norm layers. Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.

    Returns:
        A residual layer for the given config.
    """
    inflate = inflate if not isinstance(inflate, int) else (inflate, ) * blocks
    assert len(inflate) == blocks
    downsample = None
    if spatial_stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv3d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False),
            nn.BatchNorm3d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            spatial_stride,
            temporal_stride,
            dilation,
            downsample,
            style=style,
            inflate=(inflate[0] == 1),
            inflate_style=inflate_style,
            norm_cfg=norm_cfg,
            with_cp=with_cp))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                1,
                dilation,
                style=style,
                inflate=(inflate[i] == 1),
                inflate_style=inflate_style,
                norm_cfg=norm_cfg,
                with_cp=with_cp))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet3d(nn.Module):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str | None): Name of pretrained model.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Default: True.
        in_channels (int): Channel num of input features. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
        temporal_strides (Sequence[int]):
            Temporal strides of residual blocks of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
        conv1_stride_t (int): Temporal stride of the first conv layer.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        inflate (Sequence[int]): Inflate Dims of each block.
        inflate_stride (Sequence[int]):
            Inflate stride of each block.
        inflate_style (str): `3x1x1` or `1x1x1`. which determines the kernel
            sizes and padding strides for conv1 and conv2 in each block.
        norm_cfg (dict): Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN3d', requires_grad=True).
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
    """

    arch_settings = {
        # 18: (BasicBlock3d, (2, 2, 2, 2)),
        # 34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained,
                 pretrained2d=True,
                 in_channels=3,
                 num_stages=4,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 conv1_kernel=(5, 7, 7),
                 conv1_stride_t=2,
                 pool1_stride_t=2,
                 style='pytorch',
                 frozen_stages=-1,
                 inflate=(1, 1, 1, 1),
                 inflate_stride=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=True):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.in_channels = in_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(
            dilations) == num_stages
        self.conv1_kernel = conv1_kernel
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_t = pool1_stride_t
        self.style = style
        self.frozen_stages = frozen_stages
        self.stage_inflations = _ntuple(num_stages)(inflate)
        self.inflate_style = inflate_style
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                norm_cfg=norm_cfg,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def inflate_weights(self, logger):
        resnet2d = _load_checkpoint(self.pretrained)
        if 'state_dict' in resnet2d.keys():
            resnet2d = resnet2d['state_dict']
        layer_names = get_layer_names(resnet2d.keys())

        for name, module in self.named_modules():
            if isinstance(module, nn.Conv3d) and name in layer_names:
                old_weight = resnet2d[name + '.weight'].data
                new_weight = old_weight.unsqueeze(2).expand_as(
                    module.weight) / module.weight.data.shape[2]
                module.weight.data.copy_(new_weight)
                logger.info(
                    '{}.weight loaded from weights file into {}'.format(
                        name, new_weight.shape))

                if hasattr(module, 'bias') and module.bias is not None:
                    new_bias = resnet2d[name + '.bias'].data
                    module.bias.data.copy_(new_bias)
                    logger.info(
                        '{}.bias loaded from weights file into {}'.format(
                            name, new_bias))

            elif isinstance(module, nn.BatchNorm3d) and name in layer_names:
                param_names = list()
                for param_name in resnet2d.keys():
                    if param_name.startswith(name):
                        param_names.append(param_name)

                for param_name in param_names:
                    attr = param_name.split('.')[-1]
                    new_parms = resnet2d[param_name]
                    logger.info('{} loaded from weights file into {}'.format(
                        param_name, new_parms.shape))
                    setattr(module, attr, new_parms)

    def _make_stem_layer(self):
        self.conv1 = nn.Conv3d(
            self.in_channels,
            64,
            kernel_size=self.conv1_kernel,
            stride=(self.conv1_stride_t, 2, 2),
            padding=tuple([(k - 1) // 2 for k in _triple(self.conv1_kernel)]),
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(self.pool1_stride_t, 2, 2),
            padding=(0, 1, 1))

        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info('load model from: {}'.format(self.pretrained))

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
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck3d):
                        constant_init(m.norm3, 0)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0:
                x = self.pool2(x)
        return x

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
