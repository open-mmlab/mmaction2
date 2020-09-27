import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _ntuple

from mmaction.models.backbones.resnet3d import ResNet3d
from mmaction.models.common import build_norm_layer
from mmaction.models.registry import BACKBONES
from mmaction.utils import get_root_logger


class Bottleneck2dAudio(nn.Module):
    """Bottleneck2D block for ResNet2D.

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
                 inflate_style='3x1',
                 norm_cfg=dict(type='SyncBN'),
                 with_cp=False):
        super().__init__()
        assert style in ['pytorch', 'caffe']
        assert inflate_style in ['3x1']

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

        self.conv1_stride_s = 1
        self.conv2_stride_s = spatial_stride
        self.conv1_stride_t = 1
        self.conv2_stride_t = temporal_stride

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        if self.inflate:
            conv1_kernel_size = (1, 1)
            conv1_padding = 0
            conv2_kernel_size_1 = (3, 1)
            conv2_padding_1 = (dilation, 0)
            conv2_kernel_size_2 = (1, 3)
            conv2_padding_2 = (0, dilation)
            self.conv1 = nn.Conv2d(
                inplanes,
                planes,
                kernel_size=conv1_kernel_size,
                stride=(self.conv1_stride_t, self.conv1_stride_s),
                padding=conv1_padding,
                bias=False)
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    planes,
                    planes,
                    kernel_size=conv2_kernel_size_1,
                    stride=(self.conv2_stride_t, 1),
                    padding=conv2_padding_1,
                    dilation=(dilation, 1),
                    bias=False),
                nn.Conv2d(
                    planes,
                    planes,
                    kernel_size=conv2_kernel_size_2,
                    stride=(1, self.conv2_stride_s),
                    padding=conv2_padding_2,
                    dilation=(1, dilation),
                    bias=False))

        else:
            self.conv1 = nn.Conv2d(
                inplanes,
                planes,
                kernel_size=1,
                stride=(1, self.conv1_stride_s),
                bias=False)
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=(3, 3),
                stride=(self.conv2_stride_t, self.conv2_stride_s),
                padding=(dilation, dilation),
                dilation=(dilation, dilation),
                bias=False)

        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)

        self.conv3 = nn.Conv2d(
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


@BACKBONES.register_module
class ResNet2dAudio(nn.Module):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str | None): Name of pretrained model.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Default: True.
        in_channels (int): Channel num of input features. Default: 3.
        base_channels (int): Channel num of stem output features. Default: 64.
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
        50: (Bottleneck2dAudio, (3, 4, 6, 3)),
        101: (Bottleneck2dAudio, (3, 4, 23, 3)),
        152: (Bottleneck2dAudio, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained,
                 in_channels=1,
                 num_stages=4,
                 base_channels=32,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 conv1_kernel=(9, 1),
                 conv1_stride_t=2,
                 pool1_stride_t=2,
                 style='pytorch',
                 frozen_stages=-1,
                 inflate=(1, 1, 0, 0),
                 inflate_stride=(1, 1, 1, 1),
                 inflate_style='3x1',
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=True):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = base_channels
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
        self.inplanes = self.base_channels

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
                norm_cfg=norm_cfg,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * self.base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_res_layer(self,
                       block,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       style='pytorch',
                       inflate=1,
                       inflate_style='3x1',
                       norm_cfg=None,
                       with_cp=False):
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature
                in each block.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides in
                residual and conv layers. Default: 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Default: 1.
            dilation (int): Spacing between kernel elements. Default: 1.
            style (str): `pytorch` or `caffe`. If set to "pytorch",
                the stride-two layer is the 3x3 conv layer, otherwise
                the stride-two layer is the first 1x1 conv layer.
                Default: 'pytorch'.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Default: 1.
            inflate_style (str): `3x1x1` or `1x1x1`. which determines
                the kernel sizes and padding strides for conv1 and conv2
                in each block. Default: '3x1x1'.
            norm_cfg (dict): Config for norm layers. Default: None.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save
                some memory while slowing down the training speed.
                Default: False.

        Returns:
            A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks
        assert len(inflate) == blocks
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(temporal_stride, spatial_stride),
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
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

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        conv1_kernel_size1, conv1_kernel_size2 = self.conv1_kernel
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.base_channels,
                kernel_size=(conv1_kernel_size1, conv1_kernel_size2),
                stride=(1, 1),
                padding=((conv1_kernel_size1 - 1) // 2,
                         (conv1_kernel_size2 - 1) // 2),
                bias=False),
            nn.Conv2d(
                self.base_channels,
                self.base_channels,
                kernel_size=(conv1_kernel_size2, conv1_kernel_size1),
                stride=(1, 1),
                padding=((conv1_kernel_size2 - 1) // 2,
                         (conv1_kernel_size1 - 1) // 2),
                bias=False))
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, self.base_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck2dAudio):
                        constant_init(m.norm3, 0)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
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


@BACKBONES.register_module
class AVResNet(nn.Module):

    def __init__(self,
                 depth,
                 pretrained,
                 audio_pretrained,
                 pretrained2d=True,
                 in_channels=3,
                 num_stages=4,
                 base_channels=64,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 conv1_kernel=(5, 7, 7),
                 conv1_stride_t=2,
                 pool1_stride_t=2,
                 inflate=(1, 1, 1, 1),
                 inflate_stride=(1, 1, 1, 1),
                 style='pytorch',
                 frozen_stages=-1,
                 inflate_style='3x1x1',
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 audio_in_channels=1,
                 audio_base_channels=32,
                 audio_spatial_strides=(1, 2, 2, 2),
                 audio_temporal_strides=(1, 2, 2, 2),
                 audio_dilations=(1, 1, 1, 1),
                 audio_conv1_kernel=(9, 1),
                 audio_conv1_stride_t=2,
                 audio_pool1_stride_t=2,
                 audio_frozen_stages=-1,
                 audio_inflate=(1, 1, 0, 0),
                 audio_inflate_stride=(1, 1, 1, 1),
                 audio_inflate_style='3x1',
                 audio_norm_cfg=dict(type='SyncBN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=True,
                 load_img=True,
                 load_audio=True):
        super().__init__()
        self.load_img = load_img
        self.load_audio = load_audio
        self.vision_net = ResNet3d(depth, pretrained, pretrained2d,
                                   in_channels, num_stages, base_channels,
                                   spatial_strides, temporal_strides,
                                   dilations, conv1_kernel, conv1_stride_t,
                                   pool1_stride_t, style, frozen_stages,
                                   inflate, inflate_stride, inflate_style,
                                   norm_cfg, norm_eval, with_cp,
                                   zero_init_residual)
        self.audio_net = ResNet2dAudio(
            depth, audio_pretrained, audio_in_channels, num_stages,
            audio_base_channels, audio_spatial_strides, audio_temporal_strides,
            audio_dilations, audio_conv1_kernel, audio_conv1_stride_t,
            audio_pool1_stride_t, style, audio_frozen_stages, audio_inflate,
            audio_inflate_stride, audio_inflate_style, audio_norm_cfg,
            norm_eval, with_cp, zero_init_residual)

    def _freeze_stages(self):
        self.audio_net._freeze_stages()
        self.vision_net._freeze_stages()

    def init_weights(self):
        self.audio_net.init_weights()
        self.vision_net.init_weights()

    def forward(self, imgs, audios):
        vision_fea = None
        audio_fea = None
        if self.load_img:
            vision_fea = self.vision_net.forward(imgs)
        if self.load_audio:

            audio_fea = self.audio_net.forward(audios)
        return [vision_fea, audio_fea]

    def train(self, mode=True):
        super().train(mode)
        self.audio_net.train()
        self.vision_net.train()
