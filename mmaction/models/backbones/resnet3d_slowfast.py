# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import print_log

from ...utils import get_root_logger
from ..builder import BACKBONES
from .resnet3d import ResNet3d

try:
    from mmdet.models import BACKBONES as MMDET_BACKBONES
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


class ResNet3dPathway(ResNet3d):
    """A pathway of Slowfast based on ResNet3d.

    Args:
        *args (arguments): Arguments same as :class:``ResNet3d``.
        lateral (bool): Determines whether to enable the lateral connection
            from another pathway. Default: False.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            ``alpha`` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to ``beta`` in the paper.
            Default: 8.
        fusion_kernel (int): The kernel size of lateral fusion.
            Default: 5.
        **kwargs (keyword arguments): Keywords arguments for ResNet3d.
    """

    def __init__(self,
                 *args,
                 lateral=False,
                 speed_ratio=8,
                 channel_ratio=8,
                 fusion_kernel=5,
                 **kwargs):
        self.lateral = lateral
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        super().__init__(*args, **kwargs)
        self.inplanes = self.base_channels
        if self.lateral:
            self.conv1_lateral = ConvModule(
                self.inplanes // self.channel_ratio,
                # https://arxiv.org/abs/1812.03982, the
                # third type of lateral connection has out_channel:
                # 2 * \beta * C
                self.inplanes * 2 // self.channel_ratio,
                kernel_size=(fusion_kernel, 1, 1),
                stride=(self.speed_ratio, 1, 1),
                padding=((fusion_kernel - 1) // 2, 0, 0),
                bias=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
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
                        self.inplanes // self.channel_ratio,
                        self.inplanes * 2 // self.channel_ratio,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=None,
                        act_cfg=None))
                self.lateral_connections.append(lateral_name)

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
                       inflate_style='3x1x1',
                       non_local=0,
                       non_local_cfg=dict(),
                       conv_cfg=None,
                       norm_cfg=None,
                       act_cfg=None,
                       with_cp=False):
        """Build residual layer for Slowfast.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input
                feature in each block.
            planes (int): Number of channels for the output
                feature in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides
                in residual and conv layers. Default: 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Default: 1.
            dilation (int): Spacing between kernel elements. Default: 1.
            style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``,
                the stride-two layer is the 3x3 conv layer,
                otherwise the stride-two layer is the first 1x1 conv layer.
                Default: ``pytorch``.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Default: 1.
            inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines
                the kernel sizes and padding strides for conv1 and
                conv2 in each block. Default: ``3x1x1``.
            non_local (int | Sequence[int]): Determine whether to apply
                non-local module in the corresponding block of each stages.
                Default: 0.
            non_local_cfg (dict): Config for non-local module.
                Default: ``dict()``.
            conv_cfg (dict | None): Config for conv layers. Default: None.
            norm_cfg (dict | None): Config for norm layers. Default: None.
            act_cfg (dict | None): Config for activate layers. Default: None.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save
                some memory while slowing down the training speed.
                Default: False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks
        non_local = non_local if not isinstance(
            non_local, int) else (non_local, ) * blocks
        assert len(inflate) == blocks and len(non_local) == blocks
        if self.lateral:
            lateral_inplanes = inplanes * 2 // self.channel_ratio
        else:
            lateral_inplanes = 0
        if (spatial_stride != 1
                or (inplanes + lateral_inplanes) != planes * block.expansion):
            downsample = ConvModule(
                inplanes + lateral_inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
        else:
            downsample = None

        layers = []
        layers.append(
            block(
                inplanes + lateral_inplanes,
                planes,
                spatial_stride,
                temporal_stride,
                dilation,
                downsample,
                style=style,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
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
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp))

        return nn.Sequential(*layers)

    def inflate_weights(self, logger):
        """Inflate the resnet2d parameters to resnet3d pathway.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart. For pathway the ``lateral_connection`` part should
        not be inflated from 2d weights.

        Args:
            logger (logging.Logger): The logger used to print
                debugging information.
        """

        state_dict_r2d = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']

        inflated_param_names = []
        for name, module in self.named_modules():
            if 'lateral' in name:
                continue
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

    def _inflate_conv_params(self, conv3d, state_dict_2d, module_name_2d,
                             inflated_param_names):
        """Inflate a conv module from 2d to 3d.

        The differences of conv modules betweene 2d and 3d in Pathway
        mainly lie in the inplanes due to lateral connections. To fit the
        shapes of the lateral connection counterpart, it will expand
        parameters by concatting conv2d parameters and extra zero paddings.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + '.weight'
        conv2d_weight = state_dict_2d[weight_2d_name]
        old_shape = conv2d_weight.shape
        new_shape = conv3d.weight.data.shape
        kernel_t = new_shape[2]

        if new_shape[1] != old_shape[1]:
            if new_shape[1] < old_shape[1]:
                warnings.warn(f'The parameter of {module_name_2d} is not'
                              'loaded due to incompatible shapes. ')
                return
            # Inplanes may be different due to lateral connections
            new_channels = new_shape[1] - old_shape[1]
            pad_shape = old_shape
            pad_shape = pad_shape[:1] + (new_channels, ) + pad_shape[2:]
            # Expand parameters by concat extra channels
            conv2d_weight = torch.cat(
                (conv2d_weight,
                 torch.zeros(pad_shape).type_as(conv2d_weight).to(
                     conv2d_weight.device)),
                dim=1)

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
            conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

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

            if i != len(self.res_layers) and self.lateral:
                # No fusion needed in the final stage
                lateral_name = self.lateral_connections[i - 1]
                conv_lateral = getattr(self, lateral_name)
                conv_lateral.eval()
                for param in conv_lateral.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        # Override the init_weights of i3d
        super().init_weights()
        for module_name in self.lateral_connections:
            layer = getattr(self, module_name)
            for m in layer.modules():
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    kaiming_init(m)


pathway_cfg = {
    'resnet3d': ResNet3dPathway,
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

    pathway_cls = pathway_cfg[pathway_type]
    pathway = pathway_cls(*args, **kwargs, **cfg_)

    return pathway


@BACKBONES.register_module()
class ResNet3dSlowFast(nn.Module):
    """Slowfast backbone.

    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_

    Args:
        pretrained (str): The file path to a pretrained model.
        resample_rate (int): A large temporal stride ``resample_rate``
            on input frames. The actual resample rate is calculated by
            multipling the ``interval`` in ``SampleFrames`` in the
            pipeline with ``resample_rate``, equivalent to the :math:`\\tau`
            in the paper, i.e. it processes only one out of
            ``resample_rate * interval`` frames. Default: 8.
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
                 resample_rate=8,
                 speed_ratio=8,
                 channel_ratio=8,
                 slow_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=True,
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
                     pool1_stride_t=1)):
        super().__init__()
        self.pretrained = pretrained
        self.resample_rate = resample_rate
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        if slow_pathway['lateral']:
            slow_pathway['speed_ratio'] = speed_ratio
            slow_pathway['channel_ratio'] = channel_ratio

        self.slow_path = build_pathway(slow_pathway)
        self.fast_path = build_pathway(fast_pathway)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            # Directly load 3D model.
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            # Init two branch separately.
            self.fast_path.init_weights()
            self.slow_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples extracted
                by the backbone.
        """
        x_slow = nn.functional.interpolate(
            x,
            mode='nearest',
            scale_factor=(1.0 / self.resample_rate, 1.0, 1.0))
        x_slow = self.slow_path.conv1(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)

        x_fast = nn.functional.interpolate(
            x,
            mode='nearest',
            scale_factor=(1.0 / (self.resample_rate // self.speed_ratio), 1.0,
                          1.0))
        x_fast = self.fast_path.conv1(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)

        if self.slow_path.lateral:
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast)
            x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        for i, layer_name in enumerate(self.slow_path.res_layers):
            res_layer = getattr(self.slow_path, layer_name)
            x_slow = res_layer(x_slow)
            res_layer_fast = getattr(self.fast_path, layer_name)
            x_fast = res_layer_fast(x_fast)
            if (i != len(self.slow_path.res_layers) - 1
                    and self.slow_path.lateral):
                # No fusion needed in the final stage
                lateral_name = self.slow_path.lateral_connections[i]
                conv_lateral = getattr(self.slow_path, lateral_name)
                x_fast_lateral = conv_lateral(x_fast)
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        out = (x_slow, x_fast)

        return out


if mmdet_imported:
    MMDET_BACKBONES.register_module()(ResNet3dSlowFast)
