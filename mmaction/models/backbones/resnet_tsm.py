# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, NonLocal3d
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import _load_checkpoint
from torch.nn.modules.utils import _ntuple

from mmaction.registry import MODELS
from .resnet import ResNet


class NL3DWrapper(nn.Module):
    """3D Non-local wrapper for ResNet50.

    Wrap ResNet layers with 3D NonLocal modules.

    Args:
        block (nn.Module): Residual blocks to be built.
        num_segments (int): Number of frame segments.
        non_local_cfg (dict): Config for non-local layers. Default: ``dict()``.
    """

    def __init__(self, block, num_segments, non_local_cfg=dict()):
        super(NL3DWrapper, self).__init__()
        self.block = block
        self.non_local_cfg = non_local_cfg
        self.non_local_block = NonLocal3d(self.block.conv3.norm.num_features,
                                          **self.non_local_cfg)
        self.num_segments = num_segments

    def forward(self, x):
        """Defines the computation performed at every call."""
        x = self.block(x)

        n, c, h, w = x.size()
        x = x.view(n // self.num_segments, self.num_segments, c, h,
                   w).transpose(1, 2).contiguous()
        x = self.non_local_block(x)
        x = x.transpose(1, 2).contiguous().view(n, c, h, w)
        return x


class TemporalShift(nn.Module):
    """Temporal shift module.

    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_

    Args:
        net (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of divisions for shift. Default: 8.
    """

    def __init__(self, net, num_segments=3, shift_div=8):
        super().__init__()
        self.net = net
        self.num_segments = num_segments
        self.shift_div = shift_div

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.shift(x, self.num_segments, shift_div=self.shift_div)
        return self.net(x)

    @staticmethod
    def shift(x, num_segments, shift_div=3):
        """Perform temporal shift operation on the feature.

        Args:
            x (torch.Tensor): The input feature to be shifted.
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 3.

        Returns:
            torch.Tensor: The shifted feature.
        """
        # [N, C, H, W]
        n, c, h, w = x.size()

        # [N // num_segments, num_segments, C, H*W]
        # can't use 5 dimensional array on PPL2D backend for caffe
        x = x.view(-1, num_segments, c, h * w)

        # get shift fold
        fold = c // shift_div

        # split c channel into three parts:
        # left_split, mid_split, right_split
        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold:2 * fold, :]
        right_split = x[:, :, 2 * fold:, :]

        # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
        # because array on caffe inference must be got by computing

        # shift left on num_segments channel in `left_split`
        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, blank), 1)

        # shift right on num_segments channel in `mid_split`
        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((blank, mid_split), 1)

        # right_split: no shift

        # concatenate
        out = torch.cat((left_split, mid_split, right_split), 2)

        # [N, C, H, W]
        # restore the original dimension
        return out.view(n, c, h, w)


@MODELS.register_module()
class ResNetTSM(ResNet):
    """ResNet backbone for TSM.

    Args:
        num_segments (int): Number of frame segments. Defaults to 8.
        is_shift (bool): Whether to make temporal shift in reset layers.
            Defaults to True.
        non_local (Sequence[int]): Determine whether to apply non-local module
            in the corresponding block of each stages.
            Defaults to (0, 0, 0, 0).
        non_local_cfg (dict): Config for non-local module.
            Defaults to ``dict()``.
        shift_div (int): Number of div for shift. Defaults to 8.
        shift_place (str): Places in resnet layers for shift, which is chosen
            from ['block', 'blockres'].
            If set to 'block', it will apply temporal shift to all child blocks
            in each resnet layer.
            If set to 'blockres', it will apply temporal shift to each `conv1`
            layer of all child blocks in each resnet layer.
            Defaults to 'blockres'.
        temporal_pool (bool): Whether to add temporal pooling.
            Defaults to False.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Defaults to True.
        **kwargs (keyword arguments, optional): Arguments for ResNet.
    """

    def __init__(self,
                 depth,
                 num_segments=8,
                 is_shift=True,
                 non_local=(0, 0, 0, 0),
                 non_local_cfg=dict(),
                 shift_div=8,
                 shift_place='blockres',
                 temporal_pool=False,
                 pretrained2d=True,
                 **kwargs):
        super().__init__(depth, **kwargs)
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        self.non_local_stages = _ntuple(self.num_stages)(non_local)
        self.non_local_cfg = non_local_cfg
        self.pretrained2d = pretrained2d
        self.init_structure()

    def init_structure(self):
        """Initialize structure for tsm."""
        if self.is_shift:
            self.make_temporal_shift()
        if len(self.non_local_cfg) != 0:
            self.make_non_local()
        if self.temporal_pool:
            self.make_temporal_pool()

    def make_temporal_shift(self):
        """Make temporal shift for some layers."""
        if self.temporal_pool:
            num_segment_list = [
                self.num_segments, self.num_segments // 2,
                self.num_segments // 2, self.num_segments // 2
            ]
        else:
            num_segment_list = [self.num_segments] * 4
        if num_segment_list[-1] <= 0:
            raise ValueError('num_segment_list[-1] must be positive')

        if self.shift_place == 'block':

            def make_block_temporal(stage, num_segments):
                """Make temporal shift on some blocks.

                Args:
                    stage (nn.Module): Model layers to be shifted.
                    num_segments (int): Number of frame segments.

                Returns:
                    nn.Module: The shifted blocks.
                """
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(
                        b, num_segments=num_segments, shift_div=self.shift_div)
                return nn.Sequential(*blocks)

            self.layer1 = make_block_temporal(self.layer1, num_segment_list[0])
            self.layer2 = make_block_temporal(self.layer2, num_segment_list[1])
            self.layer3 = make_block_temporal(self.layer3, num_segment_list[2])
            self.layer4 = make_block_temporal(self.layer4, num_segment_list[3])

        elif 'blockres' in self.shift_place:
            n_round = 1
            if len(list(self.layer3.children())) >= 23:
                n_round = 2

            def make_block_temporal(stage, num_segments):
                """Make temporal shift on some blocks.

                Args:
                    stage (nn.Module): Model layers to be shifted.
                    num_segments (int): Number of frame segments.

                Returns:
                    nn.Module: The shifted blocks.
                """
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1.conv = TemporalShift(
                            b.conv1.conv,
                            num_segments=num_segments,
                            shift_div=self.shift_div)
                return nn.Sequential(*blocks)

            self.layer1 = make_block_temporal(self.layer1, num_segment_list[0])
            self.layer2 = make_block_temporal(self.layer2, num_segment_list[1])
            self.layer3 = make_block_temporal(self.layer3, num_segment_list[2])
            self.layer4 = make_block_temporal(self.layer4, num_segment_list[3])

        else:
            raise NotImplementedError

    def make_temporal_pool(self):
        """Make temporal pooling between layer1 and layer2, using a 3D max
        pooling layer."""

        class TemporalPool(nn.Module):
            """Temporal pool module.

            Wrap layer2 in ResNet50 with a 3D max pooling layer.

            Args:
                net (nn.Module): Module to make temporal pool.
                num_segments (int): Number of frame segments.
            """

            def __init__(self, net, num_segments):
                super().__init__()
                self.net = net
                self.num_segments = num_segments
                self.max_pool3d = nn.MaxPool3d(
                    kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

            def forward(self, x):
                """Defines the computation performed at every call."""
                # [N, C, H, W]
                n, c, h, w = x.size()
                # [N // num_segments, C, num_segments, H, W]
                x = x.view(n // self.num_segments, self.num_segments, c, h,
                           w).transpose(1, 2)
                # [N // num_segmnets, C, num_segments // 2, H, W]
                x = self.max_pool3d(x)
                # [N // 2, C, H, W]
                x = x.transpose(1, 2).contiguous().view(n // 2, c, h, w)
                return self.net(x)

        self.layer2 = TemporalPool(self.layer2, self.num_segments)

    def make_non_local(self):
        """Wrap resnet layer into non local wrapper."""
        # This part is for ResNet50
        for i in range(self.num_stages):
            non_local_stage = self.non_local_stages[i]
            if sum(non_local_stage) == 0:
                continue

            layer_name = f'layer{i + 1}'
            res_layer = getattr(self, layer_name)

            for idx, non_local in enumerate(non_local_stage):
                if non_local:
                    res_layer[idx] = NL3DWrapper(res_layer[idx],
                                                 self.num_segments,
                                                 self.non_local_cfg)

    def _get_wrap_prefix(self):
        return ['.net', '.block']

    def load_original_weights(self, logger):
        """Load weights from original checkpoint, which required converting
        keys."""
        state_dict_torchvision = _load_checkpoint(
            self.pretrained, map_location='cpu')
        if 'state_dict' in state_dict_torchvision:
            state_dict_torchvision = state_dict_torchvision['state_dict']

        wrapped_layers_map = dict()
        for name, module in self.named_modules():
            # convert torchvision keys
            ori_name = name
            for wrap_prefix in self._get_wrap_prefix():
                if wrap_prefix in ori_name:
                    ori_name = ori_name.replace(wrap_prefix, '')
                    wrapped_layers_map[ori_name] = name

            if isinstance(module, ConvModule):
                if 'downsample' in ori_name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    tv_conv_name = ori_name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    tv_bn_name = ori_name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    tv_conv_name = ori_name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    tv_bn_name = ori_name.replace('conv', 'bn')

                for conv_param in ['.weight', '.bias']:
                    if tv_conv_name + conv_param in state_dict_torchvision:
                        state_dict_torchvision[ori_name+'.conv'+conv_param] = \
                            state_dict_torchvision.pop(tv_conv_name+conv_param)

                for bn_param in [
                        '.weight', '.bias', '.running_mean', '.running_var'
                ]:
                    if tv_bn_name + bn_param in state_dict_torchvision:
                        state_dict_torchvision[ori_name+'.bn'+bn_param] = \
                            state_dict_torchvision.pop(tv_bn_name+bn_param)

        # convert wrapped keys
        for param_name in list(state_dict_torchvision.keys()):
            layer_name = '.'.join(param_name.split('.')[:-1])
            if layer_name in wrapped_layers_map:
                wrapped_name = param_name.replace(
                    layer_name, wrapped_layers_map[layer_name])
                print(f'wrapped_name {wrapped_name}')
                state_dict_torchvision[
                    wrapped_name] = state_dict_torchvision.pop(param_name)

        msg = self.load_state_dict(state_dict_torchvision, strict=False)
        logger.info(msg)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if self.pretrained2d:
            logger = MMLogger.get_current_instance()
            self.load_original_weights(logger)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()
