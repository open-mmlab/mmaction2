# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_with_prefix)
from mmpretrain.models import MobileOne

from mmaction.registry import MODELS
from .resnet_tsm import TemporalShift


@MODELS.register_module()
class MobileOneTSM(MobileOne):
    """MobileOne backbone for TSM.

    Args:
        arch (str | dict): MobileOne architecture. If use string, choose
            from 's0', 's1', 's2', 's3' and 's4'. If use dict, it should
            have below keys:

            - num_blocks (Sequence[int]): Number of blocks in each stage.
            - width_factor (Sequence[float]): Width factor in each stage.
            - num_conv_branches (Sequence[int]): Number of conv branches
              in each stage.
            - num_se_blocks (Sequence[int]): Number of SE layers in each
              stage, all the SE layers are placed in the subsequent order
              in each stage.

            Defaults to 's0'.
        num_segments (int): Number of frame segments. Defaults to 8.
        is_shift (bool): Whether to make temporal shift in reset layers.
            Defaults to True.
        shift_div (int): Number of div for shift. Defaults to 8.
        pretraind2d (bool): Whether to load pretrained 2D model.
            Defaults to True.
        **kwargs (keyword arguments, optional): Arguments for MobileOne.
    """

    def __init__(self,
                 arch: str,
                 num_segments: int = 8,
                 is_shift: bool = True,
                 shift_div: int = 8,
                 pretrained2d: bool = True,
                 **kwargs):
        super().__init__(arch, **kwargs)
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.pretrained2d = pretrained2d
        self.init_structure()

    def make_temporal_shift(self):
        """Make temporal shift for some layers.

        To make reparameterization work, we can only build the shift layer
        before the 'block', instead of the 'blockres'
        """

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

        self.stage0 = make_block_temporal(
            nn.Sequential(self.stage0), self.num_segments)[0]
        for i in range(1, 5):
            temporal_stage = make_block_temporal(
                getattr(self, f'stage{i}'), self.num_segments)
            setattr(self, f'stage{i}', temporal_stage)

    def init_structure(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if self.is_shift:
            self.make_temporal_shift()

    def load_original_weights(self, logger):
        assert self.init_cfg.get('type') == 'Pretrained', (
            'Please specify '
            'init_cfg to use pretrained 2d checkpoint')
        self.pretrained = self.init_cfg.get('checkpoint')
        prefix = self.init_cfg.get('prefix')
        if prefix is not None:
            original_state_dict = _load_checkpoint_with_prefix(
                prefix, self.pretrained, map_location='cpu')
        else:
            original_state_dict = _load_checkpoint(
                self.pretrained, map_location='cpu')
        if 'state_dict' in original_state_dict:
            original_state_dict = original_state_dict['state_dict']

        wrapped_layers_map = dict()
        for name, module in self.named_modules():
            ori_name = name
            for wrap_prefix in ['.net']:
                if wrap_prefix in ori_name:
                    ori_name = ori_name.replace(wrap_prefix, '')
                    wrapped_layers_map[ori_name] = name

        # convert wrapped keys
        for param_name in list(original_state_dict.keys()):
            layer_name = '.'.join(param_name.split('.')[:-1])
            if layer_name in wrapped_layers_map:
                wrapped_name = param_name.replace(
                    layer_name, wrapped_layers_map[layer_name])
                original_state_dict[wrapped_name] = original_state_dict.pop(
                    param_name)

        msg = self.load_state_dict(original_state_dict, strict=True)
        logger.info(msg)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if self.pretrained2d:
            logger = MMLogger.get_current_instance()
            self.load_original_weights(logger)
        else:
            super().init_weights()

    def forward(self, x):
        """unpack tuple result."""
        x = super().forward(x)
        if isinstance(x, tuple):
            assert len(x) == 1
            x = x[0]
        return x
