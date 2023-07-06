# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import _load_checkpoint

from mmaction.registry import MODELS
from .mobilenet_v2 import InvertedResidual, MobileNetV2
from .resnet_tsm import TemporalShift


@MODELS.register_module()
class MobileNetV2TSM(MobileNetV2):
    """MobileNetV2 backbone for TSM.

    Args:
        num_segments (int): Number of frame segments. Defaults to 8.
        is_shift (bool): Whether to make temporal shift in reset layers.
            Defaults to True.
        shift_div (int): Number of div for shift. Defaults to 8.
        pretraind2d (bool): Whether to load pretrained 2D model.
            Defaults to True.
        **kwargs (keyword arguments, optional): Arguments for MobilNetV2.
    """

    def __init__(self,
                 num_segments=8,
                 is_shift=True,
                 shift_div=8,
                 pretrained2d=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.pretrained2d = pretrained2d
        self.init_structure()

    def make_temporal_shift(self):
        """Make temporal shift for some layers."""
        for m in self.modules():
            if isinstance(m, InvertedResidual) and \
                    len(m.conv) == 3 and m.use_res_connect:
                m.conv[0] = TemporalShift(
                    m.conv[0],
                    num_segments=self.num_segments,
                    shift_div=self.shift_div,
                )

    def init_structure(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if self.is_shift:
            self.make_temporal_shift()

    def load_original_weights(self, logger):
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
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()
