# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseHead
from .feature_head import FeatureHead
from .gcn_head import GCNHead
from .i3d_head import I3DHead
from .mvit_head import MViTHead
from .omni_head import OmniHead
from .rgbpose_head import RGBPoseHead
from .slowfast_head import SlowFastHead
from .timesformer_head import TimeSformerHead
from .tpn_head import TPNHead
from .trn_head import TRNHead
from .tsm_head import TSMHead
from .tsn_audio_head import TSNAudioHead
from .tsn_head import TSNHead
from .uniformer_head import UniFormerHead
from .x3d_head import X3DHead

__all__ = [
    'BaseHead', 'GCNHead', 'I3DHead', 'MViTHead', 'OmniHead', 'SlowFastHead',
    'TPNHead', 'TRNHead', 'TSMHead', 'TSNAudioHead', 'TSNHead',
    'TimeSformerHead', 'UniFormerHead', 'RGBPoseHead', 'X3DHead', 'FeatureHead'
]
