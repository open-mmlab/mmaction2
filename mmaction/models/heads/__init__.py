# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseHead
from .gcn_head import GCNHead
from .i3d_head import I3DHead
from .mvit_head import MViTHead
from .slowfast_head import SlowFastHead
from .timesformer_head import TimeSformerHead
from .tpn_head import TPNHead
from .trn_head import TRNHead
from .tsm_head import TSMHead
from .tsn_audio_head import TSNAudioHead
from .tsn_head import TSNHead
from .x3d_head import X3DHead

__all__ = [
    'TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead', 'TPNHead',
    'X3DHead', 'TRNHead', 'TimeSformerHead', 'GCNHead', 'TSNAudioHead',
    'MViTHead'
]
