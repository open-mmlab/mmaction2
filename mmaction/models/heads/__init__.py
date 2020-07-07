from .base import BaseHead
from .i3d_head import I3DHead
from .slowfast_head import SlowFastHead
from .tsm_head import TSMHead
from .tsn_head import TSNHead

__all__ = ['TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead']
