from mmcv.cnn import ATTENTION
from mmcv.runner.base_module import BaseModule


@ATTENTION.register_module()
class SpaceAttention(BaseModule):
    pass


@ATTENTION.register_module()
class TimeAttention(BaseModule):
    pass


@ATTENTION.register_module()
class SpaceTimeAttention(BaseModule):
    pass
