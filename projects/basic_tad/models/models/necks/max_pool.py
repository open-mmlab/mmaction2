from mmaction.registry import MODELS
from torch.nn import MaxPool3d


@MODELS.register_module()
class MaxPool3d(MaxPool3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
