import torch.nn as nn

from ..registry import HEADS


@HEADS.register_module()
class NullHead(nn.Module):

    def forward(self, x):
        return x
