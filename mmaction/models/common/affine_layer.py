import torch
import torch.nn as nn
from mmcv.cnn import NORM_LAYERS
from torch.nn.parameter import Parameter


@NORM_LAYERS.register_module()
class AffineLayer(nn.Module):

    def __init__(self, num_features, **cfg):
        super().__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.init_weights()

    def init_weights(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # [N, C, T, H, W]
        x = x * self.weight.view((-1, 1, 1, 1))
        x = x + self.bias.view((-1, 1, 1, 1))
        return x
