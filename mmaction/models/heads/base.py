from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgConsensus(nn.Module):
    """Average consensus module.

    Attributes:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super(AvgConsensus, self).__init__()
        self.dim = dim

    def forward(self, input):
        return input.mean(dim=self.dim, keepdim=True)


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
        Methods:`init_weights`, initializing weights in some modules.
        Methods:`forward`, supporting to forward both for training and testing.

    Attributes:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
    """

    def __init__(self, num_classes, in_channels):
        super(BaseHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def loss(self, cls_score, labels):
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        losses['loss_cls'] = F.cross_entropy(cls_score, labels)

        return losses
