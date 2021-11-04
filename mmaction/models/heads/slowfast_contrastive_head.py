import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import torch.nn.functional as F

from ..builder import HEADS
from .base import AvgConsensus, BaseHead
from ..builder import build_loss
import math
import numpy as np 
import itertools

@HEADS.register_module()
class SlowFastContrastiveHead(nn.Module):
    def __init__(self,
                 feature_size,
                 contrastive_loss=dict(type='SlowFastSelfSupervisedLoss'),
                 init_std=0.001,
                 **kwargs):

        super().__init__()
        self.fc1 = nn.Linear(feature_size, 2048)
        self.relu_1 = nn.ReLU(inplace=True) 
        self.fc2 = nn.Linear(2048, 512)
        self.relu_2 = nn.ReLU(inplace=True) 
        self.encoder = nn.Sequential(self.fc1, self.relu_1, self.fc2, self.relu_2)
        self.loss = build_loss(contrastive_loss)
        self.init_std = init_std
        self.init_weights() 


    def forward(self, features):
        batch_size = features.shape[0]
        features = features.view(batch_size, -1) 
        features = self.encoder(features) 
        return features
    
    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.encoder, std=self.init_std)


@HEADS.register_module()
class TwoPathwayContrastiveHead(nn.Module):
    def __init__(self,
                 feature_size,
                 contrastive_loss=dict(type='SlowFastSelfSupervisedLoss'),
                 init_std=0.001,
                 **kwargs):

        super().__init__()
        self.fc1 = nn.Linear(feature_size, 2048)
        self.relu_1 = nn.ReLU(inplace=True) 
        self.fc2 = nn.Linear(2048, 512)
        self.relu_2 = nn.ReLU(inplace=True) 
        self.encoder = nn.Sequential(self.fc1, self.relu_1, self.fc2, self.relu_2)
        self.loss = build_loss(contrastive_loss)
        self.init_std = init_std
        self.init_weights() 


    def forward(self, features):
        batch_size = features.shape[0]
        features = features.view(batch_size, -1) 
        features = self.encoder(features) 
        return features
    
    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.encoder, std=self.init_std)