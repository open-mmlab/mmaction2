# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class SlowFastSelfSupervisedLoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.5):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
    
    def _calculate_cosine_similarity(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt


    def _forward(self, slow_features, fast_features):
        batch_size = slow_features.shape[0]
        similarity = self._calculate_cosine_similarity(slow_features, fast_features) 
        similarity = similarity / self.temperature
        similarity = similarity.exp() 
        mask = torch.eye(batch_size, dtype=torch.bool)
        positives = similarity[mask].sum(axis=-1) 
        negatives = similarity[~mask].sum(axis=-1) 
        loss = - torch.log(positives / (positives + negatives + 1e-8))  
        ret_dict = {'slowfast_selfsupervised_loss': self.loss_weight * loss} 
        return ret_dict 
    