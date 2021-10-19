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
class VCOPHead(nn.Module):

    def __init__(self,
                 num_clips,
                 feature_size,
                 vcop_loss=dict(type='CrossEntropyLoss'),
                 **kwargs):
        
        super().__init__()
        self.tuple_len = num_clips
        self.feature_size = feature_size
        self.class_num = math.factorial(self.tuple_len)
        self.fc7 = nn.Linear(self.feature_size*2, 512)
        pair_num = int(self.tuple_len*(self.tuple_len-1)/2)
        self.fc8 = nn.Linear(512*pair_num, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.vcop_loss = build_loss(vcop_loss)

    def forward(self, tuple, return_loss=False):
        f = []  # clip features
        shuffle_order = np.random.permutation(self.tuple_len)
        for i in shuffle_order:
            clip = tuple[:, i, ...]
            f.append(clip)

        # Shuffling the input clip 
        
        order_index = self.order_class_index(shuffle_order)
        pf = []  # pairwise concat
        for i in range(self.tuple_len):
            for j in range(i+1, self.tuple_len):
                pf.append(torch.cat([f[i], f[j]], dim=2))

        pf = [self.fc7(i.reshape(-1, self.feature_size*2)) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc8(h)  # logits
        h = F.softmax(h, dim=1)
        if return_loss:
            vcop_loss = self.vcop_loss(h, torch.tensor(order_index).repeat(h.size()[0]).to(h.device)) 
            return {"vcop_loss": vcop_loss * self.vcop_loss.loss_weight}
        return h

    def order_class_index(self, order):
        """Return the index of the order in its full permutation.
        
        Args:
            order (tensor): e.g. [0,1,2]
        """
        classes = list(itertools.permutations(list(range(len(order)))))
        return classes.index(tuple(order.tolist()))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for params in self.parameters():
            normal_init(params, std=self.init_std)

        
        
