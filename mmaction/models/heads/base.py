# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmengine.data import LabelData
import mmengine.dist as dist

from mmaction.core import top_k_accuracy, ActionDataSample
from ..builder import build_loss


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``forward``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
        topk (int | tuple): Top-k accuracy. Default: (1, 5).
        init_cfg (dict): the initialization config.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0,
                 topk=(1, 5),
                 average_clips=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        self.average_clips = average_clips
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    def predict(self, feats, data_samples, **kwargs) -> List[ActionDataSample]:
        cls_scores = self(feats, **kwargs)

        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)

        results = []
        for score in cls_scores:
            label = LabelData()
            label.item = score
            result = ActionDataSample()
            result.pred_scores = label
            results.append(result)

        return results

    def loss(self, feats, data_samples, **kwargs) -> Dict:
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            feats (torch.Tensor): The output features of the model.
            data_samples (List[ActionDataSample]): The data samples of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'topk_acc'(optional).
        """
        losses = dict()
        loss_aux = kwargs.pop('loss_aux', None)
        if loss_aux is not None:
            losses.update(loss_aux)

        cls_scores = self(feats, **kwargs)

        labels = [x.gt_labels.item for x in data_samples]
        labels = torch.stack(labels).to(feats.device)
        labels = labels.squeeze()

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data_samples))
        return outputs

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def average_clip(self, cls_scores, num_segs=1):
        """Averaging class scores over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score. Only called in test mode.

        Args:
            cls_scores (torch.Tensor): Class scores to be averaged.
            num_segs (int): Number of clips for each input sample.

        Returns:
            torch.Tensor: Averaged class scores.
        """

        if self.average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{self.average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        if self.average_clips is None:
            return cls_scores

        batch_size = cls_scores.shape[0]
        cls_scores = cls_scores.view(batch_size // num_segs, num_segs, -1)

        if self.average_clips == 'prob':
            cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
        elif self.average_clips == 'score':
            cls_scores = cls_scores.mean(dim=1)

        return cls_scores

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_distributed():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value, op='mean')
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
