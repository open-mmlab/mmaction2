# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import LabelData
from torch import Tensor

from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, LabelList, OptConfigType,
                            OptMultiConfig, SampleList)


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - :meth:`init_weights`, initializing weights in some modules.
    - :meth:`forward`, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
        topk (int or tuple): Top-k accuracy. Default: (1, 5).
        average_clips (dict or ConfigDict, optional): Config for
            averaging class scores over multiple clips. Default: None.
        init_cfg (dict or ConfigDict, optional): Config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class: bool = False,
                 label_smooth_eps: float = 0.0,
                 topk: Union[int, Tuple[int]] = (1, 5),
                 average_clips: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(BaseHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = MODELS.build(loss_cls)
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
    def init_weights(self) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x, **kwargs) -> Tensor:
        """Defines the computation performed at every call."""
        raise NotImplementedError

    def loss(self, feats: Union[Tensor, Tuple[Tensor]],
             data_samples: SampleList, **kwargs) -> dict:
        """Perform forward propagation of head and loss calculation on the
        features of the upstream network.

        Args:
            feats (Tensor or Tuple[Tensor]): Features from upstream network.
            data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores: Union[Tensor, Tuple[Tensor]],
                     data_samples: SampleList) -> dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = [x.gt_labels.item for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses

    def predict(self, feats: Union[Tensor, Tuple[Tensor]],
                data_samples: SampleList, **kwargs) -> LabelList:
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (Tensor or Tuple[Tensor]): Features from upstream network.
            data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            List[:obj:`LabelData`]: Recognition results wrapped
                by :obj:`LabelData`. Each item usually contains
                following keys.

                - item (Tensor): Classification scores, has a shape
                    (num_classes, )
        """
        cls_scores = self(feats, **kwargs)
        return self.predict_by_feat(cls_scores, data_samples)

    def predict_by_feat(self, cls_scores: Tensor,
                        data_samples: SampleList) -> LabelList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (Tensor): Classification scores, has a shape
                    (num_classes, )
            data_samples (List[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_labels`.

        Returns:
            List[:obj:`LabelData`]: Recognition results wrapped
                by :obj:`LabelData`. Each item usually contains following
                keys.

                - item (Tensor): Classification scores, has a shape
                    (num_classes, )
        """
        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)

        predictions: LabelList = []
        for score in cls_scores:
            label = LabelData(item=score)
            predictions.append(label)
        return predictions

    def average_clip(self, cls_scores: Tensor, num_segs: int = 1) -> Tensor:
        """Averaging class scores over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score. Only called in test mode.

        Args:
            cls_scores (Tensor): Class scores to be averaged.
            num_segs (int): Number of clips for each input sample.

        Returns:
            Tensor: Averaged class scores.
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
