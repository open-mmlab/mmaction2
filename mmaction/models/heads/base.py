# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import LabelData

from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import ForwardResults, SampleList


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - :meth:`forward`, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Defaults to False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Defaults to 0.
        topk (int or tuple): Top-k accuracy. Defaults to ``(1, 5)``.
        average_clips (dict, optional): Config for averaging class
            scores over multiple clips. Defaults to None.
        loss_components (list[str], optional): The components of the loss.
            Defaults to None.
        loss_weights (float or tuple[float]): The weights of the losses.
            Defaults to 1.
        init_cfg (dict, optional): Config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: Dict = dict(
                     type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class: bool = False,
                 label_smooth_eps: float = 0.0,
                 topk: Union[int, Tuple[int]] = (1, 5),
                 average_clips: Optional[Dict] = None,
                 loss_components: Optional[List[str]] = None,
                 loss_weights: Union[float, Tuple[float]] = 1.,
                 init_cfg: Optional[Dict] = None) -> None:
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

        if loss_components is not None:
            self.loss_components = loss_components
            if isinstance(loss_weights, float):
                loss_weights = [loss_weights] * len(loss_components)

            assert len(loss_weights) == len(loss_components)
            self.loss_weights = loss_weights

    @abstractmethod
    def forward(self, x, **kwargs) -> ForwardResults:
        """Defines the computation performed at every call."""
        raise NotImplementedError

    def loss(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
             data_samples: SampleList, **kwargs) -> Dict:
        """Perform forward propagation of head and loss calculation on the
        features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores: Union[torch.Tensor, Dict[str,
                                                                torch.Tensor]],
                     data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor | dict[str, torch.Tensor]):
                Classification prediction results of all class,
                has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = torch.stack([x.gt_labels.item for x in data_samples])
        labels = labels.squeeze()

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if self.loss_components is not None:
            losses = dict()
            for loss_name, weight in zip(self.loss_components,
                                         self.loss_weights):
                cls_score = cls_scores[loss_name]
                loss_cls = self.loss_by_scores(cls_score, labels)
                loss_cls = {
                    loss_name + '_' + k: v
                    for k, v in loss_cls.items()
                }
                loss_cls[f'{loss_name}_loss_cls'] *= weight
                losses.update(loss_cls)
            return losses
        else:
            return self.loss_by_scores(cls_scores, labels)

    def loss_by_scores(self, cls_scores: torch.Tensor,
                       labels: torch.Tensor) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction
                results of all class, has shape (batch_size, num_classes).
            labels (torch.Tensor): The labels used to calculate the loss.

        Returns:
            dict: A dictionary of loss components.
        """
        losses = dict()
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

    def predict(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
                data_samples: SampleList, **kwargs) -> SampleList:
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (torch.Tensor or tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores = self(feats, **kwargs)
        return self.predict_by_feat(cls_scores, data_samples)

    def predict_by_feat(self, cls_scores: Union[torch.Tensor,
                                                Tuple[torch.Tensor]],
                        data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor | dict[str, torch.Tensor]):
                Classification scores, has a shape (num_classes, )
            data_samples (List[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_labels`.

        Returns:
            list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        pred_scores = [LabelData() for _ in range(len(data_samples))]
        pred_labels = [LabelData() for _ in range(len(data_samples))]

        if self.loss_components is not None:
            for name in self.loss_components:
                cls_score = cls_scores[name]
                cls_score, pred_label = \
                    self.predict_by_scores(cls_score, data_samples)
                for pred_score, pred_label, score, label in zip(
                        pred_scores, pred_labels, cls_score, pred_label):
                    pred_score.set_data({f'{name}': score})
                    pred_label.set_data({f'{name}': label})
        else:
            cls_score, pred_label = self.predict_by_scores(
                cls_scores, data_samples)
            for pred_score, pred_label, score, label in zip(
                    pred_scores, pred_labels, cls_score, pred_label):
                pred_score.set_data({'item': score})
                pred_label.set_data({'item': label})

        for data_sample, pred_score, pred_label in zip(data_samples,
                                                       pred_scores,
                                                       pred_labels):
            data_sample.pred_scores = pred_score
            data_sample.pred_labels = pred_label

        return data_samples

    def predict_by_scores(self, cls_scores: torch.Tensor,
                          data_samples: SampleList) -> Tuple:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores.
            data_samples (List[:obj:`ActionDataSample`]): The annotation
                data of every samples.

        Returns:
            tuple: A tuple of the averaged classification scores and
                prediction labels.
        """

        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)
        pred_labels = cls_scores.argmax(dim=-1, keepdim=True).detach()
        return cls_scores, pred_labels

    def average_clip(self,
                     cls_scores: torch.Tensor,
                     num_segs: int = 1) -> torch.Tensor:
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

        batch_size = cls_scores.shape[0]
        cls_scores = cls_scores.view(batch_size // num_segs, num_segs, -1)

        if self.average_clips is None:
            return cls_scores
        elif self.average_clips == 'prob':
            cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
        elif self.average_clips == 'score':
            cls_scores = cls_scores.mean(dim=1)

        return cls_scores
