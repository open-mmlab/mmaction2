# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import normal_init

from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import SampleList
from .base import BaseHead


@MODELS.register_module()
class RGBPoseHead(BaseHead):
    """The classification head for RGBPoseConv3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (tuple[int]): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to ``dict(type='CrossEntropyLoss')``.
        loss_components (list[str]): The components of the loss.
            Defaults to ``['rgb', 'pose']``.
        loss_weights (float or tuple[float]): The weights of the losses.
            Defaults to 1.
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Tuple[int],
                 loss_cls: Dict = dict(type='CrossEntropyLoss'),
                 loss_components: List[str] = ['rgb', 'pose'],
                 loss_weights: Union[float, Tuple[float]] = 1.,
                 dropout: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        if isinstance(dropout, float):
            dropout = {'rgb': dropout, 'pose': dropout}
        assert isinstance(dropout, dict)

        if loss_components is not None:
            self.loss_components = loss_components
            if isinstance(loss_weights, float):
                loss_weights = [loss_weights] * len(loss_components)
            assert len(loss_weights) == len(loss_components)
            self.loss_weights = loss_weights

        self.dropout = dropout
        self.init_std = init_std

        self.dropout_rgb = nn.Dropout(p=self.dropout['rgb'])
        self.dropout_pose = nn.Dropout(p=self.dropout['pose'])

        self.fc_rgb = nn.Linear(self.in_channels[0], num_classes)
        self.fc_pose = nn.Linear(self.in_channels[1], num_classes)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_rgb, std=self.init_std)
        normal_init(self.fc_pose, std=self.init_std)

    def forward(self, x: Tuple[torch.Tensor]) -> Dict:
        """Defines the computation performed at every call."""
        x_rgb, x_pose = self.avg_pool(x[0]), self.avg_pool(x[1])
        x_rgb = x_rgb.view(x_rgb.size(0), -1)
        x_pose = x_pose.view(x_pose.size(0), -1)

        x_rgb = self.dropout_rgb(x_rgb)
        x_pose = self.dropout_pose(x_pose)

        cls_scores = dict()
        cls_scores['rgb'] = self.fc_rgb(x_rgb)
        cls_scores['pose'] = self.fc_pose(x_pose)

        return cls_scores

    def loss(self, feats: Tuple[torch.Tensor], data_samples: SampleList,
             **kwargs) -> Dict:
        """Perform forward propagation of head and loss calculation on the
        features of the upstream network.

        Args:
            feats (tuple[torch.Tensor]): Features from upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores: Dict[str, torch.Tensor],
                     data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (dict[str, torch.Tensor]): The dict of
                classification scores,
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = torch.stack([x.gt_label for x in data_samples])
        labels = labels.squeeze()

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        losses = dict()
        for loss_name, weight in zip(self.loss_components, self.loss_weights):
            cls_score = cls_scores[loss_name]
            loss_cls = self.loss_by_scores(cls_score, labels)
            loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
            loss_cls[f'{loss_name}_loss_cls'] *= weight
            losses.update(loss_cls)
        return losses

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

    def predict(self, feats: Tuple[torch.Tensor], data_samples: SampleList,
                **kwargs) -> SampleList:
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (tuple[torch.Tensor]): Features from upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores = self(feats, **kwargs)
        return self.predict_by_feat(cls_scores, data_samples)

    def predict_by_feat(self, cls_scores: Dict[str, torch.Tensor],
                        data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (dict[str, torch.Tensor]): The dict of
                classification scores,
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        pred_scores = [dict() for _ in range(len(data_samples))]

        for name in self.loss_components:
            cls_score = cls_scores[name]
            cls_score = self.predict_by_scores(cls_score, data_samples)
            for pred_score, score in zip(pred_scores, cls_score):
                pred_score[f'{name}'] = score

        for data_sample, pred_score, in zip(data_samples, pred_scores):
            data_sample.set_pred_score(pred_score)
        return data_samples

    def predict_by_scores(self, cls_scores: torch.Tensor,
                          data_samples: SampleList) -> torch.Tensor:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The annotation
                data of every samples.

        Returns:
            torch.Tensor: The averaged classification scores.
        """

        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)
        return cls_scores
