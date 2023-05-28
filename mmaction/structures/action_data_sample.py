# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Sequence, Union

import numpy as np
import torch
from mmengine.structures import BaseDataElement, InstanceData, LabelData
from mmengine.utils import is_str


def format_label(value: Union[torch.Tensor, np.ndarray, Sequence,
                              int]) -> torch.Tensor:
    """Convert various python types to label-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.

    Returns:
        :obj:`torch.Tensor`: The foramtted label tensor.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


def format_score(value: Union[torch.Tensor, np.ndarray,
                              Sequence]) -> torch.Tensor:
    """Convert various python types to score-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence): Score values.

    Returns:
        :obj:`torch.Tensor`: The foramtted score tensor.
    """

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).float()
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


class ActionDataSample(BaseDataElement):

    def set_gt_labels(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number], Number]
    ) -> 'ActionDataSample':
        """Set label of ``gt_labels``."""
        label_data = getattr(self, '_gt_label', LabelData())
        label_data.item = format_label(value)
        self.gt_labels = label_data
        return self

    def set_pred_label(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number], Number]
    ) -> 'ActionDataSample':
        """Set label of ``pred_label``."""
        label_data = getattr(self, '_pred_label', LabelData())
        label_data.item = format_label(value)
        self.pred_labels = label_data
        return self

    def set_pred_score(self, value: torch.Tensor) -> 'ActionDataSample':
        """Set score of ``pred_label``."""
        label_data = getattr(self, '_pred_label', LabelData())
        label_data.item = format_score(value)
        if hasattr(self, 'num_classes'):
            assert len(label_data.item) == self.num_classes, \
                f'The length of score {len(label_data.item)} should be '\
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes',
                value=len(label_data.item),
                field_type='metainfo')
        self.pred_scores = label_data
        return self

    @property
    def gt_labels(self):
        """Property of `gt_labels`"""
        return self._gt_labels

    @gt_labels.setter
    def gt_labels(self, value):
        """Setter of `gt_labels`"""
        self.set_field(value, '_gt_labels', LabelData)

    @gt_labels.deleter
    def gt_labels(self):
        """Deleter of `gt_labels`"""
        del self._gt_labels

    @property
    def pred_scores(self):
        """Property of `pred_scores`"""
        return self._pred_scores

    @pred_scores.setter
    def pred_scores(self, value):
        """Setter of `pred_scores`"""
        self.set_field(value, '_pred_scores', LabelData)

    @pred_scores.deleter
    def pred_scores(self):
        """Deleter of `pred_scores`"""
        del self._pred_scores

    @property
    def pred_labels(self):
        """Property of `pred_labels`"""
        return self._pred_labels

    @pred_labels.setter
    def pred_labels(self, value):
        """Setter of `pred_labels`"""
        self.set_field(value, '_pred_labels', LabelData)

    @pred_labels.deleter
    def pred_labels(self):
        """Deleter of `pred_labels`"""
        del self._pred_labels

    @property
    def proposals(self):
        """Property of `proposals`"""
        return self._proposals

    @proposals.setter
    def proposals(self, value):
        """Setter of `proposals`"""
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        """Deleter of `proposals`"""
        del self._proposals

    @property
    def gt_instances(self):
        """Property of `gt_instances`"""
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value):
        """Setter of `gt_instances`"""
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        """Deleter of `gt_instances`"""
        del self._gt_instances

    @property
    def features(self):
        """Setter of `features`"""
        return self._features

    @features.setter
    def features(self, value):
        """Setter of `features`"""
        self.set_field(value, '_features', dtype=InstanceData)

    @features.deleter
    def features(self):
        """Deleter of `features`"""
        del self._features
