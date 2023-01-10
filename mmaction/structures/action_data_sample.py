# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import numpy as np
import torch
from mmengine.structures import BaseDataElement, InstanceData, LabelData


class ActionDataSample(BaseDataElement):

    def set_gt_labels(self, value: Union[int,
                                         np.ndarray]) -> 'ActionDataSample':
        """Set label of ``gt_labels``."""
        if isinstance(value, int):
            value = torch.LongTensor([value])
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        else:
            raise TypeError(f'Type {type(value)} is not an '
                            f'available label type.')

        self.gt_labels = LabelData(item=value)
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
