# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.structures import BaseDataElement, InstanceData, LabelData


class ActionDataSample(BaseDataElement):

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
