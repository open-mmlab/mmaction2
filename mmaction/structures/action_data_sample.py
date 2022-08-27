# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.structures import BaseDataElement, InstanceData, LabelData


class ActionDataSample(BaseDataElement):

    @property
    def gt_labels(self):
        return self._gt_labels

    @gt_labels.setter
    def gt_labels(self, value):
        self.set_field(value, '_gt_labels', LabelData)

    @gt_labels.deleter
    def gt_labels(self):
        del self._gt_labels

    @property
    def pred_scores(self):
        return self._pred_scores

    @pred_scores.setter
    def pred_scores(self, value):
        self.set_field(value, '_pred_scores', LabelData)

    @pred_scores.deleter
    def pred_scores(self):
        del self._pred_scores

    @property
    def proposals(self):
        return self._proposals

    @proposals.setter
    def proposals(self, value):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self):
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances
