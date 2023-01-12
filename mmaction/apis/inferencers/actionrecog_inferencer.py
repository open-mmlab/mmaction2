# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmaction.structures import ActionDataSample
from .base_mmaction_inferencer import BaseMMAction2Inferencer


class ActionRecogInferencer(BaseMMAction2Inferencer):

    def pred2dict(self, data_sample: ActionDataSample) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (ActionDataSample): The data sample to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        result['pred_labels'] = data_sample.pred_labels.item.tolist()
        result['pred_scores'] = data_sample.pred_scores.item.tolist()
        return result
