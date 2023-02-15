# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Dict

from mmengine.model import BaseDataPreprocessor, ModuleDict

from mmaction.registry import MODELS


@MODELS.register_module()
class MultiModalDataPreprocessor(BaseDataPreprocessor):
    """Multi-Modal data pre-processor for action recognition tasks."""

    def __init__(self, preprocessors: Optional[Dict] = None) -> None:
        super().__init__()
        self.preprocessors = ModuleDict()
        if preprocessors is not None:
            for name, pre_cfg in preprocessors.items():
                self.preprocessors[name] = MODELS.build(pre_cfg)

    def forward(self, data: Dict, training: bool = False) -> Dict:
        """Preprocesses the data into the model input format.

        Args:
            data (dict): Data returned by dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        inputs, data_samples = data['inputs'], data['data_samples']
        inputs = self.cast_data(inputs)
        for modality, modality_data in inputs.items():
            preprocessor = self.preprocessors[modality]
            modality_data, data_samples = preprocessor.preprocess(
                modality_data, data_samples, training)
            inputs[modality] = modality_data

        data['inputs'] = inputs
        data['data_samples'] = data_samples
        return data
