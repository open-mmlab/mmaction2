# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmengine.model import BaseDataPreprocessor, ModuleDict

from mmaction.registry import MODELS


@MODELS.register_module()
class MultiModalDataPreprocessor(BaseDataPreprocessor):
    """Multi-Modal data pre-processor for action recognition tasks."""

    def __init__(self, preprocessors: Dict) -> None:
        super().__init__()
        self.preprocessors = ModuleDict()
        for name, pre_cfg in preprocessors.items():
            assert 'type' in pre_cfg, (
                'Each data preprocessor should contain the key type, '
                f'but got {pre_cfg}')
            self.preprocessors[name] = MODELS.build(pre_cfg)

    def forward(self, data: Dict, training: bool = False) -> Dict:
        """Preprocesses the data into the model input format.

        Args:
            data (dict): Data returned by dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        for modality, modality_data in inputs.items():
            preprocessor = self.preprocessors[modality]
            modality_data, data_samples = preprocessor.preprocess(
                modality_data, data_samples, training)
            inputs[modality] = modality_data

        data['inputs'] = inputs
        data['data_samples'] = data_samples
        return data
