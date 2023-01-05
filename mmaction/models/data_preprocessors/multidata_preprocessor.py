# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
from mmengine.model import BaseDataPreprocessor

from mmaction.registry import MODELS
from mmaction.utils import ConfigType


@MODELS.register_module()
class MultiDataPreprocessor(BaseDataPreprocessor):
    """Data pre-processor for multi-dataset tasks.

    Args:
        config_lists (List[Union[ConfigType, nn.Module]]): a list of data
            pre-processor configs or nn.Module. The length of config_lists
            should equal to the number of dataloader you use. The order of
            the configs should match the dataloader order in
            `MultiLoaderEpochBasedTrainLoop`.
    """

    def __init__(self, config_lists: List[Union[ConfigType,
                                                nn.Module]]) -> None:
        super().__init__()
        module_list = []
        for config in config_lists:
            if isinstance(config, nn.Module):
                data_preprocessor = config
            else:
                data_preprocessor = MODELS.build(config)
            module_list.append(data_preprocessor)

        self.data_preprocessors = nn.ModuleList(module_list)
        self.num_preprocessors = len(config_lists)

    def forward(self, data: Tuple[dict], training: bool = False) -> List[dict]:
        """Perform data pre-processor for multi-dataset tasks.

        Args:
            data (Tuple[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            List[dict]: Data in the same format as the model
                input.
        """
        if not isinstance(data, tuple):
            raise TypeError('Input should be a tuple of dict!')
        elif len(data) != self.num_preprocessors:
            raise ValueError('Input length should equal to the number of'
                             'data pre-processor!')
        output = []
        for idx, data_sample in enumerate(data):
            data_sample = self.data_preprocessors[idx](data_sample)
            output.append(data_sample)
        return data_sample
