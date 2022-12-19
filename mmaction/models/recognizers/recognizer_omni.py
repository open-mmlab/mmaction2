# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Union

import torch
from mmengine.model import BaseModel

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, ForwardResults


@MODELS.register_module()
class RecognizerOmni(BaseModel):
    """"""

    def __init__(self, backbone: ConfigType, cls_head: ConfigType,
                 image_preprocessor: ConfigType,
                 video_preprocessor: ConfigType):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(cls_head)
        self.image_preprocessor = MODELS.build(image_preprocessor)
        self.video_preprocessor = MODELS.build(video_preprocessor)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        pass

    def forward(self, *args, mode, **kwargs):
        if mode == 'loss' or mode == 'predict':
            preprocessed = []
            for item in args:
                assert type(item) == dict
                if len(item['inputs'][0].shape) == 3:
                    item = self.image_preprocessor(item, self.training)
                else:
                    item = self.video_preprocessor(item, self.training)
                preprocessed.append(item)
            if mode == 'loss':
                return self.loss(*preprocessed, **kwargs)
            return self.predict(*preprocessed, **kwargs)

        elif mode == 'tensor':
            assert len(args) == 1 and isinstance(args[0], torch.Tensor)
            inputs = args[0]
            if len(inputs.shape) == 4:
                print('Input a 4D tensor, using image mode.')
            elif len(inputs.shape) == 5:
                print('Input a 5D tensor, using video mode.')
            else:
                info = 'Input is a %dD tensor. ' % len(inputs.shape)
                info += 'Only 4D (BCHW) or 5D (BCTHW) tensors are supported!'
                raise ValueError(info)
            return self._forward(inputs, **kwargs)

    def loss(self, *data_samples, **kwargs):
        loss_dict = {}
        for idx, data in enumerate(data_samples):
            inputs, data_samples = data['inputs'], data['data_samples']
            feats = self.extract_feat(inputs)
            loss_cls = self.cls_head.loss(feats, data_samples)
            for key in loss_cls:
                loss_dict[key + f'_{idx}'] = loss_cls[key]
        return loss_dict

    def predict(self, *data_samples, **kwargs):
        pass

    def _forward(self,
                 inputs: torch.Tensor,
                 stage: str = 'backbone',
                 **kwargs) -> ForwardResults:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
            stage (str): Which stage to output the features.

        Returns:
            Union[tuple, torch.Tensor]: Features from ``backbone`` or ``head``
            forward.
        """
        feats, _ = self.extract_feat(inputs, stage=stage)
        return feats

    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`
        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.
        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            data = [data]
            results = self(*data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError
        return results

    def extract_feat(self,
                     inputs: torch.Tensor,
                     stage: str = 'backbone',
                     test_mode: bool = False) -> tuple:
        """Extract features of different stages.

        Args:
            inputs (torch.Tensor): The input data.
            stage (str): Which stage to output the feature.
                Defaults to ``'backbone'``.
            test_mode (bool): Whether in test mode. Defaults to False.

        Returns:
                torch.Tensor: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline. These keys are usually included:
                    ``loss_aux``.
        """

        if len(inputs.shape) == 6:
            inputs = inputs.view((-1, ) + inputs.shape[2:])

        # Check settings of test
        if test_mode:
            x = self.backbone(inputs)
        else:
            # Return features extracted through backbone
            x = self.backbone(inputs)
            if stage == 'backbone':
                return x
            x = self.cls_head(x)
            return x
