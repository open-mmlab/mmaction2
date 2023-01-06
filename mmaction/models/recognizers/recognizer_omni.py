# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence, Union

import torch
from mmengine.model import BaseModel

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, ForwardResults, SampleList


@MODELS.register_module()
class RecognizerOmni(BaseModel):
    """An Omni-souce recognizer model framework for joint-training of image and
    video recognition tasks.

    The `backbone` and `cls_head` should be able to accept both images and
    videos as inputs.
    """

    def __init__(self, backbone: ConfigType, cls_head: ConfigType,
                 data_preprocessor: ConfigType) -> None:
        super().__init__(data_preprocessor=data_preprocessor)
        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(cls_head)

    def forward(self, *data_samples, mode: str, **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes:

        - ``tensor``: Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - ``predict``: Forward and return the predictions, which are fully
        processed to a list of :obj:`ActionDataSample`.
        - ``loss``: Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            data_samples: should be a sequence of ``SampleList`` if
                ``mode="predict"`` or ``mode="loss"``. Each ``SampleList`` is
                the annotation data of one data source.
                It should be a single torch tensor if ``mode="tensor"``.
            mode (str): Return what kind of value. Defaults to ``tensor``.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of ``ActionDataSample``.
            - If ``mode="loss"``, return a dict of tensor.
        """

        if mode == 'loss' or mode == 'predict':
            if mode == 'loss':
                return self.loss(data_samples)
            return self.predict(data_samples)

        elif mode == 'tensor':

            assert isinstance(data_samples, torch.Tensor)

            data_ndim = data_samples.ndim
            if data_ndim not in [4, 5]:
                info = f'Input is a {data_ndim}D tensor. '
                info += 'Only 4D (BCHW) or 5D (BCTHW) tensors are supported!'
                raise ValueError(info)

            return self._forward(data_samples, **kwargs)

    def loss(self, data_samples: Sequence[SampleList]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            data_samples (Sequence[SampleList]): a sequence of SampleList. Each
                SampleList contains data samples from the same data source.

        Returns:
            dict: A dictionary of loss components.
        """
        loss_dict = {}
        for idx, data in enumerate(data_samples):
            inputs, data_samples = data['inputs'], data['data_samples']
            feats = self.extract_feat(inputs)
            loss_cls = self.cls_head.loss(feats, data_samples)
            for key in loss_cls:
                loss_dict[key + f'_{idx}'] = loss_cls[key]
        return loss_dict

    def predict(self, data_samples: Sequence[SampleList]) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            data_samples (Sequence[SampleList]): a sequence of SampleList. Each
                SampleList contains data samples from the same data source.

        Returns:
            List[``ActionDataSample``]: Return the recognition results.
            The returns value is ``ActionDataSample``, which usually contains
            ``pred_scores``. And the ``pred_scores`` usually contains
            following keys.

                - item (torch.Tensor): Classification scores, has a shape
                    (num_classes, )
        """
        assert len(data_samples) == 1
        feats = self.extract_feat(data_samples[0]['inputs'], test_mode=True)
        predictions = self.cls_head.predict(feats,
                                            data_samples[0]['data_samples'])
        return predictions

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
            return x
        else:
            # Return features extracted through backbone
            x = self.backbone(inputs)
            if stage == 'backbone':
                return x
            x = self.cls_head(x)
            return x
