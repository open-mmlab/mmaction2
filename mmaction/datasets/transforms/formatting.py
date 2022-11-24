# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Dict

import numpy as np
import torch
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import InstanceData, LabelData

from mmaction.registry import TRANSFORMS
from mmaction.structures import ActionDataSample


@TRANSFORMS.register_module()
class PackActionInputs(BaseTransform):
    """Pack the inputs data for the recognition.

    Args:
        meta_keys (Sequence[str]): The meta keys to saved in the
            ``metainfo`` of the packed ``data_sample``.
            Defaults to ``('img_shape', 'img_key', 'video_id', 'timestamp')``.
    """

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_labels': 'labels',
    }

    def __init__(
        self,
        meta_keys: Sequence[str] = ('img_shape', 'img_key', 'video_id',
                                    'timestamp')
    ) -> None:
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:
                - 'inputs' (torch.Tensor): The forward data of models.
                - 'data_sample' (:obj:`ActionDataSample`): The annotation
                  info of the sample.
        """
        packed_results = dict()
        if 'imgs' in results:
            imgs = results['imgs']
            packed_results['inputs'] = to_tensor(imgs)
        elif 'keypoint' in results:
            keypoint = results['keypoint']
            packed_results['inputs'] = to_tensor(keypoint)
        elif 'audios' in results:
            audios = results['audios']
            packed_results['inputs'] = to_tensor(audios)
        else:
            raise ValueError(
                'Cannot get `imgs`, `keypoint` or `audios` in the input dict '
                'of `PackActionInputs`.')

        data_sample = ActionDataSample()

        if 'gt_bboxes' in results:
            instance_data = InstanceData()
            for key in self.mapping_table.keys():
                instance_data[self.mapping_table[key]] = to_tensor(
                    results[key])
            data_sample.gt_instances = instance_data

            if 'proposals' in results:
                data_sample.proposals = InstanceData(
                    bboxes=to_tensor(results['proposals']))
        else:
            label_data = LabelData()
            label_data.item = to_tensor(results['label'])
            data_sample.gt_labels = label_data

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackLocalizationInputs(BaseTransform):

    def __init__(self, keys=(), meta_keys=('video_name', )):
        self.keys = keys
        self.meta_keys = meta_keys

    def transform(self, results):
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_samples' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'raw_feature' in results:
            raw_feature = results['raw_feature']
            packed_results['inputs'] = to_tensor(raw_feature)
        elif 'bsp_feature' in results:
            packed_results['inputs'] = torch.tensor(0.)
        else:
            raise ValueError(
                'Cannot get "raw_feature" or "bsp_feature" in the input '
                'dict of `PackActionInputs`.')

        data_sample = ActionDataSample()
        instance_data = InstanceData()
        for key in self.keys:
            if key in results:
                instance_data[key] = to_tensor(results[key])
        data_sample.gt_instances = instance_data

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class Rename(BaseTransform):
    """Rename the key in results.

    Args:
        mapping (dict): The keys in results that need to be renamed. The key of
            the dict is the original name, while the value is the new name. If
            the original name not found in results, do nothing.
            Defaults to ``dict()``.
    """

    def __init__(self, mapping: Dict = dict()) -> None:
        self.mapping = mapping

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`Rename`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        for key, value in self.mapping.items():
            if key in results:
                assert isinstance(key, str) and isinstance(value, str)
                assert value not in results, ('the new name already exists in '
                                              'results')
                results[value] = results[key]
                results.pop(key)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mapping={self.mapping})')
        return repr_str


@TRANSFORMS.register_module()
class Transpose(BaseTransform):
    """Transpose image channels to a given order.

    Args:
        keys (Sequence[str]): Required keys to be converted.
        order (Sequence[int]): Image channel order.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def transform(self, results):
        """Performs the Transpose formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, order={self.order})')


@TRANSFORMS.register_module()
class FormatShape(BaseTransform):
    """Format final imgs shape to the given input_format.

    Required keys:
        - imgs
        - num_clips
        - clip_len

    Modified Keys:
        - img
        - input_shape

    Args:
        input_format (str): Define the final imgs format.
        collapse (bool): To collapse input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Defaults to False.
    """

    def __init__(self, input_format: str, collapse: bool = False) -> None:
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def transform(self, results: dict) -> dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])
        imgs = results['imgs']
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * T
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            num_clips = results['num_clips']
            clip_len = results['clip_len']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x H x W x C
            imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
            # N_crops x N_clips x C x T x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x T x H x W
            # M' = N_crops x N_clips
        elif self.input_format == 'NCHW':
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            # M x C x H x W
        elif self.input_format == 'NCHW_Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            imgs = np.stack([x_flow, y_flow], axis=-1)

            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x H x W x C
            imgs = np.transpose(imgs, (0, 1, 2, 5, 3, 4))
            # N_crops x N_clips x T x C x H x W
            imgs = imgs.reshape((-1, imgs.shape[2] * imgs.shape[3]) +
                                imgs.shape[4:])
            # M' x C' x H x W
            # M' = N_crops x N_clips
            # C' = T x C
        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x T
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W

        if self.collapse:
            assert imgs.shape[0] == 1
            imgs = imgs.squeeze(0)

        results['imgs'] = imgs
        results['input_shape'] = imgs.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


@TRANSFORMS.register_module()
class FormatAudioShape(BaseTransform):
    """Format final audio shape to the given input_format.

    Required keys are ``audios``, ``num_clips`` and ``clip_len``, added or
    modified keys are ``audios`` and ``input_shape``.

    Args:
        input_format (str): Define the final imgs format.
    """

    def __init__(self, input_format: str) -> None:
        self.input_format = input_format
        if self.input_format not in ['NCTF']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def transform(self, results: dict) -> dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        audios = results['audios']
        # clip x sample x freq -> clip x channel x sample x freq
        clip, sample, freq = audios.shape
        audios = audios.reshape(clip, 1, sample, freq)
        results['audios'] = audios
        results['input_shape'] = audios.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


@TRANSFORMS.register_module()
class FormatGCNInput(BaseTransform):
    """Format final skeleton shape to the given ``input_format``.

    Required Keys:

    - keypoint
    - keypoint_score (optional)

    Modified Key:

    - keypoint

    Added Key:

    - input_shape

    Args:
        input_format (str): Define the final skeleton format.
    """

    def __init__(self, input_format: str, num_person: int = 2) -> None:
        self.input_format = input_format
        if self.input_format not in ['NCTVM']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')
        self.num_person = num_person

    def transform(self, results: dict) -> dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results['keypoint']

        if 'keypoint_score' in results:
            keypoint_confidence = results['keypoint_score']
            keypoint_confidence = np.expand_dims(keypoint_confidence, -1)
            keypoint_3d = np.concatenate((keypoint, keypoint_confidence),
                                         axis=-1)
        else:
            keypoint_3d = keypoint

        keypoint_3d = np.transpose(keypoint_3d,
                                   (3, 1, 2, 0))  # M T V C -> C T V M

        if keypoint_3d.shape[-1] < self.num_person:
            pad_dim = self.num_person - keypoint_3d.shape[-1]
            pad = np.zeros(
                keypoint_3d.shape[:-1] + (pad_dim, ), dtype=keypoint_3d.dtype)
            keypoint_3d = np.concatenate((keypoint_3d, pad), axis=-1)
        elif keypoint_3d.shape[-1] > self.num_person:
            keypoint_3d = keypoint_3d[:, :, :, :self.num_person]

        results['keypoint'] = keypoint_3d
        results['input_shape'] = keypoint_3d.shape
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'input_format={self.input_format}, '
                    f'num_person={self.num_person})')
        return repr_str
