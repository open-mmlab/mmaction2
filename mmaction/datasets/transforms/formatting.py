# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import InstanceData

from mmaction.registry import TRANSFORMS
from mmaction.structures import ActionDataSample


@TRANSFORMS.register_module()
class PackActionInputs(BaseTransform):
    """Pack the inputs data.

    Args:
        collect_keys (tuple[str], optional): The keys to be collected
            to ``packed_results['inputs']``. Defaults to ``
        meta_keys (Sequence[str]): The meta keys to saved in the
            `metainfo` of the `data_sample`.
            Defaults to ``('img_shape', 'img_key', 'video_id', 'timestamp')``.
        algorithm_keys (Sequence[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
    """

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_labels': 'labels',
    }

    def __init__(
            self,
            collect_keys: Optional[Tuple[str]] = None,
            meta_keys: Sequence[str] = ('img_shape', 'img_key', 'video_id',
                                        'timestamp'),
            algorithm_keys: Sequence[str] = (),
    ) -> None:
        self.collect_keys = collect_keys
        self.meta_keys = meta_keys
        self.algorithm_keys = algorithm_keys

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PackActionInputs`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        packed_results = dict()
        if self.collect_keys is not None:
            packed_results['inputs'] = dict()
            for key in self.collect_keys:
                packed_results['inputs'][key] = to_tensor(results[key])
        else:
            if 'imgs' in results:
                imgs = results['imgs']
                packed_results['inputs'] = to_tensor(imgs)
            elif 'heatmap_imgs' in results:
                heatmap_imgs = results['heatmap_imgs']
                packed_results['inputs'] = to_tensor(heatmap_imgs)
            elif 'keypoint' in results:
                keypoint = results['keypoint']
                packed_results['inputs'] = to_tensor(keypoint)
            elif 'audios' in results:
                audios = results['audios']
                packed_results['inputs'] = to_tensor(audios)
            elif 'text' in results:
                text = results['text']
                packed_results['inputs'] = to_tensor(text)
            else:
                raise ValueError(
                    'Cannot get `imgs`, `keypoint`, `heatmap_imgs`, '
                    '`audios` or `text` in the input dict of '
                    '`PackActionInputs`.')

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

        if 'label' in results:
            data_sample.set_gt_label(results['label'])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(collect_keys={self.collect_keys}, '
        repr_str += f'meta_keys={self.meta_keys})'
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
        for key in self.keys:
            if key not in results:
                continue
            elif key == 'proposals':
                instance_data = InstanceData()
                instance_data[key] = to_tensor(results[key])
                data_sample.proposals = instance_data
            else:
                if hasattr(data_sample, 'gt_instances'):
                    data_sample.gt_instances[key] = to_tensor(results[key])
                else:
                    instance_data = InstanceData()
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

        - imgs (optional)
        - heatmap_imgs (optional)
        - modality (optional)
        - num_clips
        - clip_len

    Modified Keys:

        - imgs

    Added Keys:

        - input_shape
        - heatmap_input_shape (optional)

    Args:
        input_format (str): Define the final data format.
        collapse (bool): To collapse input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Defaults to False.
    """

    def __init__(self, input_format: str, collapse: bool = False) -> None:
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in [
                'NCTHW', 'NCHW', 'NCTHW_Heatmap', 'NPTCHW'
        ]:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def transform(self, results: Dict) -> Dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])

        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * T
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            if 'imgs' in results:
                imgs = results['imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                if isinstance(clip_len, dict):
                    clip_len = clip_len['RGB']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x H x W x C
                imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['imgs'] = imgs
                results['input_shape'] = imgs.shape

            if 'heatmap_imgs' in results:
                imgs = results['heatmap_imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                # clip_len must be a dict
                clip_len = clip_len['Pose']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x C x H x W
                imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['heatmap_imgs'] = imgs
                results['heatmap_input_shape'] = imgs.shape

        elif self.input_format == 'NCTHW_Heatmap':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x C x H x W
            imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
            # N_crops x N_clips x C x T x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x T x H x W
            # M' = N_crops x N_clips
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NCHW':
            imgs = results['imgs']
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            if 'modality' in results and results['modality'] == 'Flow':
                clip_len = results['clip_len']
                imgs = imgs.reshape((-1, clip_len * imgs.shape[1]) +
                                    imgs.shape[2:])
            # M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x T
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        if self.collapse:
            assert results['imgs'].shape[0] == 1
            results['imgs'] = results['imgs'].squeeze(0)
            results['input_shape'] = results['imgs'].shape

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


@TRANSFORMS.register_module()
class FormatAudioShape(BaseTransform):
    """Format final audio shape to the given input_format.

    Required Keys:

        - audios

    Modified Keys:

        - audios

    Added Keys:

        - input_shape

    Args:
        input_format (str): Define the final imgs format.
    """

    def __init__(self, input_format: str) -> None:
        self.input_format = input_format
        if self.input_format not in ['NCTF']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def transform(self, results: Dict) -> Dict:
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

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


@TRANSFORMS.register_module()
class FormatGCNInput(BaseTransform):
    """Format final skeleton shape.

    Required Keys:

        - keypoint
        - keypoint_score (optional)
        - num_clips (optional)

    Modified Key:

        - keypoint

    Args:
        num_person (int): The maximum number of people. Defaults to 2.
        mode (str): The padding mode. Defaults to ``'zero'``.
    """

    def __init__(self, num_person: int = 2, mode: str = 'zero') -> None:
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`FormatGCNInput`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        keypoint = results['keypoint']
        if 'keypoint_score' in results:
            keypoint = np.concatenate(
                (keypoint, results['keypoint_score'][..., None]), axis=-1)

        cur_num_person = keypoint.shape[0]
        if cur_num_person < self.num_person:
            pad_dim = self.num_person - cur_num_person
            pad = np.zeros(
                (pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and cur_num_person == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif cur_num_person > self.num_person:
            keypoint = keypoint[:self.num_person]

        M, T, V, C = keypoint.shape
        nc = results.get('num_clips', 1)
        assert T % nc == 0
        keypoint = keypoint.reshape(
            (M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)

        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'num_person={self.num_person}, '
                    f'mode={self.mode})')
        return repr_str
