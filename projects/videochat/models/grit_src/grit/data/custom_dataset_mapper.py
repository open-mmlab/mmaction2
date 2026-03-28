# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Jialian Wu from https://github.com/facebookresearch/Detic/blob
# /main/detic/data/custom_dataset_mapper.py
import copy
import logging
from itertools import compress

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper

from .custom_build_augmentation import build_custom_augmentation

__all__ = ['CustomDatasetMapper', 'ObjDescription']
logger = logging.getLogger(__name__)


class CustomDatasetMapper(DatasetMapper):

    @configurable
    def __init__(self, is_train: bool, dataset_augs=[], **kwargs):
        if is_train:
            self.dataset_augs = [T.AugmentationList(x) for x in dataset_augs]
        super().__init__(is_train, **kwargs)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        if is_train:
            if cfg.INPUT.CUSTOM_AUG == 'EfficientDetResizeCrop':
                dataset_scales = cfg.DATALOADER.DATASET_INPUT_SCALE
                dataset_sizes = cfg.DATALOADER.DATASET_INPUT_SIZE
                ret['dataset_augs'] = [
                    build_custom_augmentation(cfg, True, scale, size)
                    for scale, size in zip(dataset_scales, dataset_sizes)
                ]
            else:
                assert cfg.INPUT.CUSTOM_AUG == 'ResizeShortestEdge'
                min_sizes = cfg.DATALOADER.DATASET_MIN_SIZES
                max_sizes = cfg.DATALOADER.DATASET_MAX_SIZES
                ret['dataset_augs'] = [
                    build_custom_augmentation(
                        cfg, True, min_size=mi, max_size=ma)
                    for mi, ma in zip(min_sizes, max_sizes)
                ]
        else:
            ret['dataset_augs'] = []

        return ret

    def __call__(self, dataset_dict):
        dataset_dict_out = self.prepare_data(dataset_dict)

        # When augmented image is too small, do re-augmentation
        retry = 0
        while (dataset_dict_out['image'].shape[1] < 32
               or dataset_dict_out['image'].shape[2] < 32):
            retry += 1
            if retry == 100:
                logger.info(
                    'Retry 100 times for augmentation. Make sure the image '
                    'size is not too small. ')
                logger.info('Find image information below')
                logger.info(dataset_dict)
            dataset_dict_out = self.prepare_data(dataset_dict)

        return dataset_dict_out

    def prepare_data(self, dataset_dict_in):
        dataset_dict = copy.deepcopy(dataset_dict_in)
        if 'file_name' in dataset_dict:
            ori_image = utils.read_image(
                dataset_dict['file_name'], format=self.image_format)
        else:
            ori_image, _, _ = self.tar_dataset[dataset_dict['tar_index']]
            ori_image = utils._apply_exif_orientation(ori_image)
            ori_image = utils.convert_PIL_to_numpy(ori_image,
                                                   self.image_format)
        utils.check_image_size(dataset_dict, ori_image)

        aug_input = T.AugInput(copy.deepcopy(ori_image), sem_seg=None)
        if self.is_train:
            transforms = \
                self.dataset_augs[dataset_dict['dataset_source']](aug_input)
        else:
            transforms = self.augmentations(aug_input)
        image, _ = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]
        dataset_dict['image'] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop('annotations', None)
            return dataset_dict

        if 'annotations' in dataset_dict:
            if len(dataset_dict['annotations']) > 0:
                object_descriptions = [
                    an['object_description']
                    for an in dataset_dict['annotations']
                ]
            else:
                object_descriptions = []
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict['annotations']:
                if not self.use_instance_mask:
                    anno.pop('segmentation', None)
                if not self.use_keypoint:
                    anno.pop('keypoints', None)

            all_annos = [(utils.transform_instance_annotations(
                obj,
                transforms,
                image_shape,
                keypoint_hflip_indices=self.keypoint_hflip_indices,
            ), obj.get('iscrowd', 0))
                         for obj in dataset_dict.pop('annotations')]
            annos = [ann[0] for ann in all_annos if ann[1] == 0]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format)

            instances.gt_object_descriptions = ObjDescription(
                object_descriptions)

            del all_annos
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict['instances'] = utils.filter_empty_instances(instances)

        return dataset_dict


class ObjDescription:

    def __init__(self, object_descriptions):
        self.data = object_descriptions

    def __getitem__(self, item):
        assert type(item) == torch.Tensor
        assert item.dim() == 1
        if len(item) > 0:
            assert item.dtype == torch.int64 or item.dtype == torch.bool
            if item.dtype == torch.int64:
                return ObjDescription([self.data[x.item()] for x in item])
            elif item.dtype == torch.bool:
                return ObjDescription(list(compress(self.data, item)))

        return ObjDescription(list(compress(self.data, item)))

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return 'ObjDescription({})'.format(self.data)
