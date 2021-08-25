# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import pytest
import torch
from mmcv.utils import assert_dict_has_keys
from numpy.testing import assert_array_almost_equal

from mmaction.datasets.pipelines import (LoadAudioFeature, LoadHVULabel,
                                         LoadLocalizationFeature,
                                         LoadProposals)
from .base import BaseTestLoading


class TestLoad(BaseTestLoading):

    def test_load_hvu_label(self):
        hvu_label_example1 = copy.deepcopy(self.hvu_label_example1)
        hvu_label_example2 = copy.deepcopy(self.hvu_label_example2)
        categories = hvu_label_example1['categories']
        category_nums = hvu_label_example1['category_nums']
        num_tags = sum(category_nums)
        num_categories = len(categories)

        loader = LoadHVULabel()
        assert repr(loader) == (f'{loader.__class__.__name__}('
                                f'hvu_initialized={False})')

        result1 = loader(hvu_label_example1)
        label1 = torch.zeros(num_tags)
        mask1 = torch.zeros(num_tags)
        category_mask1 = torch.zeros(num_categories)

        assert repr(loader) == (f'{loader.__class__.__name__}('
                                f'hvu_initialized={True})')

        label1[[0, 4, 5, 7, 8]] = 1.
        mask1[:10] = 1.
        category_mask1[:3] = 1.

        assert torch.all(torch.eq(label1, result1['label']))
        assert torch.all(torch.eq(mask1, result1['mask']))
        assert torch.all(torch.eq(category_mask1, result1['category_mask']))

        result2 = loader(hvu_label_example2)
        label2 = torch.zeros(num_tags)
        mask2 = torch.zeros(num_tags)
        category_mask2 = torch.zeros(num_categories)

        label2[[1, 8, 9, 11]] = 1.
        mask2[:2] = 1.
        mask2[7:] = 1.
        category_mask2[[0, 2, 3]] = 1.

        assert torch.all(torch.eq(label2, result2['label']))
        assert torch.all(torch.eq(mask2, result2['mask']))
        assert torch.all(torch.eq(category_mask2, result2['category_mask']))

    def test_load_localization_feature(self):
        target_keys = ['raw_feature']

        action_result = copy.deepcopy(self.action_results)

        # test error cases
        with pytest.raises(NotImplementedError):
            load_localization_feature = LoadLocalizationFeature(
                'unsupport_ext')

        # test normal cases
        load_localization_feature = LoadLocalizationFeature()
        load_localization_feature_result = load_localization_feature(
            action_result)
        assert assert_dict_has_keys(load_localization_feature_result,
                                    target_keys)
        assert load_localization_feature_result['raw_feature'].shape == (400,
                                                                         5)
        assert repr(load_localization_feature) == (
            f'{load_localization_feature.__class__.__name__}('
            f'raw_feature_ext=.csv)')

    def test_load_proposals(self):
        target_keys = [
            'bsp_feature', 'tmin', 'tmax', 'tmin_score', 'tmax_score',
            'reference_temporal_iou'
        ]

        action_result = copy.deepcopy(self.action_results)

        # test error cases
        with pytest.raises(NotImplementedError):
            load_proposals = LoadProposals(5, self.proposals_dir,
                                           self.bsp_feature_dir,
                                           'unsupport_ext')

        with pytest.raises(NotImplementedError):
            load_proposals = LoadProposals(5, self.proposals_dir,
                                           self.bsp_feature_dir, '.csv',
                                           'unsupport_ext')

        # test normal cases
        load_proposals = LoadProposals(5, self.proposals_dir,
                                       self.bsp_feature_dir)
        load_proposals_result = load_proposals(action_result)
        assert assert_dict_has_keys(load_proposals_result, target_keys)
        assert load_proposals_result['bsp_feature'].shape[0] == 5
        assert load_proposals_result['tmin'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['tmin'], np.arange(0.1, 0.6, 0.1), decimal=4)
        assert load_proposals_result['tmax'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['tmax'], np.arange(0.2, 0.7, 0.1), decimal=4)
        assert load_proposals_result['tmin_score'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['tmin_score'],
            np.arange(0.95, 0.90, -0.01),
            decimal=4)
        assert load_proposals_result['tmax_score'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['tmax_score'],
            np.arange(0.96, 0.91, -0.01),
            decimal=4)
        assert load_proposals_result['reference_temporal_iou'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['reference_temporal_iou'],
            np.arange(0.85, 0.80, -0.01),
            decimal=4)
        assert repr(load_proposals) == (
            f'{load_proposals.__class__.__name__}('
            f'top_k={5}, '
            f'pgm_proposals_dir={self.proposals_dir}, '
            f'pgm_features_dir={self.bsp_feature_dir}, '
            f'proposal_ext=.csv, '
            f'feature_ext=.npy)')

    def test_load_audio_feature(self):
        target_keys = ['audios']
        inputs = copy.deepcopy(self.audio_feature_results)
        load_audio_feature = LoadAudioFeature()
        results = load_audio_feature(inputs)
        assert assert_dict_has_keys(results, target_keys)

        # test when no audio feature file exists
        inputs = copy.deepcopy(self.audio_feature_results)
        inputs['audio_path'] = 'foo/foo/bar.npy'
        load_audio_feature = LoadAudioFeature()
        results = load_audio_feature(inputs)
        assert results['audios'].shape == (640, 80)
        assert assert_dict_has_keys(results, target_keys)
        assert repr(load_audio_feature) == (
            f'{load_audio_feature.__class__.__name__}('
            f'pad_method=zero)')
