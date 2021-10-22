# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys

from mmaction.datasets import SSNDataset
from .base import BaseTestDataset


class TestSSNDataset(BaseTestDataset):

    def test_proposal_pipeline(self):
        target_keys = [
            'frame_dir', 'video_id', 'total_frames', 'gts', 'proposals',
            'filename_tmpl', 'modality', 'out_proposals', 'reg_targets',
            'proposal_scale_factor', 'proposal_labels', 'proposal_type',
            'start_index'
        ]

        # SSN Dataset not in test mode
        proposal_dataset = SSNDataset(
            self.proposal_ann_file,
            self.proposal_pipeline,
            self.proposal_train_cfg,
            self.proposal_test_cfg,
            data_prefix=self.data_prefix)
        result = proposal_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # SSN Dataset with random sampling proposals
        proposal_dataset = SSNDataset(
            self.proposal_ann_file,
            self.proposal_pipeline,
            self.proposal_train_cfg,
            self.proposal_test_cfg,
            data_prefix=self.data_prefix,
            video_centric=False)
        result = proposal_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        target_keys = [
            'frame_dir', 'video_id', 'total_frames', 'gts', 'proposals',
            'filename_tmpl', 'modality', 'relative_proposal_list',
            'scale_factor_list', 'proposal_tick_list', 'reg_norm_consts',
            'start_index'
        ]

        # SSN Dataset in test mode
        proposal_dataset = SSNDataset(
            self.proposal_ann_file,
            self.proposal_test_pipeline,
            self.proposal_train_cfg,
            self.proposal_test_cfg,
            data_prefix=self.data_prefix,
            test_mode=True)
        result = proposal_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

    def test_ssn_dataset(self):
        # test ssn dataset
        ssn_dataset = SSNDataset(
            self.proposal_ann_file,
            self.proposal_pipeline,
            self.proposal_train_cfg,
            self.proposal_test_cfg,
            data_prefix=self.data_prefix)
        ssn_infos = ssn_dataset.video_infos
        assert ssn_infos[0]['video_id'] == 'imgs'
        assert ssn_infos[0]['total_frames'] == 5

        # test ssn dataset with verbose
        ssn_dataset = SSNDataset(
            self.proposal_ann_file,
            self.proposal_pipeline,
            self.proposal_train_cfg,
            self.proposal_test_cfg,
            data_prefix=self.data_prefix,
            verbose=True)
        ssn_infos = ssn_dataset.video_infos
        assert ssn_infos[0]['video_id'] == 'imgs'
        assert ssn_infos[0]['total_frames'] == 5

        # test ssn dataset with normalized proposal file
        with pytest.raises(Exception):
            ssn_dataset = SSNDataset(
                self.proposal_norm_ann_file,
                self.proposal_pipeline,
                self.proposal_train_cfg,
                self.proposal_test_cfg,
                data_prefix=self.data_prefix)
            ssn_infos = ssn_dataset.video_infos

        # test ssn dataset with reg_normalize_constants
        ssn_dataset = SSNDataset(
            self.proposal_ann_file,
            self.proposal_pipeline,
            self.proposal_train_cfg,
            self.proposal_test_cfg,
            data_prefix=self.data_prefix,
            reg_normalize_constants=[[[-0.0603, 0.0325], [0.0752, 0.1596]]])
        ssn_infos = ssn_dataset.video_infos
        assert ssn_infos[0]['video_id'] == 'imgs'
        assert ssn_infos[0]['total_frames'] == 5

        # test error case
        with pytest.raises(TypeError):
            ssn_dataset = SSNDataset(
                self.proposal_ann_file,
                self.proposal_pipeline,
                self.proposal_train_cfg,
                self.proposal_test_cfg,
                data_prefix=self.data_prefix,
                aug_ratio=('error', 'error'))
            ssn_infos = ssn_dataset.video_infos

    def test_ssn_evaluate(self):
        ssn_dataset = SSNDataset(
            self.proposal_ann_file,
            self.proposal_pipeline,
            self.proposal_train_cfg,
            self.proposal_test_cfg,
            data_prefix=self.data_prefix)
        ssn_dataset_topall = SSNDataset(
            self.proposal_ann_file,
            self.proposal_pipeline,
            self.proposal_train_cfg,
            self.proposal_test_cfg_topall,
            data_prefix=self.data_prefix)

        with pytest.raises(TypeError):
            # results must be a list
            ssn_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            ssn_dataset.evaluate([0] * 5)

        with pytest.raises(KeyError):
            # unsupported metric
            ssn_dataset.evaluate([0] * len(ssn_dataset), metrics='iou')

        # evaluate mAP metric
        results_relative_proposal_list = np.random.randn(16, 2)
        results_activity_scores = np.random.randn(16, 21)
        results_completeness_scores = np.random.randn(16, 20)
        results_bbox_preds = np.random.randn(16, 20, 2)
        results = [
            dict(
                relative_proposal_list=results_relative_proposal_list,
                activity_scores=results_activity_scores,
                completeness_scores=results_completeness_scores,
                bbox_preds=results_bbox_preds)
        ]
        eval_result = ssn_dataset.evaluate(results, metrics=['mAP'])
        assert set(eval_result) == set([
            'mAP@0.10', 'mAP@0.20', 'mAP@0.30', 'mAP@0.40', 'mAP@0.50',
            'mAP@0.50', 'mAP@0.60', 'mAP@0.70', 'mAP@0.80', 'mAP@0.90'
        ])

        # evaluate mAP metric without filtering topk
        results_relative_proposal_list = np.random.randn(16, 2)
        results_activity_scores = np.random.randn(16, 21)
        results_completeness_scores = np.random.randn(16, 20)
        results_bbox_preds = np.random.randn(16, 20, 2)
        results = [
            dict(
                relative_proposal_list=results_relative_proposal_list,
                activity_scores=results_activity_scores,
                completeness_scores=results_completeness_scores,
                bbox_preds=results_bbox_preds)
        ]
        eval_result = ssn_dataset_topall.evaluate(results, metrics=['mAP'])
        assert set(eval_result) == set([
            'mAP@0.10', 'mAP@0.20', 'mAP@0.30', 'mAP@0.40', 'mAP@0.50',
            'mAP@0.50', 'mAP@0.60', 'mAP@0.70', 'mAP@0.80', 'mAP@0.90'
        ])
