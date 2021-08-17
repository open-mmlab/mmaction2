# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys
from numpy.testing import assert_array_equal

from mmaction.datasets import ActivityNetDataset
from .base import BaseTestDataset


class TestActivitynetDataset(BaseTestDataset):

    def test_activitynet_dataset(self):
        activitynet_dataset = ActivityNetDataset(self.action_ann_file,
                                                 self.action_pipeline,
                                                 self.data_prefix)
        activitynet_infos = activitynet_dataset.video_infos
        assert activitynet_infos == [
            dict(
                video_name='v_test1',
                duration_second=1,
                duration_frame=30,
                annotations=[dict(segment=[0.3, 0.6], label='Rock climbing')],
                feature_frame=30,
                fps=30.0,
                rfps=30),
            dict(
                video_name='v_test2',
                duration_second=2,
                duration_frame=48,
                annotations=[dict(segment=[1.0, 2.0], label='Drinking beer')],
                feature_frame=48,
                fps=24.0,
                rfps=24.0)
        ]

    def test_activitynet_proposals2json(self):
        activitynet_dataset = ActivityNetDataset(self.action_ann_file,
                                                 self.action_pipeline,
                                                 self.data_prefix)
        results = [
            dict(
                video_name='v_test1',
                proposal_list=[dict(segment=[0.1, 0.9], score=0.1)]),
            dict(
                video_name='v_test2',
                proposal_list=[dict(segment=[10.1, 20.9], score=0.9)])
        ]
        result_dict = activitynet_dataset.proposals2json(results)
        assert result_dict == dict(
            test1=[{
                'segment': [0.1, 0.9],
                'score': 0.1
            }],
            test2=[{
                'segment': [10.1, 20.9],
                'score': 0.9
            }])
        result_dict = activitynet_dataset.proposals2json(results, True)
        assert result_dict == dict(
            test1=[{
                'segment': [0.1, 0.9],
                'score': 0.1
            }],
            test2=[{
                'segment': [10.1, 20.9],
                'score': 0.9
            }])

    def test_activitynet_evaluate(self):
        activitynet_dataset = ActivityNetDataset(self.action_ann_file,
                                                 self.action_pipeline,
                                                 self.data_prefix)

        with pytest.raises(TypeError):
            # results must be a list
            activitynet_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            activitynet_dataset.evaluate([0] * 5)

        with pytest.raises(KeyError):
            # unsupported metric
            activitynet_dataset.evaluate(
                [0] * len(activitynet_dataset), metrics='iou')

        # evaluate AR@AN metric
        results = [
            dict(
                video_name='v_test1',
                proposal_list=[dict(segment=[0.1, 0.9], score=0.1)]),
            dict(
                video_name='v_test2',
                proposal_list=[dict(segment=[10.1, 20.9], score=0.9)])
        ]
        eval_result = activitynet_dataset.evaluate(results, metrics=['AR@AN'])
        assert set(eval_result) == set(
            ['auc', 'AR@1', 'AR@5', 'AR@10', 'AR@100'])

    def test_activitynet_dump_results(self):
        activitynet_dataset = ActivityNetDataset(self.action_ann_file,
                                                 self.action_pipeline,
                                                 self.data_prefix)
        # test dumping json file
        results = [
            dict(
                video_name='v_test1',
                proposal_list=[dict(segment=[0.1, 0.9], score=0.1)]),
            dict(
                video_name='v_test2',
                proposal_list=[dict(segment=[10.1, 20.9], score=0.9)])
        ]
        dump_results = {
            'version': 'VERSION 1.3',
            'results': {
                'test1': [{
                    'segment': [0.1, 0.9],
                    'score': 0.1
                }],
                'test2': [{
                    'segment': [10.1, 20.9],
                    'score': 0.9
                }]
            },
            'external_data': {}
        }

        with tempfile.TemporaryDirectory() as tmpdir:

            tmp_filename = osp.join(tmpdir, 'result.json')
            activitynet_dataset.dump_results(results, tmp_filename, 'json')
            assert osp.isfile(tmp_filename)
            with open(tmp_filename, 'r+') as f:
                load_obj = mmcv.load(f, file_format='json')
            assert load_obj == dump_results

        # test dumping csv file
        results = [('test_video', np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9,
                                                              10]]))]
        with tempfile.TemporaryDirectory() as tmpdir:
            activitynet_dataset.dump_results(results, tmpdir, 'csv')
            load_obj = np.loadtxt(
                osp.join(tmpdir, 'test_video.csv'),
                dtype=np.float32,
                delimiter=',',
                skiprows=1)
            assert_array_equal(
                load_obj,
                np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                         dtype=np.float32))

    def test_action_pipeline(self):
        target_keys = ['video_name', 'data_prefix']

        # ActivityNet Dataset not in test mode
        action_dataset = ActivityNetDataset(
            self.action_ann_file,
            self.action_pipeline,
            self.data_prefix,
            test_mode=False)
        result = action_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # ActivityNet Dataset in test mode
        action_dataset = ActivityNetDataset(
            self.action_ann_file,
            self.action_pipeline,
            self.data_prefix,
            test_mode=True)
        result = action_dataset[0]
        assert assert_dict_has_keys(result, target_keys)
