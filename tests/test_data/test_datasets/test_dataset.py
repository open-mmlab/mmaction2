import os.path as osp

import numpy as np
import pytest
from mmcv import ConfigDict

from mmaction.datasets import VideoDataset


class TestDataset:

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(osp.dirname(__file__)), 'data')
        cls.frame_ann_file = osp.join(cls.data_prefix, 'frame_test_list.txt')
        cls.frame_ann_file_with_offset = osp.join(
            cls.data_prefix, 'frame_test_list_with_offset.txt')
        cls.frame_ann_file_multi_label = osp.join(
            cls.data_prefix, 'frame_test_list_multi_label.txt')
        cls.video_ann_file = osp.join(cls.data_prefix, 'video_test_list.txt')
        cls.hvu_video_ann_file = osp.join(cls.data_prefix,
                                          'hvu_video_test_anno.json')
        cls.hvu_video_eval_ann_file = osp.join(
            cls.data_prefix, 'hvu_video_eval_test_anno.json')
        cls.hvu_frame_ann_file = osp.join(cls.data_prefix,
                                          'hvu_frame_test_anno.json')
        cls.action_ann_file = osp.join(cls.data_prefix,
                                       'action_test_anno.json')
        cls.proposal_ann_file = osp.join(cls.data_prefix,
                                         'proposal_test_list.txt')
        cls.proposal_norm_ann_file = osp.join(cls.data_prefix,
                                              'proposal_normalized_list.txt')
        cls.audio_ann_file = osp.join(cls.data_prefix, 'audio_test_list.txt')
        cls.audio_feature_ann_file = osp.join(cls.data_prefix,
                                              'audio_feature_test_list.txt')
        cls.rawvideo_test_anno_txt = osp.join(cls.data_prefix,
                                              'rawvideo_test_anno.txt')
        cls.rawvideo_test_anno_json = osp.join(cls.data_prefix,
                                               'rawvideo_test_anno.json')
        cls.rawvideo_pipeline = []

        cls.frame_pipeline = [
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='RawFrameDecode', io_backend='disk')
        ]
        cls.audio_pipeline = [
            dict(type='AudioDecodeInit'),
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='AudioDecode')
        ]
        cls.audio_feature_pipeline = [
            dict(type='LoadAudioFeature'),
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='AudioFeatureSelector')
        ]
        cls.video_pipeline = [
            dict(type='OpenCVInit'),
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='OpenCVDecode')
        ]
        cls.action_pipeline = []
        cls.proposal_pipeline = [
            dict(
                type='SampleProposalFrames',
                clip_len=1,
                body_segments=5,
                aug_segments=(2, 2),
                aug_ratio=0.5),
            dict(type='RawFrameDecode', io_backend='disk')
        ]
        cls.proposal_test_pipeline = [
            dict(
                type='SampleProposalFrames',
                clip_len=1,
                body_segments=5,
                aug_segments=(2, 2),
                aug_ratio=0.5,
                mode='test'),
            dict(type='RawFrameDecode', io_backend='disk')
        ]

        cls.proposal_train_cfg = ConfigDict(
            dict(
                ssn=dict(
                    assigner=dict(
                        positive_iou_threshold=0.7,
                        background_iou_threshold=0.01,
                        incomplete_iou_threshold=0.5,
                        background_coverage_threshold=0.02,
                        incomplete_overlap_threshold=0.01),
                    sampler=dict(
                        num_per_video=8,
                        positive_ratio=1,
                        background_ratio=1,
                        incomplete_ratio=6,
                        add_gt_as_proposals=True),
                    loss_weight=dict(
                        comp_loss_weight=0.1, reg_loss_weight=0.1),
                    debug=False)))
        cls.proposal_test_cfg = ConfigDict(
            dict(
                ssn=dict(
                    sampler=dict(test_interval=6, batch_size=16),
                    evaluater=dict(
                        top_k=2000,
                        nms=0.2,
                        softmax_before_filter=True,
                        cls_top_k=2))))
        cls.proposal_test_cfg_topall = ConfigDict(
            dict(
                ssn=dict(
                    sampler=dict(test_interval=6, batch_size=16),
                    evaluater=dict(
                        top_k=-1,
                        nms=0.2,
                        softmax_before_filter=True,
                        cls_top_k=2))))

        cls.hvu_categories = [
            'action', 'attribute', 'concept', 'event', 'object', 'scene'
        ]

        cls.hvu_category_nums = [739, 117, 291, 69, 1679, 248]

        cls.hvu_categories_for_eval = ['action', 'scene', 'object']
        cls.hvu_category_nums_for_eval = [3, 3, 3]

        cls.filename_tmpl = 'img_{:05d}.jpg'

    def test_video_dataset(self):
        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            data_prefix=self.data_prefix)
        video_infos = video_dataset.video_infos
        video_filename = osp.join(self.data_prefix, 'test.mp4')
        assert video_infos == [dict(filename=video_filename, label=0)] * 2
        assert video_dataset.start_index == 0

    def test_video_pipeline(self):
        target_keys = ['filename', 'label', 'start_index', 'modality']

        # VideoDataset not in test mode
        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            data_prefix=self.data_prefix,
            test_mode=False)
        result = video_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # VideoDataset in test mode
        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            data_prefix=self.data_prefix,
            test_mode=True)
        result = video_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

    def test_video_evaluate(self):
        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            data_prefix=self.data_prefix)

        with pytest.raises(TypeError):
            # results must be a list
            video_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            video_dataset.evaluate([0] * 5)

        with pytest.raises(TypeError):
            # topk must be int or tuple of int
            video_dataset.evaluate(
                [0] * len(video_dataset),
                metric_options=dict(top_k_accuracy=dict(topk=1.)))

        with pytest.raises(KeyError):
            # unsupported metric
            video_dataset.evaluate([0] * len(video_dataset), metrics='iou')

        # evaluate top_k_accuracy and mean_class_accuracy metric
        results = [np.array([0.1, 0.5, 0.4])] * 2
        eval_result = video_dataset.evaluate(
            results, metrics=['top_k_accuracy', 'mean_class_accuracy'])
        assert set(eval_result.keys()) == set(
            ['top1_acc', 'top5_acc', 'mean_class_accuracy'])

    def test_base_dataset(self):
        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            data_prefix=self.data_prefix,
            start_index=3)
        assert len(video_dataset) == 2
        assert video_dataset.start_index == 3
