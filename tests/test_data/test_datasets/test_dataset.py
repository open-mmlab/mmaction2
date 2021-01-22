import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest
from mmcv import ConfigDict
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmaction.datasets import (ActivityNetDataset, AudioDataset,
                               AudioFeatureDataset, AudioVisualDataset,
                               HVUDataset, RawframeDataset, RawVideoDataset,
                               RepeatDataset, SSNDataset, VideoDataset)


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

    def test_rawvideo_dataset(self):
        # Try to load txt file
        rawvideo_dataset = RawVideoDataset(
            ann_file=self.rawvideo_test_anno_txt,
            pipeline=self.rawvideo_pipeline,
            clipname_tmpl='part_{}.mp4',
            sampling_strategy='positive',
            data_prefix=self.data_prefix)
        result = rawvideo_dataset[0]
        clipname = osp.join(self.data_prefix, 'test_rawvideo_dataset',
                            'part_0.mp4')
        assert result['filename'] == clipname

        # Try to load json file
        rawvideo_dataset = RawVideoDataset(
            ann_file=self.rawvideo_test_anno_json,
            pipeline=self.rawvideo_pipeline,
            clipname_tmpl='part_{}.mp4',
            sampling_strategy='random',
            data_prefix=self.data_prefix,
            test_mode=True)
        result = rawvideo_dataset[0]

    def test_hvu_dataset(self):
        hvu_frame_dataset = HVUDataset(
            ann_file=self.hvu_frame_ann_file,
            pipeline=self.frame_pipeline,
            tag_categories=self.hvu_categories,
            tag_category_nums=self.hvu_category_nums,
            filename_tmpl=self.filename_tmpl,
            data_prefix=self.data_prefix,
            start_index=1)
        hvu_frame_infos = hvu_frame_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'test_imgs')
        assert hvu_frame_infos == [
            dict(
                frame_dir=frame_dir,
                total_frames=5,
                label=dict(
                    concept=[250, 131, 42, 51, 57, 155, 122],
                    object=[1570, 508],
                    event=[16],
                    action=[180],
                    scene=[206]),
                categories=self.hvu_categories,
                category_nums=self.hvu_category_nums,
                filename_tmpl=self.filename_tmpl,
                start_index=1,
                modality='RGB')
        ] * 2

        hvu_video_dataset = HVUDataset(
            ann_file=self.hvu_video_ann_file,
            pipeline=self.video_pipeline,
            tag_categories=self.hvu_categories,
            tag_category_nums=self.hvu_category_nums,
            data_prefix=self.data_prefix)
        hvu_video_infos = hvu_video_dataset.video_infos
        filename = osp.join(self.data_prefix, 'tmp.mp4')
        assert hvu_video_infos == [
            dict(
                filename=filename,
                label=dict(
                    concept=[250, 131, 42, 51, 57, 155, 122],
                    object=[1570, 508],
                    event=[16],
                    action=[180],
                    scene=[206]),
                categories=self.hvu_categories,
                category_nums=self.hvu_category_nums)
        ] * 2

        hvu_video_eval_dataset = HVUDataset(
            ann_file=self.hvu_video_eval_ann_file,
            pipeline=self.video_pipeline,
            tag_categories=self.hvu_categories_for_eval,
            tag_category_nums=self.hvu_category_nums_for_eval,
            data_prefix=self.data_prefix)

        results = [
            np.array([
                -1.59812844, 0.24459082, 1.38486497, 0.28801252, 1.09813449,
                -0.28696971, 0.0637848, 0.22877678, -1.82406999
            ]),
            np.array([
                0.87904563, 1.64264224, 0.46382051, 0.72865088, -2.13712525,
                1.28571358, 1.01320328, 0.59292737, -0.05502892
            ])
        ]
        mAP = hvu_video_eval_dataset.evaluate(results)
        assert_array_almost_equal(mAP['action_mAP'], 1.0)
        assert_array_almost_equal(mAP['scene_mAP'], 0.5)
        assert_array_almost_equal(mAP['object_mAP'], 0.75)

    def test_rawframe_dataset(self):
        rawframe_dataset = RawframeDataset(self.frame_ann_file,
                                           self.frame_pipeline,
                                           self.data_prefix)
        rawframe_infos = rawframe_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'test_imgs')
        assert rawframe_infos == [
            dict(frame_dir=frame_dir, total_frames=5, label=127)
        ] * 2
        assert rawframe_dataset.start_index == 1

    def test_audio_dataset(self):
        audio_dataset = AudioDataset(
            self.audio_ann_file,
            self.audio_pipeline,
            data_prefix=self.data_prefix)
        audio_infos = audio_dataset.video_infos
        wav_path = osp.join(self.data_prefix, 'test.wav')
        assert audio_infos == [
            dict(audio_path=wav_path, total_frames=100, label=127)
        ] * 2

    def test_audio_feature_dataset(self):
        audio_dataset = AudioFeatureDataset(
            self.audio_feature_ann_file,
            self.audio_feature_pipeline,
            data_prefix=self.data_prefix)
        audio_infos = audio_dataset.video_infos
        feature_path = osp.join(self.data_prefix, 'test.npy')
        assert audio_infos == [
            dict(audio_path=feature_path, total_frames=100, label=127)
        ] * 2

    def test_rawframe_dataset_with_offset(self):
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_with_offset,
            self.frame_pipeline,
            self.data_prefix,
            with_offset=True)
        rawframe_infos = rawframe_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'test_imgs')
        assert rawframe_infos == [
            dict(frame_dir=frame_dir, offset=2, total_frames=5, label=127)
        ] * 2
        assert rawframe_dataset.start_index == 1

    def test_rawframe_dataset_multi_label(self):
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_multi_label,
            self.frame_pipeline,
            self.data_prefix,
            multi_class=True,
            num_classes=100)
        rawframe_infos = rawframe_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'test_imgs')
        label0 = [1]
        label1 = [3, 5]
        labels = [label0, label1]
        for info, label in zip(rawframe_infos, labels):
            assert info['frame_dir'] == frame_dir
            assert info['total_frames'] == 5
            assert set(info['label']) == set(label)
        assert rawframe_dataset.start_index == 1

    def test_audio_visual_dataset(self):
        test_dataset = AudioVisualDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            video_prefix=self.data_prefix,
            data_prefix=self.data_prefix)
        video_infos = test_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'test_imgs')
        audio_path = osp.join(self.data_prefix, 'test_imgs.npy')
        filename = osp.join(self.data_prefix, 'test_imgs.mp4')
        assert video_infos == [
            dict(
                frame_dir=frame_dir,
                audio_path=audio_path,
                filename=filename,
                total_frames=5,
                label=127)
        ] * 2
        assert test_dataset.start_index == 1

    def test_dataset_realpath(self):
        dataset = RawframeDataset(self.frame_ann_file, self.frame_pipeline,
                                  '.')
        assert dataset.data_prefix == osp.realpath('.')
        dataset = RawframeDataset(self.frame_ann_file, self.frame_pipeline,
                                  's3://good')
        assert dataset.data_prefix == 's3://good'

        dataset = RawframeDataset(self.frame_ann_file, self.frame_pipeline)
        assert dataset.data_prefix is None
        assert dataset.video_infos[0]['frame_dir'] == 'test_imgs'

    def test_video_dataset(self):
        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            data_prefix=self.data_prefix)
        video_infos = video_dataset.video_infos
        video_filename = osp.join(self.data_prefix, 'test.mp4')
        assert video_infos == [dict(filename=video_filename, label=0)] * 2
        assert video_dataset.start_index == 0

    def test_rawframe_pipeline(self):
        target_keys = [
            'frame_dir', 'total_frames', 'label', 'filename_tmpl',
            'start_index', 'modality'
        ]

        # RawframeDataset not in test mode
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            test_mode=False)
        result = rawframe_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # RawframeDataset in multi-class tasks
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            multi_class=True,
            num_classes=400,
            test_mode=False)
        result = rawframe_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # RawframeDataset with offset
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_with_offset,
            self.frame_pipeline,
            self.data_prefix,
            with_offset=True,
            num_classes=400,
            test_mode=False)
        result = rawframe_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys + ['offset'])

        # RawframeDataset in test mode
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            test_mode=True)
        result = rawframe_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # RawframeDataset in multi-class tasks in test mode
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            multi_class=True,
            num_classes=400,
            test_mode=True)
        result = rawframe_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # RawframeDataset with offset
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_with_offset,
            self.frame_pipeline,
            self.data_prefix,
            with_offset=True,
            num_classes=400,
            test_mode=True)
        result = rawframe_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys + ['offset'])

    def test_audio_pipeline(self):
        target_keys = [
            'audio_path', 'label', 'start_index', 'modality', 'audios_shape',
            'length', 'sample_rate', 'total_frames'
        ]

        # Audio dataset not in test mode
        audio_dataset = AudioDataset(
            self.audio_ann_file,
            self.audio_pipeline,
            data_prefix=self.data_prefix,
            test_mode=False)
        result = audio_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # Audio dataset in test mode
        audio_dataset = AudioDataset(
            self.audio_ann_file,
            self.audio_pipeline,
            data_prefix=self.data_prefix,
            test_mode=True)
        result = audio_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

    def test_audio_feature_pipeline(self):
        target_keys = [
            'audio_path', 'label', 'start_index', 'modality', 'audios',
            'total_frames'
        ]

        # Audio feature dataset not in test mode
        audio_feature_dataset = AudioFeatureDataset(
            self.audio_feature_ann_file,
            self.audio_feature_pipeline,
            data_prefix=self.data_prefix,
            test_mode=False)
        result = audio_feature_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # Audio dataset in test mode
        audio_feature_dataset = AudioFeatureDataset(
            self.audio_feature_ann_file,
            self.audio_feature_pipeline,
            data_prefix=self.data_prefix,
            test_mode=True)
        result = audio_feature_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

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

    def test_action_pipeline(self):
        target_keys = ['video_name', 'data_prefix']

        # ActivityNet Dataset not in test mode
        action_dataset = ActivityNetDataset(
            self.action_ann_file,
            self.action_pipeline,
            self.data_prefix,
            test_mode=False)
        result = action_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # ActivityNet Dataset in test mode
        action_dataset = ActivityNetDataset(
            self.action_ann_file,
            self.action_pipeline,
            self.data_prefix,
            test_mode=True)
        result = action_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

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
        assert self.check_keys_contain(result.keys(), target_keys)

        # SSN Dataset with random sampling proposals
        proposal_dataset = SSNDataset(
            self.proposal_ann_file,
            self.proposal_pipeline,
            self.proposal_train_cfg,
            self.proposal_test_cfg,
            data_prefix=self.data_prefix,
            video_centric=False)
        result = proposal_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

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
        assert self.check_keys_contain(result.keys(), target_keys)

    def test_rawframe_evaluate(self):
        rawframe_dataset = RawframeDataset(self.frame_ann_file,
                                           self.frame_pipeline,
                                           self.data_prefix)

        with pytest.raises(TypeError):
            # results must be a list
            rawframe_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            rawframe_dataset.evaluate([0] * 5)

        with pytest.raises(TypeError):
            # topk must be int or tuple of int
            rawframe_dataset.evaluate(
                [0] * len(rawframe_dataset),
                metric_options=dict(top_k_accuracy=dict(topk=1.)))

        with pytest.raises(KeyError):
            # unsupported metric
            rawframe_dataset.evaluate(
                [0] * len(rawframe_dataset), metrics='iou')

        # evaluate top_k_accuracy and mean_class_accuracy metric
        results = [np.array([0.1, 0.5, 0.4])] * 2
        eval_result = rawframe_dataset.evaluate(
            results, metrics=['top_k_accuracy', 'mean_class_accuracy'])
        assert set(eval_result.keys()) == set(
            ['top1_acc', 'top5_acc', 'mean_class_accuracy'])

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

    def test_repeat_dataset(self):
        rawframe_dataset = RawframeDataset(self.frame_ann_file,
                                           self.frame_pipeline,
                                           self.data_prefix)
        repeat_dataset = RepeatDataset(rawframe_dataset, 5)
        assert len(repeat_dataset) == 10
        result_a = repeat_dataset[0]
        result_b = repeat_dataset[2]
        assert set(result_a.keys()) == set(result_b.keys())
        for key in result_a:
            if isinstance(result_a[key], np.ndarray):
                assert np.equal(result_a[key], result_b[key]).all()
            elif isinstance(result_a[key], list):
                assert all(
                    np.array_equal(a, b)
                    for (a, b) in zip(result_a[key], result_b[key]))
            else:
                assert result_a[key] == result_b[key]

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

        tmp_filename = osp.join(tempfile.gettempdir(), 'result.json')
        activitynet_dataset.dump_results(results, tmp_filename, 'json')
        assert osp.isfile(tmp_filename)
        with open(tmp_filename, 'r+') as f:
            load_obj = mmcv.load(f, file_format='json')
        assert load_obj == dump_results
        os.remove(tmp_filename)

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

    def test_ssn_dataset(self):
        # test ssn dataset
        ssn_dataset = SSNDataset(
            self.proposal_ann_file,
            self.proposal_pipeline,
            self.proposal_train_cfg,
            self.proposal_test_cfg,
            data_prefix=self.data_prefix)
        ssn_infos = ssn_dataset.video_infos
        assert ssn_infos[0]['video_id'] == 'test_imgs'
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
        assert ssn_infos[0]['video_id'] == 'test_imgs'
        assert ssn_infos[0]['total_frames'] == 5

        # test ssn datatset with normalized proposal file
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
        assert ssn_infos[0]['video_id'] == 'test_imgs'
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

    def test_audio_evaluate(self):
        audio_dataset = AudioDataset(
            self.audio_ann_file,
            self.audio_pipeline,
            data_prefix=self.data_prefix)

        with pytest.raises(TypeError):
            # results must be a list
            audio_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            audio_dataset.evaluate([0] * 5)

        with pytest.raises(TypeError):
            # topk must be int or tuple of int
            audio_dataset.evaluate(
                [0] * len(audio_dataset),
                metric_options=dict(top_k_accuracy=dict(topk=1.)))

        with pytest.raises(KeyError):
            # unsupported metric
            audio_dataset.evaluate([0] * len(audio_dataset), metrics='iou')

        # evaluate top_k_accuracy and mean_class_accuracy metric
        results = [np.array([0.1, 0.5, 0.4])] * 2
        eval_result = audio_dataset.evaluate(
            results, metrics=['top_k_accuracy', 'mean_class_accuracy'])
        assert set(eval_result.keys()) == set(
            ['top1_acc', 'top5_acc', 'mean_class_accuracy'])

    def test_audio_feature_evaluate(self):
        audio_dataset = AudioFeatureDataset(
            self.audio_feature_ann_file,
            self.audio_feature_pipeline,
            data_prefix=self.data_prefix)

        with pytest.raises(TypeError):
            # results must be a list
            audio_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            audio_dataset.evaluate([0] * 5)

        with pytest.raises(TypeError):
            # topk must be int or tuple of int
            audio_dataset.evaluate(
                [0] * len(audio_dataset),
                metric_options=dict(top_k_accuracy=dict(topk=1.)))

        with pytest.raises(KeyError):
            # unsupported metric
            audio_dataset.evaluate([0] * len(audio_dataset), metrics='iou')

        # evaluate top_k_accuracy and mean_class_accuracy metric
        results = [np.array([0.1, 0.5, 0.4])] * 2
        eval_result = audio_dataset.evaluate(
            results, metrics=['top_k_accuracy', 'mean_class_accuracy'])
        assert set(eval_result) == set(
            ['top1_acc', 'top5_acc', 'mean_class_accuracy'])
