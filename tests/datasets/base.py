# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmengine import ConfigDict


class BaseTestDataset:

    @classmethod
    def setup_class(cls):
        # prefix path
        cls.data_prefix = osp.normpath(
            osp.join(osp.dirname(__file__), '../data/'))
        cls.ann_file_prefix = osp.join(cls.data_prefix, 'annotations')

        # annotations path
        cls.action_ann_file = osp.join(cls.ann_file_prefix,
                                       'action_test_anno.json')
        cls.audio_feature_ann_file = osp.join(cls.ann_file_prefix,
                                              'audio_feature_test_list.txt')
        cls.audio_ann_file = osp.join(cls.ann_file_prefix,
                                      'audio_test_list.txt')
        cls.frame_ann_file_multi_label = osp.join(
            cls.ann_file_prefix, 'rawframe_test_list_multi_label.txt')
        cls.frame_ann_file_with_offset = osp.join(
            cls.ann_file_prefix, 'rawframe_test_list_with_offset.txt')
        cls.frame_ann_file = osp.join(cls.ann_file_prefix,
                                      'rawframe_test_list.txt')
        cls.hvu_frame_ann_file = osp.join(cls.ann_file_prefix,
                                          'hvu_frame_test_anno.json')
        cls.hvu_video_ann_file = osp.join(cls.ann_file_prefix,
                                          'hvu_video_test_anno.json')
        cls.hvu_video_eval_ann_file = osp.join(
            cls.ann_file_prefix, 'hvu_video_eval_test_anno.json')
        cls.proposal_ann_file = osp.join(cls.ann_file_prefix,
                                         'proposal_test_list.txt')
        cls.proposal_norm_ann_file = osp.join(cls.ann_file_prefix,
                                              'proposal_normalized_list.txt')
        cls.rawvideo_test_anno_json = osp.join(cls.ann_file_prefix,
                                               'rawvideo_test_anno.json')
        cls.rawvideo_test_anno_txt = osp.join(cls.ann_file_prefix,
                                              'rawvideo_test_anno.txt')
        cls.video_ann_file = osp.join(cls.ann_file_prefix,
                                      'video_test_list.txt')
        cls.video_ann_file_multi_label = osp.join(
            cls.ann_file_prefix, 'video_test_list_multi_label.txt')
        cls.video_text_ann_file = osp.join(cls.ann_file_prefix,
                                           'video_text_test_list.json')
        cls.pose_ann_file = osp.join(cls.ann_file_prefix, 'sample.pkl')

        # pipeline configuration
        cls.action_pipeline = []
        cls.audio_feature_pipeline = [
            dict(type='LoadAudioFeature'),
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='AudioFeatureSelector')
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
        cls.frame_pipeline = [
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='RawFrameDecode', io_backend='disk')
        ]
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
        cls.rawvideo_pipeline = []
        cls.video_pipeline = [
            dict(type='OpenCVInit'),
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='OpenCVDecode')
        ]

        cls.video_text_pipeline = [
            dict(type='OpenCVInit'),
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='OpenCVDecode'),
            dict(type='CLIPTokenize')
        ]

        cls.hvu_categories = [
            'action', 'attribute', 'concept', 'event', 'object', 'scene'
        ]
        cls.hvu_category_nums = [739, 117, 291, 69, 1679, 248]
        cls.hvu_categories_for_eval = ['action', 'scene', 'object']
        cls.hvu_category_nums_for_eval = [3, 3, 3]

        cls.filename_tmpl = 'img_{:05d}.jpg'
