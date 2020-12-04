import os.path as osp

import mmcv
import numpy as np


class ExampleSSNInstance:

    def __init__(self,
                 start_frame,
                 end_frame,
                 num_frames,
                 label=None,
                 best_iou=None,
                 overlap_self=None):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, num_frames)
        self.label = label if label is not None else -1
        self.coverage = (end_frame - start_frame) / num_frames
        self.best_iou = best_iou
        self.overlap_self = overlap_self


class TestLoading:

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    @classmethod
    def setup_class(cls):
        cls.img_path = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test.jpg')
        cls.video_path = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test.mp4')
        cls.wav_path = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test.wav')
        cls.audio_spec_path = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test.npy')
        cls.img_dir = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test_imgs')
        cls.raw_feature_dir = osp.join(
            osp.dirname(osp.dirname(__file__)),
            'data/test_activitynet_features')
        cls.bsp_feature_dir = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test_bsp_features')
        cls.proposals_dir = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test_proposals')
        cls.total_frames = 5
        cls.filename_tmpl = 'img_{:05}.jpg'
        cls.flow_filename_tmpl = '{}_{:05d}.jpg'
        video_total_frames = len(mmcv.VideoReader(cls.video_path))
        cls.audio_total_frames = video_total_frames
        cls.video_results = dict(
            filename=cls.video_path,
            label=1,
            total_frames=video_total_frames,
            start_index=0)
        cls.audio_results = dict(
            audios=np.random.randn(1280, ),
            audio_path=cls.wav_path,
            total_frames=cls.audio_total_frames,
            label=1,
            start_index=0)
        cls.audio_feature_results = dict(
            audios=np.random.randn(128, 80),
            audio_path=cls.audio_spec_path,
            total_frames=cls.audio_total_frames,
            label=1,
            start_index=0)
        cls.frame_results = dict(
            frame_dir=cls.img_dir,
            total_frames=cls.total_frames,
            filename_tmpl=cls.filename_tmpl,
            start_index=1,
            modality='RGB',
            offset=0,
            label=1)
        cls.flow_frame_results = dict(
            frame_dir=cls.img_dir,
            total_frames=cls.total_frames,
            filename_tmpl=cls.flow_filename_tmpl,
            modality='Flow',
            offset=0,
            label=1)
        cls.action_results = dict(
            video_name='v_test1',
            data_prefix=cls.raw_feature_dir,
            temporal_scale=5,
            boundary_ratio=0.1,
            duration_second=10,
            duration_frame=10,
            feature_frame=8,
            annotations=[{
                'segment': [3.0, 5.0],
                'label': 'Rock climbing'
            }])
        cls.proposal_results = dict(
            frame_dir=cls.img_dir,
            video_id='test_imgs',
            total_frames=cls.total_frames,
            filename_tmpl=cls.filename_tmpl,
            start_index=1,
            out_proposals=[[[
                'test_imgs',
                ExampleSSNInstance(1, 4, 10, 1, 1, 1)
            ], 0], [['test_imgs',
                     ExampleSSNInstance(2, 5, 10, 2, 1, 1)], 0]])

        cls.ava_results = dict(
            fps=30, timestamp=902, timestamp_start=840, shot_info=(0, 27000))

        cls.hvu_label_example1 = dict(
            categories=['action', 'object', 'scene', 'concept'],
            category_nums=[2, 5, 3, 2],
            label=dict(action=[0], object=[2, 3], scene=[0, 1]))
        cls.hvu_label_example2 = dict(
            categories=['action', 'object', 'scene', 'concept'],
            category_nums=[2, 5, 3, 2],
            label=dict(action=[1], scene=[1, 2], concept=[1]))
