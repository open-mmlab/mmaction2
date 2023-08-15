# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import platform

import mmcv
import numpy as np
import pytest
import torch
from mmengine.testing import assert_dict_has_keys
from numpy.testing import assert_array_almost_equal

from mmaction.datasets.transforms import (DecordDecode, DecordInit,
                                          GenerateLocalizationLabels,
                                          LoadAudioFeature, LoadHVULabel,
                                          LoadLocalizationFeature,
                                          LoadProposals, LoadRGBFromFile,
                                          OpenCVDecode, OpenCVInit, PIMSDecode,
                                          PIMSInit, PyAVDecode,
                                          PyAVDecodeMotionVector, PyAVInit)

from mmaction.datasets.transforms import RawFrameDecode  # isort:skip


class BaseTestLoading:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.normpath(
            osp.join(osp.dirname(__file__), '../../data'))
        cls.img_path = osp.join(cls.data_prefix, 'test.jpg')
        cls.video_path = osp.join(cls.data_prefix, 'test.mp4')
        cls.wav_path = osp.join(cls.data_prefix, 'test.wav')
        cls.audio_spec_path = osp.join(cls.data_prefix, 'test.npy')
        cls.img_dir = osp.join(cls.data_prefix, 'imgs')
        cls.raw_feature_dir = osp.join(cls.data_prefix, 'activitynet_features')
        cls.bsp_feature_dir = osp.join(cls.data_prefix, 'bsp_features')
        cls.proposals_dir = osp.join(cls.data_prefix, 'proposals')

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
        cls.action_results['feature_path'] = osp.join(cls.raw_feature_dir,
                                                      'v_test1.csv')

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


class TestDecode(BaseTestLoading):

    def test_pyav_init(self):
        target_keys = ['video_reader', 'total_frames']
        video_result = copy.deepcopy(self.video_results)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        assert assert_dict_has_keys(pyav_init_result, target_keys)
        assert pyav_init_result['total_frames'] == 300
        assert repr(
            pyav_init) == f'{pyav_init.__class__.__name__}(io_backend=disk)'

    def test_pyav_decode(self):
        target_keys = ['frame_inds', 'imgs', 'original_shape']

        # test PyAV with 2 dim input and start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames,
                                               2)[:, np.newaxis]
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode()
        pyav_decode_result = pyav_decode(video_result)
        assert assert_dict_has_keys(pyav_decode_result, target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)
        assert repr(pyav_decode) == (f'{pyav_decode.__class__.__name__}('
                                     f'multi_thread={False}, mode=accurate)')

        # test PyAV with 1 dim input and start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames, 5)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode()
        pyav_decode_result = pyav_decode(video_result)
        assert assert_dict_has_keys(pyav_decode_result, target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # PyAV with multi thread and start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames, 5)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode(multi_thread=True)
        pyav_decode_result = pyav_decode(video_result)
        assert assert_dict_has_keys(pyav_decode_result, target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)
        assert repr(pyav_decode) == (f'{pyav_decode.__class__.__name__}('
                                     f'multi_thread={True}, mode=accurate)')

        # test PyAV with 2 dim input
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames,
                                               2)[:, np.newaxis]
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode()
        pyav_decode_result = pyav_decode(video_result)
        assert assert_dict_has_keys(pyav_decode_result, target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test PyAV with 1 dim input
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 5)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode()
        pyav_decode_result = pyav_decode(video_result)
        assert assert_dict_has_keys(pyav_decode_result, target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # PyAV with multi thread
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 5)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode(multi_thread=True)
        pyav_decode_result = pyav_decode(video_result)
        assert assert_dict_has_keys(pyav_decode_result, target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # PyAV with efficient mode
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 5)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode(multi_thread=True, mode='efficient')
        pyav_decode_result = pyav_decode(video_result)
        assert assert_dict_has_keys(pyav_decode_result, target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)
        assert pyav_decode_result['video_reader'] is None

        assert (repr(pyav_decode) == pyav_decode.__class__.__name__ +
                f'(multi_thread={True}, mode=efficient)')

    def test_pims_init(self):
        target_keys = ['video_reader', 'total_frames']
        video_result = copy.deepcopy(self.video_results)
        pims_init = PIMSInit()
        pims_init_result = pims_init(video_result)
        assert assert_dict_has_keys(pims_init_result, target_keys)
        assert pims_init_result['total_frames'] == 300

        pims_init = PIMSInit(mode='efficient')
        pims_init_result = pims_init(video_result)
        assert assert_dict_has_keys(pims_init_result, target_keys)
        assert pims_init_result['total_frames'] == 300

        assert repr(pims_init) == (f'{pims_init.__class__.__name__}'
                                   f'(io_backend=disk, mode=efficient)')

    def test_pims_decode(self):
        target_keys = ['frame_inds', 'imgs', 'original_shape']

        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames,
                                               2)[:, np.newaxis]
        pims_init = PIMSInit()
        pims_init_result = pims_init(video_result)

        pims_decode = PIMSDecode()
        pims_decode_result = pims_decode(pims_init_result)
        assert assert_dict_has_keys(pims_decode_result, target_keys)
        assert pims_decode_result['original_shape'] == (256, 340)
        assert np.shape(pims_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

    def test_decord_init(self):
        target_keys = ['video_reader', 'total_frames', 'avg_fps']
        video_result = copy.deepcopy(self.video_results)
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        assert assert_dict_has_keys(decord_init_result, target_keys)
        assert decord_init_result['total_frames'] == len(
            decord_init_result['video_reader'])
        assert decord_init_result['avg_fps'] == 30

        assert repr(decord_init) == (f'{decord_init.__class__.__name__}('
                                     f'io_backend=disk, '
                                     f'num_threads=1)')

    def test_decord_decode(self):
        target_keys = ['frame_inds', 'imgs', 'original_shape']

        # test Decord with 2 dim input using accurate mode
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames,
                                               3)[:, np.newaxis]
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        video_result['video_reader'] = decord_init_result['video_reader']

        decord_decode = DecordDecode()
        decord_decode_result = decord_decode(video_result)
        assert assert_dict_has_keys(decord_decode_result, target_keys)
        assert decord_decode_result['original_shape'] == (256, 340)
        assert np.shape(decord_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test Decord with 1 dim input using accurate mode
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames, 3)
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        video_result['video_reader'] = decord_init_result['video_reader']

        decord_decode = DecordDecode()
        decord_decode_result = decord_decode(video_result)
        assert assert_dict_has_keys(decord_decode_result, target_keys)
        assert decord_decode_result['original_shape'] == (256, 340)
        assert np.shape(decord_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test Decord with 2 dim input using efficient mode
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames,
                                               3)[:, np.newaxis]
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        video_result['video_reader'] = decord_init_result['video_reader']

        decord_decode = DecordDecode(mode='efficient')
        decord_decode_result = decord_decode(video_result)
        assert assert_dict_has_keys(decord_decode_result, target_keys)
        assert decord_decode_result['original_shape'] == (256, 340)
        assert np.shape(decord_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test Decord with 1 dim input using efficient mode
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 3)
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        video_result['video_reader'] = decord_init_result['video_reader']

        decord_decode = DecordDecode(mode='efficient')
        decord_decode_result = decord_decode(video_result)
        assert assert_dict_has_keys(decord_decode_result, target_keys)
        assert decord_decode_result['original_shape'] == (256, 340)
        assert np.shape(decord_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)
        assert repr(decord_decode) == (f'{decord_decode.__class__.__name__}('
                                       f'mode=efficient)')

    def test_opencv_init(self):
        target_keys = ['new_path', 'video_reader', 'total_frames']
        video_result = copy.deepcopy(self.video_results)
        opencv_init = OpenCVInit()
        opencv_init_result = opencv_init(video_result)
        assert assert_dict_has_keys(opencv_init_result, target_keys)
        assert opencv_init_result['total_frames'] == len(
            opencv_init_result['video_reader'])
        assert repr(opencv_init) == (f'{opencv_init.__class__.__name__}('
                                     f'io_backend=disk)')

    def test_opencv_decode(self):
        target_keys = ['frame_inds', 'imgs', 'original_shape']

        # test OpenCV with 2 dim input when start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames,
                                               2)[:, np.newaxis]
        opencv_init = OpenCVInit()
        opencv_init_result = opencv_init(video_result)
        video_result['video_reader'] = opencv_init_result['video_reader']

        opencv_decode = OpenCVDecode()
        opencv_decode_result = opencv_decode(video_result)
        assert assert_dict_has_keys(opencv_decode_result, target_keys)
        assert opencv_decode_result['original_shape'] == (256, 340)
        assert np.shape(opencv_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test OpenCV with 2 dim input
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames,
                                               2)[:, np.newaxis]
        opencv_init = OpenCVInit()
        opencv_init_result = opencv_init(video_result)
        video_result['video_reader'] = opencv_init_result['video_reader']

        opencv_decode = OpenCVDecode()
        opencv_decode_result = opencv_decode(video_result)
        assert assert_dict_has_keys(opencv_decode_result, target_keys)
        assert opencv_decode_result['original_shape'] == (256, 340)
        assert np.shape(opencv_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test OpenCV with 1 dim input when start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames, 3)
        opencv_init = OpenCVInit()
        opencv_init_result = opencv_init(video_result)
        video_result['video_reader'] = opencv_init_result['video_reader']

        # test OpenCV with 1 dim input
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 3)
        opencv_init = OpenCVInit()
        opencv_init_result = opencv_init(video_result)
        video_result['video_reader'] = opencv_init_result['video_reader']

        opencv_decode = OpenCVDecode()
        opencv_decode_result = opencv_decode(video_result)
        assert assert_dict_has_keys(opencv_decode_result, target_keys)
        assert opencv_decode_result['original_shape'] == (256, 340)
        assert np.shape(opencv_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

    def test_rawframe_decode(self):
        target_keys = ['frame_inds', 'imgs', 'original_shape', 'modality']

        # test frame selector with 2 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 2)[:,
                                                                  np.newaxis]
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1

        inputs['gt_bboxes'] = np.array([[0, 0, 1, 1]])
        inputs['proposals'] = np.array([[0, 0, 1, 1]])
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 2 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)[:,
                                                                  np.newaxis]
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input when start_index = 0
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 5)
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 5)
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 2)
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input for flow images
        inputs = copy.deepcopy(self.flow_frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 2)
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 2)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input for flow images
        inputs = copy.deepcopy(self.flow_frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 2)
        assert results['original_shape'] == (240, 320)

        return
        # cannot install turbojpeg for CI
        if platform.system() != 'Windows':
            # test frame selector in turbojpeg decoding backend
            # when start_index = 0
            inputs = copy.deepcopy(self.frame_results)
            inputs['frame_inds'] = np.arange(0, self.total_frames, 5)
            # since the test images start with index 1, we plus 1 to frame_inds
            # in order to pass the CI
            inputs['frame_inds'] = inputs['frame_inds'] + 1
            frame_selector = RawFrameDecode(
                io_backend='disk', decoding_backend='turbojpeg')
            results = frame_selector(inputs)
            assert assert_dict_has_keys(results, target_keys)
            assert np.shape(results['imgs']) == (len(inputs['frame_inds']),
                                                 240, 320, 3)
            assert results['original_shape'] == (240, 320)

            # test frame selector in turbojpeg decoding backend
            inputs = copy.deepcopy(self.frame_results)
            inputs['frame_inds'] = np.arange(1, self.total_frames, 5)
            frame_selector = RawFrameDecode(
                io_backend='disk', decoding_backend='turbojpeg')
            results = frame_selector(inputs)
            assert assert_dict_has_keys(results, target_keys)
            assert np.shape(results['imgs']) == (len(inputs['frame_inds']),
                                                 240, 320, 3)
            assert results['original_shape'] == (240, 320)
            assert repr(frame_selector) == (
                f'{frame_selector.__class__.__name__}(io_backend=disk, '
                f'decoding_backend=turbojpeg)')

    def test_pyav_decode_motion_vector(self):
        pyav_init = PyAVInit()
        pyav = PyAVDecodeMotionVector()

        # test pyav with 2-dim input
        results = {
            'filename': self.video_path,
            'frame_inds': np.arange(0, 32, 1)[:, np.newaxis]
        }
        results = pyav_init(results)
        results = pyav(results)
        target_keys = ['motion_vectors']
        assert assert_dict_has_keys(results, target_keys)

        # test pyav with 1 dim input
        results = {
            'filename': self.video_path,
            'frame_inds': np.arange(0, 32, 1)
        }
        pyav_init = PyAVInit()
        results = pyav_init(results)
        pyav = PyAVDecodeMotionVector()
        results = pyav(results)

        assert assert_dict_has_keys(results, target_keys)


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
        with pytest.raises(TypeError):
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
        assert repr(load_localization_feature
                    ) == f'{load_localization_feature.__class__.__name__}'

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


class TestLocalization(BaseTestLoading):

    def test_generate_localization_label(self):
        action_result = copy.deepcopy(self.action_results)
        action_result['raw_feature'] = np.random.randn(400, 5)

        # test default setting
        target_keys = ['gt_bbox']
        generate_localization_labels = GenerateLocalizationLabels()
        generate_localization_labels_result = generate_localization_labels(
            action_result)
        assert assert_dict_has_keys(generate_localization_labels_result,
                                    target_keys)

        assert_array_almost_equal(
            generate_localization_labels_result['gt_bbox'], [[0.375, 0.625]],
            decimal=4)


class TestLoadImageFromFile:

    def test_load_img(self):
        data_prefix = osp.join(osp.dirname(__file__), '../../data')

        results = dict(img_path=osp.join(data_prefix, 'test.jpg'))
        transform = LoadRGBFromFile()
        results = transform(copy.deepcopy(results))
        assert results['img_path'] == osp.join(data_prefix, 'test.jpg')
        assert results['img'].shape == (240, 320, 3)
        assert results['img'].dtype == np.uint8
        assert results['img_shape'] == (240, 320)
        assert results['ori_shape'] == (240, 320)
        assert repr(transform) == transform.__class__.__name__ + \
            "(ignore_empty=False, to_float32=False, color_type='color', " + \
            "imdecode_backend='cv2', io_backend='disk')"

        # to_float32
        transform = LoadRGBFromFile(to_float32=True)
        results = transform(copy.deepcopy(results))
        assert results['img'].dtype == np.float32

        # test load empty
        fake_img_path = osp.join(data_prefix, 'fake.jpg')
        results['img_path'] = fake_img_path
        transform = LoadRGBFromFile(ignore_empty=False)
        with pytest.raises(FileNotFoundError):
            transform(copy.deepcopy(results))
        transform = LoadRGBFromFile(ignore_empty=True)
        assert transform(copy.deepcopy(results)) is None
