# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
from mmcv.utils import assert_dict_has_keys

from mmaction.datasets.pipelines import (AudioDecode, AudioDecodeInit,
                                         DecordDecode, DecordInit,
                                         OpenCVDecode, OpenCVInit, PIMSDecode,
                                         PIMSInit, PyAVDecode,
                                         PyAVDecodeMotionVector, PyAVInit,
                                         RawFrameDecode)
from .base import BaseTestLoading


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
        target_keys = ['video_reader', 'total_frames']
        video_result = copy.deepcopy(self.video_results)
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        assert assert_dict_has_keys(decord_init_result, target_keys)
        assert decord_init_result['total_frames'] == len(
            decord_init_result['video_reader'])
        assert repr(decord_init) == (f'{decord_init.__class__.__name__}('
                                     f'io_backend=disk, '
                                     f'num_threads={1})')

    def test_decord_decode(self):
        target_keys = ['frame_inds', 'imgs', 'original_shape']

        # test Decord with 2 dim input and start_index = 0
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

        # test Decord with 1 dim input and start_index = 0
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

        # test Decord with 2 dim input and start_index = 0
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

        # test Decord with 1 dim input
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
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']) * 2,
                                             240, 320)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input for flow images
        inputs = copy.deepcopy(self.flow_frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']) * 2,
                                             240, 320)
        assert results['original_shape'] == (240, 320)

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
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector in turbojpeg decoding backend
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 5)
        frame_selector = RawFrameDecode(
            io_backend='disk', decoding_backend='turbojpeg')
        results = frame_selector(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)
        assert repr(frame_selector) == (f'{frame_selector.__class__.__name__}('
                                        f'io_backend=disk, '
                                        f'decoding_backend=turbojpeg)')

    def test_audio_decode_init(self):
        target_keys = ['audios', 'length', 'sample_rate']
        inputs = copy.deepcopy(self.audio_results)
        audio_decode_init = AudioDecodeInit()
        results = audio_decode_init(inputs)
        assert assert_dict_has_keys(results, target_keys)

        # test when no audio file exists
        inputs = copy.deepcopy(self.audio_results)
        inputs['audio_path'] = 'foo/foo/bar.wav'
        audio_decode_init = AudioDecodeInit()
        results = audio_decode_init(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert results['audios'].shape == (10.0 *
                                           audio_decode_init.sample_rate, )
        assert repr(audio_decode_init) == (
            f'{audio_decode_init.__class__.__name__}('
            f'io_backend=disk, '
            f'sample_rate=16000, '
            f'pad_method=zero)')

    def test_audio_decode(self):
        target_keys = ['frame_inds', 'audios']
        inputs = copy.deepcopy(self.audio_results)
        inputs['frame_inds'] = np.arange(0, self.audio_total_frames,
                                         2)[:, np.newaxis]
        inputs['num_clips'] = 1
        inputs['length'] = 1280
        audio_selector = AudioDecode()
        results = audio_selector(inputs)
        assert assert_dict_has_keys(results, target_keys)

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
