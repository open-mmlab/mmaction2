import copy

import numpy as np
from tests.test_data.test_pipelines.test_loading.test_base_loading import \
    TestLoading

from mmaction.datasets.pipelines import (DecordDecode, DecordInit,
                                         OpenCVDecode, OpenCVInit, PyAVDecode,
                                         PyAVDecodeMotionVector, PyAVInit,
                                         RawFrameDecode)


class TestDecode(TestLoading):

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
        assert self.check_keys_contain(decord_decode_result.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(decord_decode_result.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(decord_decode_result.keys(),
                                       target_keys)
        assert decord_decode_result['original_shape'] == (256, 340)
        assert np.shape(decord_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test Decord with 1 dim input
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 3)
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        video_result['video_reader'] = decord_init_result['video_reader']

        decord_decode = DecordDecode()
        decord_decode_result = decord_decode(video_result)
        assert self.check_keys_contain(decord_decode_result.keys(),
                                       target_keys)
        assert decord_decode_result['original_shape'] == (256, 340)
        assert np.shape(decord_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

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
        assert self.check_keys_contain(opencv_decode_result.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(opencv_decode_result.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(opencv_decode_result.keys(),
                                       target_keys)
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
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 2 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)[:,
                                                                  np.newaxis]
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
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
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 5)
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
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
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
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
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']) * 2,
                                             240, 320)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input for flow images
        inputs = copy.deepcopy(self.flow_frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']) * 2,
                                             240, 320)
        assert results['original_shape'] == (240, 320)

        # test frame selector in turbojpeg decording backend
        # when start_index = 0
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 5)
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        frame_selector = RawFrameDecode(
            io_backend='disk', decoding_backend='turbojpeg')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector in turbojpeg decording backend
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 5)
        frame_selector = RawFrameDecode(
            io_backend='disk', decoding_backend='turbojpeg')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)
        assert repr(frame_selector) == (f'{frame_selector.__class__.__name__}('
                                        f'io_backend=disk, '
                                        f'decoding_backend=turbojpeg)')

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
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)
        assert repr(pyav_decode) == (f'{pyav_decode.__class__.__name__}('
                                     f'multi_thread={False})')

        # test PyAV with 1 dim input and start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames, 5)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode()
        pyav_decode_result = pyav_decode(video_result)
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
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
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)
        assert repr(pyav_decode) == (f'{pyav_decode.__class__.__name__}('
                                     f'multi_thread={True})')

        # test PyAV with 2 dim input
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames,
                                               2)[:, np.newaxis]
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode()
        pyav_decode_result = pyav_decode(video_result)
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
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
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
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
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        assert repr(pyav_decode) == pyav_decode.__class__.__name__ + \
            f'(multi_thread={True})'

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
        assert self.check_keys_contain(results.keys(), target_keys)

        # test pyav with 1 dim input
        results = {
            'filename': self.video_path,
            'frame_inds': np.arange(0, 32, 1)
        }
        pyav_init = PyAVInit()
        results = pyav_init(results)
        pyav = PyAVDecodeMotionVector()
        results = pyav(results)

        assert self.check_keys_contain(results.keys(), target_keys)
