import copy

from tests.test_data.test_pipelines.test_loading.test_base_loading import \
    TestLoading

from mmaction.datasets.pipelines import DecordInit, OpenCVInit, PyAVInit


class TestInit(TestLoading):

    def test_pyav_init(self):
        target_keys = ['video_reader', 'total_frames']
        video_result = copy.deepcopy(self.video_results)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        assert self.check_keys_contain(pyav_init_result.keys(), target_keys)
        assert pyav_init_result['total_frames'] == 300
        assert repr(
            pyav_init) == f'{pyav_init.__class__.__name__}(io_backend=disk)'

    def test_decord_init(self):
        target_keys = ['video_reader', 'total_frames']
        video_result = copy.deepcopy(self.video_results)
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        assert self.check_keys_contain(decord_init_result.keys(), target_keys)
        assert decord_init_result['total_frames'] == len(
            decord_init_result['video_reader'])
        assert repr(decord_init) == (f'{decord_init.__class__.__name__}('
                                     f'io_backend=disk, '
                                     f'num_threads={1})')

    def test_opencv_init(self):
        target_keys = ['new_path', 'video_reader', 'total_frames']
        video_result = copy.deepcopy(self.video_results)
        opencv_init = OpenCVInit()
        opencv_init_result = opencv_init(video_result)
        assert self.check_keys_contain(opencv_init_result.keys(), target_keys)
        assert opencv_init_result['total_frames'] == len(
            opencv_init_result['video_reader'])
        assert repr(opencv_init) == (f'{opencv_init.__class__.__name__}('
                                     f'io_backend=disk)')
