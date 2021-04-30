import numpy as np  # isort: skip
from numpy.testing import assert_array_equal  # isort: skip

from mmaction.datasets.pipelines.pose_loading import UniformSampleFrames


class TestPoseLoading:

    def test_uniform_sample_frames(self):
        results = dict(total_frames=64, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=1, test_mode=True, seed=0)

        assert str(sampling) == ('UniformSampleFrames(clip_len=8, '
                                 'num_clips=1, test_mode=True, seed=0)')
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert_array_equal(sampling_results['frame_inds'],
                           np.array([4, 15, 21, 24, 35, 43, 51, 63]))

        results = dict(total_frames=15, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=1, test_mode=True, seed=0)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert_array_equal(sampling_results['frame_inds'],
                           np.array([0, 2, 4, 6, 8, 9, 11, 13]))

        results = dict(total_frames=7, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=1, test_mode=True, seed=0)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert_array_equal(sampling_results['frame_inds'],
                           np.array([0, 1, 2, 3, 4, 5, 6, 0]))

        results = dict(total_frames=64, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=4, test_mode=True, seed=0)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 4
        assert_array_equal(
            sampling_results['frame_inds'],
            np.array([
                4, 15, 21, 24, 35, 43, 51, 63, 1, 11, 21, 26, 36, 47, 54, 56,
                0, 12, 18, 25, 38, 47, 55, 62, 0, 9, 21, 25, 37, 40, 49, 60
            ]))

        results = dict(total_frames=64, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=1, test_mode=False, seed=0)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert len(sampling_results['frame_inds']) == 8
