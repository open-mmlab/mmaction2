import copy

import numpy as np
from tests.test_data.test_pipelines.test_loading.test_base_loading import \
    TestLoading

from mmaction.datasets.pipelines import (AudioDecode, AudioDecodeInit,
                                         AudioFeatureSelector,
                                         LoadAudioFeature)


class TestAudio(TestLoading):

    def test_audio_decode_init(self):
        target_keys = ['audios', 'length', 'sample_rate']
        inputs = copy.deepcopy(self.audio_results)
        audio_decode_init = AudioDecodeInit()
        results = audio_decode_init(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)

        # test when no audio file exists
        inputs = copy.deepcopy(self.audio_results)
        inputs['audio_path'] = 'foo/foo/bar.wav'
        audio_decode_init = AudioDecodeInit()
        results = audio_decode_init(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
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
        assert self.check_keys_contain(results.keys(), target_keys)

    def test_load_audio_feature(self):
        target_keys = ['audios']
        inputs = copy.deepcopy(self.audio_feature_results)
        load_audio_feature = LoadAudioFeature()
        results = load_audio_feature(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)

        # test when no audio feature file exists
        inputs = copy.deepcopy(self.audio_feature_results)
        inputs['audio_path'] = 'foo/foo/bar.npy'
        load_audio_feature = LoadAudioFeature()
        results = load_audio_feature(inputs)
        assert results['audios'].shape == (640, 80)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert repr(load_audio_feature) == (
            f'{load_audio_feature.__class__.__name__}('
            f'pad_method=zero)')

    def test_audio_feature_selector(self):
        target_keys = ['audios']
        # test frame selector with 2 dim input
        inputs = copy.deepcopy(self.audio_feature_results)
        inputs['frame_inds'] = np.arange(0, self.audio_total_frames,
                                         2)[:, np.newaxis]
        inputs['num_clips'] = 1
        inputs['length'] = 1280
        audio_feature_selector = AudioFeatureSelector()
        results = audio_feature_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert repr(audio_feature_selector) == (
            f'{audio_feature_selector.__class__.__name__}('
            f'fix_length={128})')
