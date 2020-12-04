import numpy as np
import pytest
from tests.test_data.test_pipelines.test_augmentations.test_base_aug import \
    TestAugumentations

from mmaction.datasets.pipelines import AudioAmplify, MelSpectrogram


class TestAudio(TestAugumentations):

    def test_audio_amplify(self):
        target_keys = ['audios', 'amplify_ratio']
        with pytest.raises(TypeError):
            # ratio should be float
            AudioAmplify(1)

        audio = (np.random.rand(8, ))
        results = dict(audios=audio)
        amplifier = AudioAmplify(1.5)
        results = amplifier(results)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert repr(amplifier) == (f'{amplifier.__class__.__name__}'
                                   f'(ratio={amplifier.ratio})')

    def test_melspectrogram(self):
        target_keys = ['audios']
        with pytest.raises(TypeError):
            # ratio should be float
            MelSpectrogram(window_size=12.5)
        audio = (np.random.rand(1, 160000))

        # test padding
        results = dict(audios=audio, sample_rate=16000)
        results['num_clips'] = 1
        results['sample_rate'] = 16000
        mel = MelSpectrogram()
        results = mel(results)
        assert self.check_keys_contain(results.keys(), target_keys)

        # test truncating
        audio = (np.random.rand(1, 160000))
        results = dict(audios=audio, sample_rate=16000)
        results['num_clips'] = 1
        results['sample_rate'] = 16000
        mel = MelSpectrogram(fixed_length=1)
        results = mel(results)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert repr(mel) == (f'{mel.__class__.__name__}'
                             f'(window_size={mel.window_size}), '
                             f'step_size={mel.step_size}, '
                             f'n_mels={mel.n_mels}, '
                             f'fixed_length={mel.fixed_length})')
