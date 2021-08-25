# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import sys
from multiprocessing import Pool

import mmcv
import numpy as np
from scipy.io import wavfile

try:
    import librosa
    import lws
except ImportError:
    print('Please import librosa, lws first.')

sys.path.append('..')

SILENCE_THRESHOLD = 2
FMIN = 125
FMAX = 7600
FRAME_SHIFT_MS = None
MIN_LEVEL_DB = -100
REF_LEVEL_DB = 20
RESCALING = True
RESCALING_MAX = 0.999
ALLOW_CLIPPING_IN_NORMALIZATION = True
LOG_SCALE_MIN = -32.23619130191664
NORM_AUDIO = True


class AudioTools:
    """All methods related to audio feature extraction. Code Reference:

            <https://github.com/r9y9/deepvoice3_pytorch>`_,
            <https://pypi.org/project/lws/1.2.6/>`_.

    Args:
        frame_rate (int): The frame rate per second of the video. Default: 30.
        sample_rate (int): The sample rate for audio sampling. Default: 16000.
        num_mels (int): Number of channels of the melspectrogram. Default: 80.
        fft_size (int): fft_size / sample_rate is window size. Default: 1280.
        hop_size (int): hop_size / sample_rate is step size. Default: 320.
    """

    def __init__(self,
                 frame_rate=30,
                 sample_rate=16000,
                 num_mels=80,
                 fft_size=1280,
                 hop_size=320,
                 spectrogram_type='lws'):
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.silence_threshold = SILENCE_THRESHOLD
        self.num_mels = num_mels
        self.fmin = FMIN
        self.fmax = FMAX
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.frame_shift_ms = FRAME_SHIFT_MS
        self.min_level_db = MIN_LEVEL_DB
        self.ref_level_db = REF_LEVEL_DB
        self.rescaling = RESCALING
        self.rescaling_max = RESCALING_MAX
        self.allow_clipping_in_normalization = ALLOW_CLIPPING_IN_NORMALIZATION
        self.log_scale_min = LOG_SCALE_MIN
        self.norm_audio = NORM_AUDIO
        self.spectrogram_type = spectrogram_type
        assert spectrogram_type in ['lws', 'librosa']

    def load_wav(self, path):
        """Load an audio file into numpy array."""
        return librosa.core.load(path, sr=self.sample_rate)[0]

    @staticmethod
    def audio_normalize(samples, desired_rms=0.1, eps=1e-4):
        """RMS normalize the audio data."""
        rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
        samples = samples * (desired_rms / rms)
        return samples

    def generate_spectrogram_magphase(self, audio, with_phase=False):
        """Separate a complex-valued spectrogram D into its magnitude (S)

            and phase (P) components, so that D = S * P.

        Args:
            audio (np.ndarray): The input audio signal.
            with_phase (bool): Determines whether to output the
                phase components. Default: False.

        Returns:
            np.ndarray: magnitude and phase component of the complex-valued
                spectrogram.
        """
        spectro = librosa.core.stft(
            audio,
            hop_length=self.get_hop_size(),
            n_fft=self.fft_size,
            center=True)
        spectro_mag, spectro_phase = librosa.core.magphase(spectro)
        spectro_mag = np.expand_dims(spectro_mag, axis=0)
        if with_phase:
            spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
            return spectro_mag, spectro_phase

        return spectro_mag

    def save_wav(self, wav, path):
        """Save the wav to disk."""
        # 32767 = (2 ^ 15 - 1) maximum of int16
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, self.sample_rate, wav.astype(np.int16))

    def trim(self, quantized):
        """Trim the audio wavfile."""
        start, end = self.start_and_end_indices(quantized,
                                                self.silence_threshold)
        return quantized[start:end]

    def adjust_time_resolution(self, quantized, mel):
        """Adjust time resolution by repeating features.

        Args:
            quantized (np.ndarray): (T,)
            mel (np.ndarray): (N, D)

        Returns:
            tuple: Tuple of (T,) and (T, D)
        """
        assert quantized.ndim == 1
        assert mel.ndim == 2

        upsample_factor = quantized.size // mel.shape[0]
        mel = np.repeat(mel, upsample_factor, axis=0)
        n_pad = quantized.size - mel.shape[0]
        if n_pad != 0:
            assert n_pad > 0
            mel = np.pad(
                mel, [(0, n_pad), (0, 0)], mode='constant', constant_values=0)

        # trim
        start, end = self.start_and_end_indices(quantized,
                                                self.silence_threshold)

        return quantized[start:end], mel[start:end, :]

    @staticmethod
    def start_and_end_indices(quantized, silence_threshold=2):
        """Trim the audio file when reaches the silence threshold."""
        for start in range(quantized.size):
            if abs(quantized[start] - 127) > silence_threshold:
                break
        for end in range(quantized.size - 1, 1, -1):
            if abs(quantized[end] - 127) > silence_threshold:
                break

        assert abs(quantized[start] - 127) > silence_threshold
        assert abs(quantized[end] - 127) > silence_threshold

        return start, end

    def melspectrogram(self, y):
        """Generate the melspectrogram."""
        D = self._lws_processor().stft(y).T
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        if not self.allow_clipping_in_normalization:
            assert S.max() <= 0 and S.min() - self.min_level_db >= 0
        return self._normalize(S)

    def get_hop_size(self):
        """Calculate the hop size."""
        hop_size = self.hop_size
        if hop_size is None:
            assert self.frame_shift_ms is not None
            hop_size = int(self.frame_shift_ms / 1000 * self.sample_rate)
        return hop_size

    def _lws_processor(self):
        """Perform local weighted sum.

        Please refer to <https://pypi.org/project/lws/1.2.6/>`_.
        """
        return lws.lws(self.fft_size, self.get_hop_size(), mode='speech')

    @staticmethod
    def lws_num_frames(length, fsize, fshift):
        """Compute number of time frames of lws spectrogram.

        Please refer to <https://pypi.org/project/lws/1.2.6/>`_.
        """
        pad = (fsize - fshift)
        if length % fshift == 0:
            M = (length + pad * 2 - fsize) // fshift + 1
        else:
            M = (length + pad * 2 - fsize) // fshift + 2
        return M

    def lws_pad_lr(self, x, fsize, fshift):
        """Compute left and right padding lws internally uses.

        Please refer to <https://pypi.org/project/lws/1.2.6/>`_.
        """
        M = self.lws_num_frames(len(x), fsize, fshift)
        pad = (fsize - fshift)
        T = len(x) + 2 * pad
        r = (M - 1) * fshift + fsize - T
        return pad, pad + r

    def _linear_to_mel(self, spectrogram):
        """Warp linear scale spectrograms to the mel scale.

        Please refer to <https://github.com/r9y9/deepvoice3_pytorch>`_
        """
        global _mel_basis
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _build_mel_basis(self):
        """Build mel filters.

        Please refer to <https://github.com/r9y9/deepvoice3_pytorch>`_
        """
        assert self.fmax <= self.sample_rate // 2
        return librosa.filters.mel(
            self.sample_rate,
            self.fft_size,
            fmin=self.fmin,
            fmax=self.fmax,
            n_mels=self.num_mels)

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    @staticmethod
    def _db_to_amp(x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, S):
        return np.clip((S - self.min_level_db) / -self.min_level_db, 0, 1)

    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.min_level_db) + self.min_level_db

    def read_audio(self, audio_path):
        wav = self.load_wav(audio_path)
        if self.norm_audio:
            wav = self.audio_normalize(wav)
        else:
            wav = wav / np.abs(wav).max()

        return wav

    def audio_to_spectrogram(self, wav):
        if self.spectrogram_type == 'lws':
            spectrogram = self.melspectrogram(wav).astype(np.float32).T
        elif self.spectrogram_type == 'librosa':
            spectrogram = self.generate_spectrogram_magphase(wav)
        return spectrogram


def extract_audio_feature(wav_path, audio_tools, mel_out_dir):
    file_name, _ = osp.splitext(osp.basename(wav_path))
    # Write the spectrograms to disk:
    mel_filename = os.path.join(mel_out_dir, file_name + '.npy')
    if not os.path.exists(mel_filename):
        try:
            wav = audio_tools.read_audio(wav_path)

            spectrogram = audio_tools.audio_to_spectrogram(wav)

            np.save(
                mel_filename,
                spectrogram.astype(np.float32),
                allow_pickle=False)

        except BaseException:
            print(f'Read audio [{wav_path}] failed.')


if __name__ == '__main__':
    audio_tools = AudioTools(
        fft_size=512, hop_size=256)  # window_size:32ms hop_size:16ms

    parser = argparse.ArgumentParser()
    parser.add_argument('audio_home_path', type=str)
    parser.add_argument('spectrogram_save_path', type=str)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--ext', default='.m4a')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--part', type=str, default='1/1')
    args = parser.parse_args()

    mmcv.mkdir_or_exist(args.spectrogram_save_path)

    files = glob.glob(
        osp.join(args.audio_home_path, '*/' * args.level, '*' + args.ext))
    print(f'found {len(files)} files.')
    files = sorted(files)
    if args.part is not None:
        [this_part, num_parts] = [int(i) for i in args.part.split('/')]
        part_len = len(files) // num_parts

    p = Pool(args.num_workers)
    for file in files[part_len * (this_part - 1):(
            part_len * this_part) if this_part != num_parts else len(files)]:
        p.apply_async(
            extract_audio_feature,
            args=(file, audio_tools, args.spectrogram_save_path))
    p.close()
    p.join()
