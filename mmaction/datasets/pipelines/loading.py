import os.path as osp

import mmcv
import numpy as np

from mmaction.utils import FileClient
from ..registry import PIPELINES


@PIPELINES.register_module
class SampleFrames(object):
    """Sample frames from the video.

    Required keys are "filename", added or modified keys are "total_frames",
    "frame_inds", "frame_interval" and "num_clips".

    Attributes:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 test_mode=False):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.test_mode = test_mode

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        Calculate the average interval for selected frames, and randomly
        shift them within offsets between [0, avg_interval]. If the total
        number of frames is smaller than clips num or origin frames length,
        it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        else:
            clip_offsets = np.zeros((self.num_clips, ))

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2 . If the total number of frames is not
        enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)

        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
        else:
            clip_offsets = np.zeros((self.num_clips, ))

        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose frame indices for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, results):
        if 'total_frames' not in results:
            # TODO: find a better way to get the total frames number for video
            video_reader = mmcv.VideoReader(results['filename'])
            total_frames = len(video_reader)
            results['total_frames'] = total_frames
        else:
            total_frames = results['total_frames']

        # TODO: index in different mode may be different
        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = np.mod(frame_inds, total_frames)

        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results


@PIPELINES.register_module
class PyAVDecode(object):
    """Using pyav to decode the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "filename" and "frame_inds",
    added or modified keys are "imgs" and "ori_shape".

    Attributes:
        multi_thread (bool): If set to True, it will apply multi
            thread processing. Default: False.
    """

    def __init__(self, multi_thread=False):
        self.multi_thread = multi_thread

    def __call__(self, results):
        try:
            import av
        except ImportError:
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip install av" to install PyAV first.')

        container = av.open(results['filename'])

        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        # set max indice to make early stop
        max_inds = max(results['frame_inds'])
        i = 0
        for frame in container.decode(video=0):
            if i > max_inds + 1:
                break
            imgs.append(frame.to_rgb().to_ndarray())
            i += 1

        imgs = np.array(imgs)
        # the available frame in pyav may be less than its length,
        # which may raise error
        if len(imgs) <= max_inds:
            results['frame_inds'] = np.mod(results['frame_inds'], len(imgs))

        imgs = imgs[results['frame_inds']]
        results['imgs'] = np.array(imgs)
        results['ori_shape'] = imgs.shape[1:3]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(multi_thread={self.multi_thread})'


@PIPELINES.register_module
class DecordDecode(object):
    """Using decord to decode the video.

    Decord: https://github.com/zhreshold/decord

    Required keys are "filename" and "frame_inds",
    added or modified keys are "imgs" and "ori_shape".
    """

    def __call__(self, results):
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        container = decord.VideoReader(results['filename'])
        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        for frame_idx in results['frame_inds']:
            cur_frame = container[frame_idx].asnumpy()
            imgs.append(cur_frame)
        imgs = np.array(imgs)

        results['imgs'] = np.array(imgs)
        results['ori_shape'] = imgs.shape[1:3]
        return results


@PIPELINES.register_module
class OpenCVDecode(object):
    """Using OpenCV to decode the video.

    Required keys are "filename" and "frame_inds",
    added or modified keys are "imgs" and "ori_shape".
    """

    def __call__(self, results):
        container = mmcv.VideoReader(results['filename'])
        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        for frame_ind in results['frame_inds']:
            cur_frame = container[frame_ind]
            # last frame may be None in OpenCV
            while isinstance(cur_frame, type(None)):
                frame_ind -= 1
                cur_frame = container[frame_ind]
            imgs.append(cur_frame)

        imgs = np.array(imgs)
        # The default channel order of OpenCV is BGR, thus we change it to RGB
        imgs = imgs[:, :, :, ::-1]
        results['imgs'] = np.array(imgs)
        results['ori_shape'] = imgs.shape[1:3]

        return results


@PIPELINES.register_module
class FrameSelector(object):
    """Select raw frames with given indices

    Required keys are "file_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "imgs" and "ori_shape".

    Attributes:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        for frame_idx in results['frame_inds']:
            # temporary solution for frame index offset.
            # TODO: add offset attributes in datasets.
            if frame_idx == 0:
                frame_idx += 1
            filepath = osp.join(directory, filename_tmpl.format(frame_idx))
            img_bytes = self.file_client.get(filepath)
            # Get frame with channel order RGB directly.
            cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            imgs.append(cur_frame)

        imgs = np.array(imgs)
        results['imgs'] = imgs
        results['ori_shape'] = imgs.shape[1:3]

        return results
