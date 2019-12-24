import os.path as osp

import av
import mmcv
import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module
class SampleFrames:
    """Sample frames from the video.

    Required keys are "filename", added or modified keys are "total_frames"
    and "frame_inds".

    Attributes:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
        num_clips (int): Number of clips to be sampled.
        temporal_jitter (bool): Whether to apply temporal jittering.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter

    def _sample_clips(self, num_frames):
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len) // self.num_clips
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

    def __call__(self, results):
        video_reader = mmcv.VideoReader(results['filename'])
        total_frames = len(video_reader)
        results['total_frames'] = total_frames

        clip_offsets = self._sample_clips(total_frames)

        frame_inds = clip_offsets + np.arange(
            self.clip_len)[None, :] * self.frame_interval

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                (self.num_clips, self.frame_interval), size=self.clip_len)
            frame_inds += perframe_offsets

        results['frame_inds'] = frame_inds

        return results


@PIPELINES.register_module
class PyAVDecode:

    def __init__(self, multi_thread=False):
        self.multi_thread = multi_thread

    def __call__(self, results):
        if results['data_prefix'] is not None:
            filename = osp.join(results['data_prefix'],
                                results['video_info']['filename'])
        else:
            filename = results['video_info']['filename']

        container = av.open(filename)
        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
