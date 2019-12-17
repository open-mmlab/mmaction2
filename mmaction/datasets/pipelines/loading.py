import os.path as osp

import av
import numpy as np

from ..registry import PIPELINES


def sample_clips(num_frames,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False):
    """Sample clips from the video.

    Args:
        num_frames (int): Total frames of the video.
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
        num_clips (int): Number of clips to be sampled.
        temporal_jitter (bool): Whether to apply temporal jittering.

    Returns:
        np.ndarray: Shape (num_clips, clip_len)
    """
    ori_clip_len = clip_len * frame_interval
    avg_interval = (num_frames - ori_clip_len) // num_clips
    if avg_interval > 0:
        base_offsets = np.arange(num_clips) * avg_interval
        clip_offsets = base_offsets + np.random.randint(
            avg_interval, size=num_clips)
    elif num_frames > max(num_clips, ori_clip_len):
        clip_offsets = np.sort(
            np.random.randint(num_frames - ori_clip_len + 1, size=num_clips))
    else:
        clip_offsets = np.zeros((num_clips, ))

    frame_inds = clip_offsets + np.arange(clip_len)[None, :] * frame_interval

    if temporal_jitter:
        perframe_offsets = np.random.randint((num_clips, frame_interval),
                                             size=clip_len)
        frame_inds += perframe_offsets

    return frame_inds


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
