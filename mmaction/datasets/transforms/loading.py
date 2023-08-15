# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import io
import os
import os.path as osp
import shutil
from typing import Dict, List, Optional, Union

import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient

from mmaction.registry import TRANSFORMS
from mmaction.utils import get_random_string, get_shm_dir, get_thread_id


@TRANSFORMS.register_module()
class LoadRGBFromFile(BaseTransform):
    """Load a RGB image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 io_backend: str = 'disk',
                 ignore_empty: bool = False,
                 **kwargs) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.file_client = FileClient(io_backend, **kwargs)
        self.io_backend = io_backend

    def transform(self, results: dict) -> dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes,
                flag=self.color_type,
                channel_order='rgb',
                backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f"io_backend='{self.io_backend}')")
        return repr_str


@TRANSFORMS.register_module()
class LoadHVULabel(BaseTransform):
    """Convert the HVU label from dictionaries to torch tensors.

    Required keys are "label", "categories", "category_nums", added or modified
    keys are "label", "mask" and "category_mask".
    """

    def __init__(self, **kwargs):
        self.hvu_initialized = False
        self.kwargs = kwargs

    def init_hvu_info(self, categories, category_nums):
        """Initialize hvu information."""
        assert len(categories) == len(category_nums)
        self.categories = categories
        self.category_nums = category_nums
        self.num_categories = len(self.categories)
        self.num_tags = sum(self.category_nums)
        self.category2num = dict(zip(categories, category_nums))
        self.start_idx = [0]
        for i in range(self.num_categories - 1):
            self.start_idx.append(self.start_idx[-1] + self.category_nums[i])
        self.category2startidx = dict(zip(categories, self.start_idx))
        self.hvu_initialized = True

    def transform(self, results):
        """Convert the label dictionary to 3 tensors: "label", "mask" and
        "category_mask".

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        if not self.hvu_initialized:
            self.init_hvu_info(results['categories'], results['category_nums'])

        onehot = torch.zeros(self.num_tags)
        onehot_mask = torch.zeros(self.num_tags)
        category_mask = torch.zeros(self.num_categories)

        for category, tags in results['label'].items():
            # skip if not training on this category
            if category not in self.categories:
                continue
            category_mask[self.categories.index(category)] = 1.
            start_idx = self.category2startidx[category]
            category_num = self.category2num[category]
            tags = [idx + start_idx for idx in tags]
            onehot[tags] = 1.
            onehot_mask[start_idx:category_num + start_idx] = 1.

        results['label'] = onehot
        results['mask'] = onehot_mask
        results['category_mask'] = category_mask
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'hvu_initialized={self.hvu_initialized})')
        return repr_str


@TRANSFORMS.register_module()
class SampleFrames(BaseTransform):
    """Sample frames from the video.

    Required Keys:

        - total_frames
        - start_index

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Defaults to 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Defaults to False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Defaults to False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Defaults to 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Defaults to False.
        target_fps (optional, int): Convert input videos with arbitrary frame
            rates to the unified target FPS before sampling frames. If
            ``None``, the frame rate will not be adjusted. Defaults to
            ``None``.
    """

    def __init__(self,
                 clip_len: int,
                 frame_interval: int = 1,
                 num_clips: int = 1,
                 temporal_jitter: bool = False,
                 twice_sample: bool = False,
                 out_of_bound_opt: str = 'loop',
                 test_mode: bool = False,
                 keep_tail_frames: bool = False,
                 target_fps: Optional[int] = None,
                 **kwargs) -> None:

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        self.target_fps = target_fps
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames: int,
                         ori_clip_len: float) -> np.array:
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int32)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int32)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int32)

        return clip_offsets

    def _get_test_clips(self, num_frames: int,
                        ori_clip_len: float) -> np.array:
        """Get clip offsets in test mode.

        If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        if self.clip_len == 1:  # 2D recognizer
            # assert self.frame_interval == 1
            avg_interval = num_frames / float(self.num_clips)
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + avg_interval / 2.0
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:  # 3D recognizer
            max_offset = max(num_frames - ori_clip_len, 0)
            if self.twice_sample:
                num_clips = self.num_clips * 2
            else:
                num_clips = self.num_clips
            if num_clips > 1:
                num_segments = self.num_clips - 1
                # align test sample strategy with `PySlowFast` repo
                if self.target_fps is not None:
                    offset_between = np.floor(max_offset / float(num_segments))
                    clip_offsets = np.arange(num_clips) * offset_between
                else:
                    offset_between = max_offset / float(num_segments)
                    clip_offsets = np.arange(num_clips) * offset_between
                    clip_offsets = np.round(clip_offsets)
            else:
                clip_offsets = np.array([max_offset // 2])
        return clip_offsets

    def _sample_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames, ori_clip_len)
        else:
            clip_offsets = self._get_train_clips(num_frames, ori_clip_len)

        return clip_offsets

    def _get_ori_clip_len(self, fps_scale_ratio: float) -> float:
        """calculate length of clip segment for different strategy.

        Args:
            fps_scale_ratio (float): Scale ratio to adjust fps.
        """
        if self.target_fps is not None:
            # align test sample strategy with `PySlowFast` repo
            ori_clip_len = self.clip_len * self.frame_interval
            ori_clip_len = np.maximum(1, ori_clip_len * fps_scale_ratio)
        elif self.test_mode:
            ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        else:
            ori_clip_len = self.clip_len * self.frame_interval

        return ori_clip_len

    def transform(self, results: dict) -> dict:
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        # if can't get fps, same value of `fps` and `target_fps`
        # will perform nothing
        fps = results.get('avg_fps')
        if self.target_fps is None or not fps:
            fps_scale_ratio = 1.0
        else:
            fps_scale_ratio = fps / self.target_fps
        ori_clip_len = self._get_ori_clip_len(fps_scale_ratio)
        clip_offsets = self._sample_clips(total_frames, ori_clip_len)

        if self.target_fps:
            frame_inds = clip_offsets[:, None] + np.linspace(
                0, ori_clip_len - 1, self.clip_len).astype(np.int32)
        else:
            frame_inds = clip_offsets[:, None] + np.arange(
                self.clip_len)[None, :] * self.frame_interval
            frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@TRANSFORMS.register_module()
class UniformSample(BaseTransform):
    """Uniformly sample frames from the video.

    Modified from https://github.com/facebookresearch/SlowFast/blob/64a
    bcc90ccfdcbb11cf91d6e525bed60e92a8796/slowfast/datasets/ssv2.py#L159.

    To sample an n-frame clip from the video. UniformSample basically
    divides the video into n segments of equal length and randomly samples one
    frame from each segment.

    Required keys:

        - total_frames
        - start_index

    Added keys:

        - frame_inds
        - clip_len
        - frame_interval
        - num_clips

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Defaults to 1.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(self,
                 clip_len: int,
                 num_clips: int = 1,
                 test_mode: bool = False) -> None:

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode

    def _get_sample_clips(self, num_frames: int) -> np.ndarray:
        """To sample an n-frame clip from the video. UniformSample basically
        divides the video into n segments of equal length and randomly samples
        one frame from each segment. When the duration of video frames is
        shorter than the desired length of the target clip, this approach will
        duplicate the sampled frame instead of looping the sample in "loop"
        mode. In the test mode, when we need to sample multiple clips,
        specifically 'n' clips, this method will further divide the segments
        based on the number of clips to be sampled. The 'i-th' clip will.

        sample the frame located at the position 'i * len(segment) / n'
        within the segment.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            seq (np.ndarray): the indexes of frames of sampled from the video.
        """
        seg_size = float(num_frames - 1) / self.clip_len
        inds = []
        if not self.test_mode:
            for i in range(self.clip_len):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                inds.append(np.random.randint(start, end + 1))
        else:
            duration = seg_size / (self.num_clips + 1)
            for k in range(self.num_clips):
                for i in range(self.clip_len):
                    start = int(np.round(seg_size * i))
                    frame_index = start + int(duration * (k + 1))
                    inds.append(frame_index)

        return np.array(inds)

    def transform(self, results: Dict) -> Dict:
        """Perform the Uniform Sampling.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        num_frames = results['total_frames']

        inds = self._get_sample_clips(num_frames)
        start_index = results['start_index']
        inds = inds + start_index

        results['frame_inds'] = inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'test_mode={self.test_mode}')
        return repr_str


@TRANSFORMS.register_module()
class UntrimmedSampleFrames(BaseTransform):
    """Sample frames from the untrimmed video.

    Required keys are "filename", "total_frames", added or modified keys are
    "frame_inds", "clip_interval" and "num_clips".

    Args:
        clip_len (int): The length of sampled clips. Defaults to  1.
        clip_interval (int): Clip interval of adjacent center of sampled
            clips. Defaults to 16.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Defaults to 1.
    """

    def __init__(self, clip_len=1, clip_interval=16, frame_interval=1):
        self.clip_len = clip_len
        self.clip_interval = clip_interval
        self.frame_interval = frame_interval

    def transform(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        start_index = results['start_index']

        clip_centers = np.arange(self.clip_interval // 2, total_frames,
                                 self.clip_interval)
        num_clips = clip_centers.shape[0]
        frame_inds = clip_centers[:, None] + np.arange(
            -(self.clip_len // 2 * self.frame_interval),
            self.frame_interval *
            (self.clip_len -
             (self.clip_len // 2)), self.frame_interval)[None, :]
        # clip frame_inds to legal range
        frame_inds = np.clip(frame_inds, 0, total_frames - 1)

        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['clip_interval'] = self.clip_interval
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'clip_interval={self.clip_interval}, '
                    f'frame_interval={self.frame_interval})')
        return repr_str


@TRANSFORMS.register_module()
class DenseSampleFrames(SampleFrames):
    """Select frames from the video by dense sample strategy.

    Required keys:

    - total_frames
    - start_index

    Added keys:

    - frame_inds
    - clip_len
    - frame_interval
    - num_clips

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
           Defaults to 1.
        num_clips (int): Number of clips to be sampled. Defaults to 1.
        sample_range (int): Total sample range for dense sample.
            Defaults to 64.
        num_sample_positions (int): Number of sample start positions, Which is
            only used in test mode. Defaults to 10. That is to say, by default,
            there are at least 10 clips for one input sample in test mode.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Defaults to False.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(self,
                 *args,
                 sample_range: int = 64,
                 num_sample_positions: int = 10,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_range = sample_range
        self.num_sample_positions = num_sample_positions

    def _get_train_clips(self, num_frames: int) -> np.array:
        """Get clip offsets by dense sample strategy in train mode.

        It will calculate a sample position and sample interval and set
        start index 0 when sample_pos == 1 or randomly choose from
        [0, sample_pos - 1]. Then it will shift the start index by each
        base offset.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_idx = 0 if sample_position == 1 else np.random.randint(
            0, sample_position - 1)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = (base_offsets + start_idx) % num_frames
        return clip_offsets

    def _get_test_clips(self, num_frames: int) -> np.array:
        """Get clip offsets by dense sample strategy in test mode.

        It will calculate a sample position and sample interval and evenly
        sample several start indexes as start positions between
        [0, sample_position-1]. Then it will shift each start index by the
        base offsets.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_list = np.linspace(
            0, sample_position - 1, num=self.num_sample_positions, dtype=int)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = list()
        for start_idx in start_list:
            clip_offsets.extend((base_offsets + start_idx) % num_frames)
        clip_offsets = np.array(clip_offsets)
        return clip_offsets

    def _sample_clips(self, num_frames: int) -> np.array:
        """Choose clip offsets for the video in a given mode.

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

    def transform(self, results: dict) -> dict:
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'sample_range={self.sample_range}, '
                    f'num_sample_positions={self.num_sample_positions}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@TRANSFORMS.register_module()
class SampleAVAFrames(SampleFrames):

    def __init__(self, clip_len, frame_interval=2, test_mode=False):

        super().__init__(clip_len, frame_interval, test_mode=test_mode)

    def _get_clips(self, center_index, skip_offsets, shot_info):
        """Get clip offsets."""
        start = center_index - (self.clip_len // 2) * self.frame_interval
        end = center_index + ((self.clip_len + 1) // 2) * self.frame_interval
        frame_inds = list(range(start, end, self.frame_interval))
        if not self.test_mode:
            frame_inds = frame_inds + skip_offsets
        frame_inds = np.clip(frame_inds, shot_info[0], shot_info[1] - 1)
        return frame_inds

    def transform(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        fps = results['fps']
        timestamp = results['timestamp']
        timestamp_start = results['timestamp_start']
        start_index = results.get('start_index', 0)
        if results.get('total_frames') is not None:
            shot_info = (0, results['total_frames'])
        else:
            shot_info = results['shot_info']

        center_index = fps * (timestamp - timestamp_start) + start_index

        skip_offsets = np.random.randint(
            -self.frame_interval // 2, (self.frame_interval + 1) // 2,
            size=self.clip_len)
        frame_inds = self._get_clips(center_index, skip_offsets, shot_info)

        frame_inds = np.array(frame_inds, dtype=np.int32) + start_index
        results['frame_inds'] = frame_inds
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = 1
        results['crop_quadruple'] = np.array([0, 0, 1, 1], dtype=np.float32)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@TRANSFORMS.register_module()
class PyAVInit(BaseTransform):
    """Using pyav to initialize the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "filename",
    added or modified keys are "video_reader", and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results):
        """Perform the PyAV initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import av
        except ImportError:
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip install av" to install PyAV first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = av.open(file_obj)

        results['video_reader'] = container
        results['total_frames'] = container.streams.video[0].frames

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(io_backend={self.io_backend})'
        return repr_str


@TRANSFORMS.register_module()
class PyAVDecode(BaseTransform):
    """Using PyAV to decode the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        multi_thread (bool): If set to True, it will apply multi
            thread processing. Default: False.
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            the nearest key frames, which may be duplicated and inaccurate,
            and more suitable for large scene-based video datasets.
            Default: 'accurate'.
    """

    def __init__(self, multi_thread=False, mode='accurate'):
        self.multi_thread = multi_thread
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    @staticmethod
    def frame_generator(container, stream):
        """Frame generator for PyAV."""
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame:
                    return frame.to_rgb().to_ndarray()

    def transform(self, results):
        """Perform the PyAV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        if self.mode == 'accurate':
            # set max indice to make early stop
            max_inds = max(results['frame_inds'])
            i = 0
            for frame in container.decode(video=0):
                if i > max_inds + 1:
                    break
                imgs.append(frame.to_rgb().to_ndarray())
                i += 1

            # the available frame in pyav may be less than its length,
            # which may raise error
            results['imgs'] = [
                imgs[i % len(imgs)] for i in results['frame_inds']
            ]
        elif self.mode == 'efficient':
            for frame in container.decode(video=0):
                backup_frame = frame
                break
            stream = container.streams.video[0]
            for idx in results['frame_inds']:
                pts_scale = stream.average_rate * stream.time_base
                frame_pts = int(idx / pts_scale)
                container.seek(
                    frame_pts, any_frame=False, backward=True, stream=stream)
                frame = self.frame_generator(container, stream)
                if frame is not None:
                    imgs.append(frame)
                    backup_frame = frame
                else:
                    imgs.append(backup_frame)
            results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        results['video_reader'] = None
        del container

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(multi_thread={self.multi_thread}, mode={self.mode})'
        return repr_str


@TRANSFORMS.register_module()
class PIMSInit(BaseTransform):
    """Use PIMS to initialize the video.

    PIMS: https://github.com/soft-matter/pims

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will always use ``pims.PyAVReaderIndexed``
            to decode videos into accurate frames. If set to 'efficient', it
            will adopt fast seeking by using ``pims.PyAVReaderTimed``.
            Both will return the accurate frames in most cases.
            Default: 'accurate'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', mode='accurate', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def transform(self, results):
        """Perform the PIMS initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import pims
        except ImportError:
            raise ImportError('Please run "conda install pims -c conda-forge" '
                              'or "pip install pims" to install pims first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        if self.mode == 'accurate':
            container = pims.PyAVReaderIndexed(file_obj)
        else:
            container = pims.PyAVReaderTimed(file_obj)

        results['video_reader'] = container
        results['total_frames'] = len(container)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(io_backend={self.io_backend}, '
                    f'mode={self.mode})')
        return repr_str


@TRANSFORMS.register_module()
class PIMSDecode(BaseTransform):
    """Using PIMS to decode the videos.

    PIMS: https://github.com/soft-matter/pims

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".
    """

    def transform(self, results):
        """Perform the PIMS decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        imgs = [container[idx] for idx in frame_inds]

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@TRANSFORMS.register_module()
class PyAVDecodeMotionVector(PyAVDecode):
    """Using pyav to decode the motion vectors from video.

    Reference: https://github.com/PyAV-Org/PyAV/
        blob/main/tests/test_decode.py

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "motion_vectors", "frame_inds".
    """

    @staticmethod
    def _parse_vectors(mv, vectors, height, width):
        """Parse the returned vectors."""
        (w, h, src_x, src_y, dst_x,
         dst_y) = (vectors['w'], vectors['h'], vectors['src_x'],
                   vectors['src_y'], vectors['dst_x'], vectors['dst_y'])
        val_x = dst_x - src_x
        val_y = dst_y - src_y
        start_x = dst_x - w // 2
        start_y = dst_y - h // 2
        end_x = start_x + w
        end_y = start_y + h
        for sx, ex, sy, ey, vx, vy in zip(start_x, end_x, start_y, end_y,
                                          val_x, val_y):
            if (sx >= 0 and ex < width and sy >= 0 and ey < height):
                mv[sy:ey, sx:ex] = (vx, vy)

        return mv

    def transform(self, results):
        """Perform the PyAV motion vector decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        # set max index to make early stop
        max_idx = max(results['frame_inds'])
        i = 0
        stream = container.streams.video[0]
        codec_context = stream.codec_context
        codec_context.options = {'flags2': '+export_mvs'}
        for packet in container.demux(stream):
            for frame in packet.decode():
                if i > max_idx + 1:
                    break
                i += 1
                height = frame.height
                width = frame.width
                mv = np.zeros((height, width, 2), dtype=np.int8)
                vectors = frame.side_data.get('MOTION_VECTORS')
                if frame.key_frame:
                    # Key frame don't have motion vectors
                    assert vectors is None
                if vectors is not None and len(vectors) > 0:
                    mv = self._parse_vectors(mv, vectors.to_ndarray(), height,
                                             width)
                imgs.append(mv)

        results['video_reader'] = None
        del container

        # the available frame in pyav may be less than its length,
        # which may raise error
        results['motion_vectors'] = np.array(
            [imgs[i % len(imgs)] for i in results['frame_inds']])
        return results


@TRANSFORMS.register_module()
class DecordInit(BaseTransform):
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required Keys:

        - filename

    Added Keys:

        - video_reader
        - total_frames
        - fps

    Args:
        io_backend (str): io backend where frames are store.
            Defaults to ``'disk'``.
        num_threads (int): Number of thread to decode the video. Defaults to 1.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend: str = 'disk',
                 num_threads: int = 1,
                 **kwargs) -> None:
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def _get_video_reader(self, filename: str) -> object:
        if osp.splitext(filename)[0] == filename:
            filename = filename + '.mp4'
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        file_obj = io.BytesIO(self.file_client.get(filename))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        return container

    def transform(self, results: Dict) -> Dict:
        """Perform the Decord initialization.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        container = self._get_video_reader(results['filename'])
        results['total_frames'] = len(container)

        results['video_reader'] = container
        results['avg_fps'] = container.get_avg_fps()
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str


@TRANSFORMS.register_module()
class DecordDecode(BaseTransform):
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required Keys:

        - video_reader
        - frame_inds

    Added Keys:

        - imgs
        - original_shape
        - img_shape

    Args:
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            key frames, which may be duplicated and inaccurate, and more
            suitable for large scene-based video datasets.
            Defaults to ``'accurate'``.
    """

    def __init__(self, mode: str = 'accurate') -> None:
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def _decord_load_frames(self, container: object,
                            frame_inds: np.ndarray) -> List[np.ndarray]:
        if self.mode == 'accurate':
            imgs = container.get_batch(frame_inds).asnumpy()
            imgs = list(imgs)
        elif self.mode == 'efficient':
            # This mode is faster, however it always returns I-FRAME
            container.seek(0)
            imgs = list()
            for idx in frame_inds:
                container.seek(idx)
                frame = container.next()
                imgs.append(frame.asnumpy())
        return imgs

    def transform(self, results: Dict) -> Dict:
        """Perform the Decord decoding.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        imgs = self._decord_load_frames(container, frame_inds)

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}(mode={self.mode})'
        return repr_str


@TRANSFORMS.register_module()
class OpenCVInit(BaseTransform):
    """Using OpenCV to initialize the video_reader.

    Required keys are ``'filename'``, added or modified keys are `
    `'new_path'``, ``'video_reader'`` and ``'total_frames'``.

    Args:
        io_backend (str): io backend where frames are store.
            Defaults to ``'disk'``.
    """

    def __init__(self, io_backend: str = 'disk', **kwargs) -> None:
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None
        self.tmp_folder = None
        if self.io_backend != 'disk':
            random_string = get_random_string()
            thread_id = get_thread_id()
            self.tmp_folder = osp.join(get_shm_dir(),
                                       f'{random_string}_{thread_id}')
            os.mkdir(self.tmp_folder)

    def transform(self, results: dict) -> dict:
        """Perform the OpenCV initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.io_backend == 'disk':
            new_path = results['filename']
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend, **self.kwargs)

            thread_id = get_thread_id()
            # save the file of same thread at the same place
            new_path = osp.join(self.tmp_folder, f'tmp_{thread_id}.mp4')
            with open(new_path, 'wb') as f:
                f.write(self.file_client.get(results['filename']))

        container = mmcv.VideoReader(new_path)
        results['new_path'] = new_path
        results['video_reader'] = container
        results['total_frames'] = len(container)

        return results

    def __del__(self):
        if self.tmp_folder and osp.exists(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend})')
        return repr_str


@TRANSFORMS.register_module()
class OpenCVDecode(BaseTransform):
    """Using OpenCV to decode the video.

    Required keys are ``'video_reader'``, ``'filename'`` and ``'frame_inds'``,
    added or modified keys are ``'imgs'``, ``'img_shape'`` and
    ``'original_shape'``.
    """

    def transform(self, results: dict) -> dict:
        """Perform the OpenCV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
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

        results['video_reader'] = None
        del container

        imgs = np.array(imgs)
        # The default channel order of OpenCV is BGR, thus we change it to RGB
        imgs = imgs[:, :, :, ::-1]
        results['imgs'] = list(imgs)
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@TRANSFORMS.register_module()
class RawFrameDecode(BaseTransform):
    """Load and decode frames with given indices.

    Required Keys:

    - frame_dir
    - filename_tmpl
    - frame_inds
    - modality
    - offset (optional)

    Added Keys:

    - img
    - img_shape
    - original_shape

    Args:
        io_backend (str): IO backend where frames are stored.
            Defaults to ``'disk'``.
        decoding_backend (str): Backend used for image decoding.
            Defaults to ``'cv2'``.
    """

    def __init__(self,
                 io_backend: str = 'disk',
                 decoding_backend: str = 'cv2',
                 **kwargs) -> None:
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results: dict) -> dict:
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        cache = {}
        for i, frame_idx in enumerate(results['frame_inds']):
            # Avoid loading duplicated frames
            if frame_idx in cache:
                imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                continue
            else:
                cache[frame_idx] = i

            frame_idx += offset
            if modality == 'RGB':
                filepath = osp.join(directory, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.append(np.stack([x_frame, y_frame], axis=-1))
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str


@TRANSFORMS.register_module()
class InferencerPackInput(BaseTransform):

    def __init__(self,
                 input_format='video',
                 filename_tmpl='img_{:05}.jpg',
                 modality='RGB',
                 start_index=1) -> None:
        self.input_format = input_format
        self.filename_tmpl = filename_tmpl
        self.modality = modality
        self.start_index = start_index

    def transform(self, video: Union[str, np.ndarray, dict]) -> dict:
        if self.input_format == 'dict':
            results = video
        elif self.input_format == 'video':
            results = dict(
                filename=video, label=-1, start_index=0, modality='RGB')
        elif self.input_format == 'rawframes':
            import re

            # count the number of frames that match the format of
            # `filename_tmpl`
            # RGB pattern example: img_{:05}.jpg -> ^img_\d+.jpg$
            # Flow patteren example: {}_{:05d}.jpg -> ^x_\d+.jpg$
            pattern = f'^{self.filename_tmpl}$'
            if self.modality == 'Flow':
                pattern = pattern.replace('{}', 'x')
            pattern = pattern.replace(
                pattern[pattern.find('{'):pattern.find('}') + 1], '\\d+')
            total_frames = len(
                list(
                    filter(lambda x: re.match(pattern, x) is not None,
                           os.listdir(video))))
            results = dict(
                frame_dir=video,
                total_frames=total_frames,
                label=-1,
                start_index=self.start_index,
                filename_tmpl=self.filename_tmpl,
                modality=self.modality)
        elif self.input_format == 'array':
            modality_map = {2: 'Flow', 3: 'RGB'}
            modality = modality_map.get(video.shape[-1])
            results = dict(
                total_frames=video.shape[0],
                label=-1,
                start_index=0,
                array=video,
                modality=modality)

        return results


@TRANSFORMS.register_module()
class ArrayDecode(BaseTransform):
    """Load and decode frames with given indices from a 4D array.

    Required keys are "array and "frame_inds", added or modified keys are
    "imgs", "img_shape" and "original_shape".
    """

    def transform(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        modality = results['modality']
        array = results['array']

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        for i, frame_idx in enumerate(results['frame_inds']):

            frame_idx += offset
            if modality == 'RGB':
                imgs.append(array[frame_idx])
            elif modality == 'Flow':
                imgs.extend(
                    [array[frame_idx, ..., 0], array[frame_idx, ..., 1]])
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@TRANSFORMS.register_module()
class ImageDecode(BaseTransform):
    """Load and decode images.

    Required key is "filename", added or modified keys are "imgs", "img_shape"
    and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results):
        """Perform the ``ImageDecode`` to load image given the file path.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        filename = results['filename']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()
        img_bytes = self.file_client.get(filename)

        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        imgs.append(img)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadAudioFeature(BaseTransform):
    """Load offline extracted audio features.

    Required Keys:

        - audio_path

    Added Keys:

        - length
        - audios

    Args:
        pad_method (str): Padding method. Defaults to ``'zero'``.
    """

    def __init__(self, pad_method: str = 'zero') -> None:
        if pad_method not in ['zero', 'random']:
            raise NotImplementedError
        self.pad_method = pad_method

    @staticmethod
    def _zero_pad(shape: int) -> np.ndarray:
        """Zero padding method."""
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def _random_pad(shape: int) -> np.ndarray:
        """Random padding method."""
        # spectrogram is normalized into a distribution of 0~1
        return np.random.rand(shape).astype(np.float32)

    def transform(self, results: Dict) -> Dict:
        """Perform the numpy loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if osp.exists(results['audio_path']):
            feature_map = np.load(results['audio_path'])
        else:
            # Generate a random dummy 10s input
            # Some videos do not have audio stream
            pad_func = getattr(self, f'_{self.pad_method}_pad')
            feature_map = pad_func((640, 80))

        results['length'] = feature_map.shape[0]
        results['audios'] = feature_map
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'pad_method={self.pad_method})')
        return repr_str


@TRANSFORMS.register_module()
class BuildPseudoClip(BaseTransform):
    """Build pseudo clips with one single image by repeating it n times.

    Required key is "imgs", added or modified key is "imgs", "num_clips",
        "clip_len".

    Args:
        clip_len (int): Frames of the generated pseudo clips.
    """

    def __init__(self, clip_len):
        self.clip_len = clip_len

    def transform(self, results):
        """Perform the building of pseudo clips.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # the input should be one single image
        assert len(results['imgs']) == 1
        im = results['imgs'][0]
        for _ in range(1, self.clip_len):
            results['imgs'].append(np.copy(im))
        results['clip_len'] = self.clip_len
        results['num_clips'] = 1
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'fix_length={self.fixed_length})')
        return repr_str


@TRANSFORMS.register_module()
class AudioFeatureSelector(BaseTransform):
    """Sample the audio feature w.r.t. the frames selected.

    Required Keys:

        - audios
        - frame_inds
        - num_clips
        - length
        - total_frames

    Modified Keys:

        - audios

    Added Keys:

        - audios_shape

    Args:
        fixed_length (int): As the features selected by frames sampled may
            not be exactly the same, `fixed_length` will truncate or pad them
            into the same size. Defaults to 128.
    """

    def __init__(self, fixed_length: int = 128) -> None:
        self.fixed_length = fixed_length

    def transform(self, results: Dict) -> Dict:
        """Perform the ``AudioFeatureSelector`` to pick audio feature clips.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        audio = results['audios']
        frame_inds = results['frame_inds']
        num_clips = results['num_clips']
        resampled_clips = list()

        frame_inds = frame_inds.reshape(num_clips, -1)
        for clip_idx in range(num_clips):
            clip_frame_inds = frame_inds[clip_idx]
            start_idx = max(
                0,
                int(
                    round((clip_frame_inds[0] + 1) / results['total_frames'] *
                          results['length'])))
            end_idx = min(
                results['length'],
                int(
                    round((clip_frame_inds[-1] + 1) / results['total_frames'] *
                          results['length'])))
            cropped_audio = audio[start_idx:end_idx, :]
            if cropped_audio.shape[0] >= self.fixed_length:
                truncated_audio = cropped_audio[:self.fixed_length, :]
            else:
                truncated_audio = np.pad(
                    cropped_audio,
                    ((0, self.fixed_length - cropped_audio.shape[0]), (0, 0)),
                    mode='constant')

            resampled_clips.append(truncated_audio)
        results['audios'] = np.array(resampled_clips)
        results['audios_shape'] = results['audios'].shape
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'fix_length={self.fixed_length})')
        return repr_str


@TRANSFORMS.register_module()
class LoadLocalizationFeature(BaseTransform):
    """Load Video features for localizer with given video_name list.

    The required key is "feature_path", added or modified keys
    are "raw_feature".

    Args:
        raw_feature_ext (str): Raw feature file extension.  Default: '.csv'.
    """

    def transform(self, results):
        """Perform the LoadLocalizationFeature loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data_path = results['feature_path']
        raw_feature = np.loadtxt(
            data_path, dtype=np.float32, delimiter=',', skiprows=1)

        results['raw_feature'] = np.transpose(raw_feature, (1, 0))

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}'
        return repr_str


@TRANSFORMS.register_module()
class GenerateLocalizationLabels(BaseTransform):
    """Load video label for localizer with given video_name list.

    Required keys are "duration_frame", "duration_second", "feature_frame",
    "annotations", added or modified keys are "gt_bbox".
    """

    def transform(self, results):
        """Perform the GenerateLocalizationLabels loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_frame = results['duration_frame']
        video_second = results['duration_second']
        feature_frame = results['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second
        annotations = results['annotations']

        gt_bbox = []

        for annotation in annotations:
            current_start = max(
                min(1, annotation['segment'][0] / corrected_second), 0)
            current_end = max(
                min(1, annotation['segment'][1] / corrected_second), 0)
            gt_bbox.append([current_start, current_end])

        gt_bbox = np.array(gt_bbox)
        results['gt_bbox'] = gt_bbox
        return results


@TRANSFORMS.register_module()
class LoadProposals(BaseTransform):
    """Loading proposals with given proposal results.

    Required keys are "video_name", added or modified keys are 'bsp_feature',
    'tmin', 'tmax', 'tmin_score', 'tmax_score' and 'reference_temporal_iou'.

    Args:
        top_k (int): The top k proposals to be loaded.
        pgm_proposals_dir (str): Directory to load proposals.
        pgm_features_dir (str): Directory to load proposal features.
        proposal_ext (str): Proposal file extension. Default: '.csv'.
        feature_ext (str): Feature file extension. Default: '.npy'.
    """

    def __init__(self,
                 top_k,
                 pgm_proposals_dir,
                 pgm_features_dir,
                 proposal_ext='.csv',
                 feature_ext='.npy'):
        self.top_k = top_k
        self.pgm_proposals_dir = pgm_proposals_dir
        self.pgm_features_dir = pgm_features_dir
        valid_proposal_ext = ('.csv', )
        if proposal_ext not in valid_proposal_ext:
            raise NotImplementedError
        self.proposal_ext = proposal_ext
        valid_feature_ext = ('.npy', )
        if feature_ext not in valid_feature_ext:
            raise NotImplementedError
        self.feature_ext = feature_ext

    def transform(self, results):
        """Perform the LoadProposals loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_name = results['video_name']
        proposal_path = osp.join(self.pgm_proposals_dir,
                                 video_name + self.proposal_ext)
        if self.proposal_ext == '.csv':
            pgm_proposals = np.loadtxt(
                proposal_path, dtype=np.float32, delimiter=',', skiprows=1)

        pgm_proposals = np.array(pgm_proposals[:self.top_k])
        tmin = pgm_proposals[:, 0]
        tmax = pgm_proposals[:, 1]
        tmin_score = pgm_proposals[:, 2]
        tmax_score = pgm_proposals[:, 3]
        reference_temporal_iou = pgm_proposals[:, 5]

        feature_path = osp.join(self.pgm_features_dir,
                                video_name + self.feature_ext)
        if self.feature_ext == '.npy':
            bsp_feature = np.load(feature_path).astype(np.float32)

        bsp_feature = bsp_feature[:self.top_k, :]
        results['bsp_feature'] = bsp_feature
        results['tmin'] = tmin
        results['tmax'] = tmax
        results['tmin_score'] = tmin_score
        results['tmax_score'] = tmax_score
        results['reference_temporal_iou'] = reference_temporal_iou

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'top_k={self.top_k}, '
                    f'pgm_proposals_dir={self.pgm_proposals_dir}, '
                    f'pgm_features_dir={self.pgm_features_dir}, '
                    f'proposal_ext={self.proposal_ext}, '
                    f'feature_ext={self.feature_ext})')
        return repr_str
