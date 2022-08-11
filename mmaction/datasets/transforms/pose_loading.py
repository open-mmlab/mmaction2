# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import pickle

import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient
from scipy.stats import mode

from mmaction.registry import TRANSFORMS
from .processing import Flip


@TRANSFORMS.register_module()
class UniformSampleFrames(BaseTransform):
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        seed (int): The random seed used during test time. Default: 255.
    """

    def __init__(self, clip_len, num_clips=1, test_mode=False, seed=255):

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.seed = seed

    def _get_train_clips(self, num_frames, clip_len):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """

        assert self.num_clips == 1
        if num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            inds = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            inds = np.random.choice(
                clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int32)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset
        return inds

    def _get_test_clips(self, num_frames, clip_len):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """

        np.random.seed(self.seed)
        if num_frames < clip_len:
            # Then we use a simple strategy
            if num_frames < self.num_clips:
                start_inds = list(range(self.num_clips))
            else:
                start_inds = [
                    i * num_frames // self.num_clips
                    for i in range(self.num_clips)
                ]
            inds = np.concatenate(
                [np.arange(i, i + clip_len) for i in start_inds])
        elif clip_len <= num_frames < clip_len * 2:
            all_inds = []
            for i in range(self.num_clips):
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int32)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
                all_inds.append(inds)
            inds = np.concatenate(all_inds)
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            all_inds = []
            for i in range(self.num_clips):
                offset = np.random.randint(bsize)
                all_inds.append(bst + offset)
            inds = np.concatenate(all_inds)
        return inds

    def transform(self, results):
        num_frames = results['total_frames']

        if self.test_mode:
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results['start_index']
        inds = inds + start_index

        results['frame_inds'] = inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'test_mode={self.test_mode}, '
                    f'seed={self.seed})')
        return repr_str


@TRANSFORMS.register_module()
class PoseDecode(BaseTransform):
    """Load and decode pose with given indices.

    Required keys are "keypoint", "frame_inds" (optional), "keypoint_score"
    (optional), added or modified keys are "keypoint", "keypoint_score" (if
    applicable).
    """

    def transform(self, results):

        if 'total_frames' not in results:
            results['total_frames'] = results['keypoint'].shape[1]

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            kpscore = results['keypoint_score']
            results['keypoint_score'] = kpscore[:,
                                                frame_inds].astype(np.float32)

        if 'keypoint' in results:
            results['keypoint'] = results['keypoint'][:, frame_inds].astype(
                np.float32)

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@TRANSFORMS.register_module()
class LoadKineticsPose(BaseTransform):
    """Load Kinetics Pose given filename (The format should be pickle)

    Required keys are "filename", "total_frames", "img_shape", "frame_inds",
    "anno_inds" (for mmpose source, optional), added or modified keys are
    "keypoint", "keypoint_score".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        squeeze (bool): Whether to remove frames with no human pose.
            Default: True.
        max_person (int): The max number of persons in a frame. Default: 10.
        keypoint_weight (dict): The weight of keypoints. We set the confidence
            score of a person as the weighted sum of confidence scores of each
            joint. Persons with low confidence scores are dropped (if exceed
            max_person). Default: dict(face=1, torso=2, limb=3).
        source (str): The sources of the keypoints used. Choices are 'mmpose'
            and 'openpose-18'. Default: 'mmpose'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self,
                 io_backend='disk',
                 squeeze=True,
                 max_person=100,
                 keypoint_weight=dict(face=1, torso=2, limb=3),
                 source='mmpose',
                 **kwargs):

        self.io_backend = io_backend
        self.squeeze = squeeze
        self.max_person = max_person
        self.keypoint_weight = cp.deepcopy(keypoint_weight)
        self.source = source

        if source == 'openpose-18':
            self.kpsubset = dict(
                face=[0, 14, 15, 16, 17],
                torso=[1, 2, 8, 5, 11],
                limb=[3, 4, 6, 7, 9, 10, 12, 13])
        elif source == 'mmpose':
            self.kpsubset = dict(
                face=[0, 1, 2, 3, 4],
                torso=[5, 6, 11, 12],
                limb=[7, 8, 9, 10, 13, 14, 15, 16])
        else:
            raise NotImplementedError('Unknown source of Kinetics Pose')

        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results):

        assert 'filename' in results
        filename = results.pop('filename')

        # only applicable to source == 'mmpose'
        anno_inds = None
        if 'anno_inds' in results:
            assert self.source == 'mmpose'
            anno_inds = results.pop('anno_inds')
        results.pop('box_score', None)

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        bytes = self.file_client.get(filename)

        # only the kp array is in the pickle file, each kp include x, y, score.
        kps = pickle.loads(bytes)

        total_frames = results['total_frames']

        frame_inds = results.pop('frame_inds')

        if anno_inds is not None:
            kps = kps[anno_inds]
            frame_inds = frame_inds[anno_inds]

        frame_inds = list(frame_inds)

        def mapinds(inds):
            uni = np.unique(inds)
            map_ = {x: i for i, x in enumerate(uni)}
            inds = [map_[x] for x in inds]
            return np.array(inds, dtype=np.int16)

        if self.squeeze:
            frame_inds = mapinds(frame_inds)
            total_frames = np.max(frame_inds) + 1

        # write it back
        results['total_frames'] = total_frames

        h, w = results['img_shape']
        if self.source == 'openpose-18':
            kps[:, :, 0] *= w
            kps[:, :, 1] *= h

        num_kp = kps.shape[1]
        num_person = mode(frame_inds)[-1][0]

        new_kp = np.zeros([num_person, total_frames, num_kp, 2],
                          dtype=np.float16)
        new_kpscore = np.zeros([num_person, total_frames, num_kp],
                               dtype=np.float16)
        # 32768 is enough
        num_person_frame = np.zeros([total_frames], dtype=np.int16)

        for frame_ind, kp in zip(frame_inds, kps):
            person_ind = num_person_frame[frame_ind]
            new_kp[person_ind, frame_ind] = kp[:, :2]
            new_kpscore[person_ind, frame_ind] = kp[:, 2]
            num_person_frame[frame_ind] += 1

        kpgrp = self.kpsubset
        weight = self.keypoint_weight
        results['num_person'] = num_person

        if num_person > self.max_person:
            for i in range(total_frames):
                np_frame = num_person_frame[i]
                val = new_kpscore[:np_frame, i]

                val = (
                    np.sum(val[:, kpgrp['face']], 1) * weight['face'] +
                    np.sum(val[:, kpgrp['torso']], 1) * weight['torso'] +
                    np.sum(val[:, kpgrp['limb']], 1) * weight['limb'])
                inds = sorted(range(np_frame), key=lambda x: -val[x])
                new_kpscore[:np_frame, i] = new_kpscore[inds, i]
                new_kp[:np_frame, i] = new_kp[inds, i]
            results['num_person'] = self.max_person

        results['keypoint'] = new_kp[:self.max_person]
        results['keypoint_score'] = new_kpscore[:self.max_person]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'squeeze={self.squeeze}, '
                    f'max_person={self.max_person}, '
                    f'keypoint_weight={self.keypoint_weight}, '
                    f'source={self.source}, '
                    f'kwargs={self.kwargs})')
        return repr_str


@TRANSFORMS.register_module()
class GeneratePoseTarget(BaseTransform):
    """Generate pseudo heatmaps based on joint coordinates and confidence.

    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".

    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
    """

    def __init__(self,
                 sigma=0.6,
                 use_score=True,
                 with_kp=True,
                 with_limb=False,
                 skeletons=((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                            (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                            (13, 15), (6, 12), (12, 14), (14, 16), (11, 12)),
                 double=False,
                 left_kp=(1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp=(2, 4, 6, 8, 10, 12, 14, 16)):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double

        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons

    def generate_a_heatmap(self, img_h, img_w, centers, sigma, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            if max_value < self.eps:
                continue

            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y,
                    st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                            patch)

        return heatmap

    def generate_a_limb_heatmap(self, img_h, img_w, starts, ends, sigma,
                                start_values, end_values):
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            starts (np.ndarray): The coordinates of one keypoint in the
                corresponding limbs (of multiple persons).
            ends (np.ndarray): The coordinates of the other keypoint in the
                corresponding limbs (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            start_values (np.ndarray): The max values of one keypoint in the
                corresponding limbs.
            end_values (np.ndarray): The max values of the other keypoint in
                the corresponding limbs.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for start, end, start_value, end_value in zip(starts, ends,
                                                      start_values,
                                                      end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0])**2 + (y - start[1])**2)

            # distance to end keypoints
            d2_end = ((x - end[0])**2 + (y - end[1])**2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

            if d2_ab < 1:
                full_map = self.generate_a_heatmap(img_h, img_w, [start],
                                                   sigma, [start_value])
                heatmap = np.maximum(heatmap, full_map)
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (
                end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
            d2_seg = (
                a_dominate * d2_start + b_dominate * d2_end +
                seg_dominate * d2_line)

            patch = np.exp(-d2_seg / 2. / sigma**2)
            patch = patch * value_coeff

            heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                heatmap[min_y:max_y, min_x:max_x], patch)

        return heatmap

    def generate_heatmap(self, img_h, img_w, kps, sigma, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            kps (np.ndarray): The coordinates of keypoints in this frame.
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The confidence score of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmaps = []
        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                heatmap = self.generate_a_heatmap(img_h, img_w, kps[:, i],
                                                  sigma, max_values[:, i])
                heatmaps.append(heatmap)

        if self.with_limb:
            for limb in self.skeletons:
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                heatmap = self.generate_a_limb_heatmap(img_h, img_w, starts,
                                                       ends, sigma,
                                                       start_values,
                                                       end_values)
                heatmaps.append(heatmap)

        return np.stack(heatmaps, axis=-1)

    def gen_an_aug(self, results):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint']
        kp_shape = all_kps.shape

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results['img_shape']
        num_frame = kp_shape[1]

        imgs = []
        for i in range(num_frame):
            sigma = self.sigma
            kps = all_kps[:, i]
            kpscores = all_kpscores[:, i]

            max_values = np.ones(kpscores.shape, dtype=np.float32)
            if self.use_score:
                max_values = kpscores

            hmap = self.generate_heatmap(img_h, img_w, kps, sigma, max_values)
            imgs.append(hmap)

        return imgs

    def transform(self, results):
        if not self.double:
            results['imgs'] = np.stack(self.gen_an_aug(results))
        else:
            results_ = cp.deepcopy(results)
            flip = Flip(
                flip_ratio=1, left_kp=self.left_kp, right_kp=self.right_kp)
            results_ = flip(results_)
            results['imgs'] = np.concatenate(
                [self.gen_an_aug(results),
                 self.gen_an_aug(results_)])
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'with_kp={self.with_kp}, '
                    f'with_limb={self.with_limb}, '
                    f'skeletons={self.skeletons}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp})')
        return repr_str


@TRANSFORMS.register_module()
class PaddingWithLoop(BaseTransform):
    """Sample frames from the video.

    To sample an n-frame clip from the video, PaddingWithLoop samples
    the frames from zero index, and loop the frames if the length of
    video frames is less than the value of 'clip_len'.

    Required keys are "total_frames", added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
    """

    def __init__(self, clip_len, num_clips=1):

        self.clip_len = clip_len
        self.num_clips = num_clips

    def transform(self, results):
        num_frames = results['total_frames']

        start_index = results['start_index']
        inds = np.arange(start_index, start_index + self.clip_len)
        inds = np.mod(inds, num_frames)

        results['frame_inds'] = inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips})')
        return repr_str
