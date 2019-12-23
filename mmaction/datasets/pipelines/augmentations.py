import random

import numpy as np
import torch.nn.functional as F


class MultiScaleCrop:
    """Crop an image with a randomly selected scale.

    Randomly select the w and h scales from a list of scales. Scale of 1 means
    the base size, which is the minimal of image weight and height. The scale
    level of w and h is controlled to be smaller than a certain value to
    prevent too large or small aspect ratio.
    Required keys are "imgs", added or modified keys are "imgs" and
    "crop_bbox".

    Attributes:
        input_size (tuple): (w, h) of network input.
        scales (list[float]): Weight and height scales to be selected.
        max_wh_scale_gap (int): Maximum gap of w and h scale levels.
            Default: 1.
        random_crop (bool): If set to True, the cropping bbox will be randomly
            sampled, otherwise it will be sampler from 5 fixed regions:
            "upper left", "upper right", "lower left", "lower right", "center"
            Default: False.
    """

    def __init__(self,
                 input_size,
                 scales,
                 max_wh_scale_gap=1,
                 random_crop=False):
        assert isinstance(scales, list)
        self.input_size = input_size
        self.scales = scales
        self.max_wh_scale_gap = max_wh_scale_gap
        self.random_crop = random_crop

    def __call__(self, results):
        imgs = results['imgs']
        img_h, img_w = imgs.shape[-2:]

        base_size = min(img_h, img_w)

        crop_sizes = [int(base_size * s) for s in self.scales]

        candidate_sizes = []
        for i, h in enumerate(crop_sizes):
            for j, w in enumerate(crop_sizes):
                if abs(i - j) <= self.max_wh_scale_gap:
                    candidate_sizes.append([w, h])

        crop_size = random.choice(candidate_sizes)
        for i in range(2):
            if abs(crop_size[i] - self.input_size[i]) < 3:
                crop_size[i] = self.input_size[i]

        crop_w, crop_h = crop_size

        if self.random_crop:
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)
        else:
            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4
            candidate_offsets = []
            candidate_offsets.append((0, 0))  # upper left
            candidate_offsets.append((4 * w_step, 0))  # upper right
            candidate_offsets.append((0, 4 * h_step))  # lower left
            candidate_offsets.append((4 * w_step, 4 * h_step))  # lower right
            candidate_offsets.append((2 * w_step, 2 * h_step))  # center

            x_offset, y_offset = random.choice(candidate_offsets)

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h],
            dtype=np.int32)
        results['imgs'] = imgs[:, :, y_offset:y_offset + crop_h,
                               x_offset:x_offset + crop_w]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(input_size={}, scales={}, max_wh_scale_gap={}, '
                     'random_crop={})').format(self.input_size, self.scales,
                                               self.max_wh_scale_gap,
                                               self.random_crop)
        return repr_str


class Resize:
    """Resize images to a specific size.

    Required keys are "imgs", added or modified keys are "imgs".

    Attributes:
        size (int or Tuple[int]): Target spatial size (h, w).
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, results):
        results['imgs'] = F.interpolate(
            results['imgs'],
            size=self.size,
            mode=self.interpolation,
            align_corners=False)
        return results
