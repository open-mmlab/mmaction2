# adapted from basicTAD
# https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmaction.registry import TASK_UTILS
from torch import Tensor


@TASK_UTILS.register_module()
class Anchor1DGenerator:

    def __init__(self,
                 strides,
                 scales=None,
                 base_sizes=None,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        # check center and center_offset
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                                    f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        # calculate base sizes of anchors
        self.strides = strides
        self.base_sizes = strides if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self) -> List[Tensor]:
        """Generate base anchors.
        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size: Union[int, float],
                                      scales: Tensor,
                                      center: Optional[Tuple[float]] = None) \
            -> Tensor:
        """Generate base anchors of a single level.
        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.
        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        if center is None:
            center = self.center_offset * (base_size - 1)

        intervals = base_size * scales

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [center - 0.5 * intervals, center + 0.5 * intervals]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def grid_priors(self, featmap_tsizes, dtype=torch.float32, device='cuda'):
        """Get points according to feature map sizes.

        Args:
            featmap_tsizes (list[int]): Multi-level feature map temporal sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
        Returns:
            tuple: points of each image.
        """
        assert self.num_levels == len(featmap_tsizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(
                featmap_tsizes[i],
                level_idx=i,
                dtype=dtype,
                device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(self, featmap_tsize, level_idx, dtype,
                                 device):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_tsize (int): Temporal size of the feature maps.
            stride (int, optional): Stride of the feature map.
                Defaults to .
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        stride = self.strides[level_idx]

        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        shifts = torch.arange(0, featmap_tsize, device=device) * stride
        shifts = shifts.type_as(base_anchors)
        # add A anchors (1, A, 2) to K shifts (K, 1, 1) to get
        # shifted anchors (K, A, 2), reshape to (K*A, 2)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, None]
        all_anchors = all_anchors.view(-1, 2)
        # first A rows correspond to A anchors of 0 in feature map,
        # then 1, 2, ...
        return all_anchors

    def valid_flags(self, featmap_tsizes, pad_tsize, device='cuda'):
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_tsizes (list(tuple)): List of feature map temporal sizes in
                multiple feature levels.
            pad_tsize (int): The padded temporal size of the video.
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_tsizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_tsize = featmap_tsizes[i]
            valid_feat_tsize = min(
                int(np.ceil(pad_tsize / anchor_stride)), feat_tsize)
            flags = self._single_level_valid_flags(
                feat_tsize,
                valid_feat_tsize,
                self.num_base_anchors[i],
                device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def _single_level_valid_flags(self,
                                  featmap_tsize,
                                  valid_tsize,
                                  num_base_anchors,
                                  device='cuda'):
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_tsize (int): The temporal size of feature maps.
            valid_tsize (int): The valid temporal size of the feature
                maps.
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level
                feature map.
        """
        assert valid_tsize <= featmap_tsize
        valid = torch.zeros(featmap_tsize, dtype=torch.bool, device=device)
        valid[:valid_tsize] = 1
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_anchors(self) -> List[int]:
        """list[int]: total number of base anchors in a feature grid"""
        return self.num_base_priors

    @property
    def num_base_priors(self) -> List[int]:
        """list[int]: The number of priors (anchors) at a point
        on the feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]
