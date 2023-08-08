# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import torch
from mmaction.registry import TASK_UTILS
from ..segments_ops import segment_overlaps


def cast_tensor_type(x, scale=1., dtype=None):
    if dtype == 'fp16':
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


@TASK_UTILS.register_module()
class SegmentOverlaps(object):
    """1D IoU Calculator."""

    def __init__(self, scale=1., dtype=None):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, segments1, segments2, mode='iou', is_aligned=False):
        """Calculate IoU between 1D segments.

        Args:
            segments1 (Tensor): segments have shape (m, 2) in <start, end>
                format, or shape (m, 3) in <start, end, score> format.
            segments2 (Tensor): segments have shape (m, 2) in <start, end>
                format, shape (m, 3) in <start, end, score> format, or be
                empty. If is_aligned is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or iof (intersection
                over foreground).

        Returns:
            ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
        """
        assert segments1.size(-1) in [0, 2, 3]
        assert segments2.size(-1) in [0, 2, 3]
        if segments1.size(-1) == 3:
            segments1 = segments1[..., :2]
        if segments2.size(-1) == 3:
            segments2 = segments2[..., :2]

        if self.dtype == 'fp16':
            # change tensor type to save cpu and cuda memory and keep speed
            segments1 = cast_tensor_type(segments1, self.scale, self.dtype)
            segments2 = cast_tensor_type(segments2, self.scale, self.dtype)
            overlaps = segment_overlaps(segments1, segments2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return segment_overlaps(segments1, segments2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str
