from mmdet.models.task_modules.assigners.iou2d_calculator import BboxOverlaps2D
from mmaction.registry import TASK_UTILS
from mmdet.structures.bbox import get_box_tensor


@TASK_UTILS.register_module()
class BboxOverlaps1D(BboxOverlaps2D):
    """IoU Calculator that ignore the y1 and y2."""

    def __call__(self, bboxes1, bboxes2, *args, **kwargs):
        bboxes1, bboxes2 = get_box_tensor(bboxes1), get_box_tensor(bboxes2)
        bboxes1[:, 1] = 0.1
        bboxes2[:, 1] = 0.1
        bboxes1[:, 3] = 0.9
        bboxes2[:, 3] = 0.9
        return super().__call__(bboxes1, bboxes2, *args, **kwargs)
