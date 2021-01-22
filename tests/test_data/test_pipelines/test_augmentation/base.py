import numpy as np
from numpy.testing import assert_array_almost_equal


def check_crop(origin_imgs, result_imgs, result_bbox, num_crops=1):
    """Check if the result_bbox is in correspond to result_imgs."""

    def check_single_crop(origin_imgs, result_imgs, result_bbox):
        result_img_shape = result_imgs[0].shape[:2]
        crop_w = result_bbox[2] - result_bbox[0]
        crop_h = result_bbox[3] - result_bbox[1]
        crop_shape = (crop_h, crop_w)
        if not crop_shape == result_img_shape:
            return False
        left, top, right, bottom = result_bbox
        return np.array_equal(
            np.array(origin_imgs)[:, top:bottom, left:right, :],
            np.array(result_imgs))

    if result_bbox.ndim == 1:
        return check_single_crop(origin_imgs, result_imgs, result_bbox)
    elif result_bbox.ndim == 2:
        num_batch = len(origin_imgs)
        for i, bbox in enumerate(result_bbox):
            if num_crops == 10:
                if (i // num_batch) % 2 == 0:
                    flag = check_single_crop([origin_imgs[i % num_batch]],
                                             [result_imgs[i]], bbox)
                else:
                    flag = check_single_crop([origin_imgs[i % num_batch]],
                                             [np.flip(result_imgs[i], axis=1)],
                                             bbox)
            else:
                flag = check_single_crop([origin_imgs[i % num_batch]],
                                         [result_imgs[i]], bbox)
            if not flag:
                return False
        return True
    else:
        # bbox has a wrong dimension
        return False


def check_flip(origin_imgs, result_imgs, flip_type):
    """Check if the origin_imgs are flipped correctly into result_imgs in
    different flip_types."""
    n = len(origin_imgs)
    h, w, c = origin_imgs[0].shape
    if flip_type == 'horizontal':
        # yapf: disable
        for i in range(n):
            for j in range(h):
                for k in range(w):
                    for channel in range(c):
                        if result_imgs[i][j, k, channel] != origin_imgs[i][j, w - 1 - k, channel]:  # noqa:E501
                            return False
        # yapf: enable
    else:
        # yapf: disable
        for i in range(n):
            for j in range(h):
                for k in range(w):
                    for channel in range(c):
                        if result_imgs[i][j, k, channel] != origin_imgs[i][h - 1 - j, k, channel]:  # noqa:E501
                            return False
        # yapf: enable
    return True


def check_normalize(origin_imgs, result_imgs, norm_cfg):
    """Check if the origin_imgs are normalized correctly into result_imgs in a
    given norm_cfg."""
    target_imgs = result_imgs.copy()
    target_imgs *= norm_cfg['std']
    target_imgs += norm_cfg['mean']
    if norm_cfg['to_bgr']:
        target_imgs = target_imgs[..., ::-1].copy()
    assert_array_almost_equal(origin_imgs, target_imgs, decimal=4)
