# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

from .builder import BLENDINGS

__all__ = ['BaseMiniBatchBlending', 'MixupBlending', 'CutmixBlending']


class BaseMiniBatchBlending(metaclass=ABCMeta):
    """Base class for Image Aliasing."""

    def __init__(self, num_classes):
        self.num_classes = num_classes

    @abstractmethod
    def do_blending(self, imgs, label, **kwargs):
        pass

    def __call__(self, imgs, label, **kwargs):
        """Blending data in a mini-batch.

        Images are float tensors with the shape of (B, N, C, H, W) for 2D
        recognizers or (B, N, C, T, H, W) for 3D recognizers.

        Besides, labels are converted from hard labels to soft labels.
        Hard labels are integer tensors with the shape of (B, 1) and all of the
        elements are in the range [0, num_classes - 1].
        Soft labels (probablity distribution over classes) are float tensors
        with the shape of (B, 1, num_classes) and all of the elements are in
        the range [0, 1].

        Args:
            imgs (torch.Tensor): Model input images, float tensor with the
                shape of (B, N, C, H, W) or (B, N, C, T, H, W).
            label (torch.Tensor): Hard labels, integer tensor with the shape
                of (B, 1) and all elements are in range [0, num_classes).
            kwargs (dict, optional): Other keyword argument to be used to
                blending imgs and labels in a mini-batch.

        Returns:
            mixed_imgs (torch.Tensor): Blending images, float tensor with the
                same shape of the input imgs.
            mixed_label (torch.Tensor): Blended soft labels, float tensor with
                the shape of (B, 1, num_classes) and all elements are in range
                [0, 1].
        """
        one_hot_label = F.one_hot(label, num_classes=self.num_classes)

        mixed_imgs, mixed_label = self.do_blending(imgs, one_hot_label,
                                                   **kwargs)

        return mixed_imgs, mixed_label


@BLENDINGS.register_module()
class MixupBlending(BaseMiniBatchBlending):
    """Implementing Mixup in a mini-batch.

    This module is proposed in `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`_.
    Code Reference https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/utils/mixup.py # noqa

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label = lam * label + (1 - lam) * label[rand_index, :]

        return mixed_imgs, mixed_label


@BLENDINGS.register_module()
class CutmixBlending(BaseMiniBatchBlending):
    """Implementing Cutmix in a mini-batch.

    This module is proposed in `CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features <https://arxiv.org/abs/1905.04899>`_.
    Code Reference https://github.com/clovaai/CutMix-PyTorch

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # uniform
        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.beta.sample()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
        imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[rand_index, ..., bby1:bby2,
                                                  bbx1:bbx2]
        lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))

        label = lam * label + (1 - lam) * label[rand_index, :]

        return imgs, label
