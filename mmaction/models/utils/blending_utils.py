# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.beta import Beta

from mmaction.registry import MODELS
from mmaction.utils import SampleList

__all__ = ['BaseMiniBatchBlending', 'MixupBlending', 'CutmixBlending']


class BaseMiniBatchBlending(metaclass=ABCMeta):
    """Base class for Image Aliasing.

    Args:
        num_classes (int): Number of classes.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    @abstractmethod
    def do_blending(self, imgs: Tensor, label: Tensor, **kwargs) -> tuple:
        """Blending images process."""
        raise NotImplementedError

    def __call__(self, imgs: Tensor, batch_data_samples: SampleList,
                 **kwargs) -> tuple:
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
            imgs (Tensor): Model input images, float tensor with the
                shape of (B, N, C, H, W) or (B, N, C, T, H, W).
            batch_data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_labels`.

        Returns:
            mixed_imgs (Tensor): Blending images, float tensor with the
                same shape of the input imgs.
            batch_data_samples (List[:obj:`ActionDataSample`]): The modified
                batch data samples. ``gt_labels`` in each data sample are
                converted from a hard label to a blended soft label, float
                tensor with the shape of (1, num_classes) and all elements are
                in range [0, 1].
        """
        label = [x.gt_labels.item for x in batch_data_samples]
        label = torch.tensor(label, dtype=torch.long).to(imgs.device)

        one_hot_label = F.one_hot(label, num_classes=self.num_classes)

        mixed_imgs, mixed_label = self.do_blending(imgs, one_hot_label,
                                                   **kwargs)

        for label_item, sample in zip(mixed_label, batch_data_samples):
            sample.gt_labels.item = label_item

        return mixed_imgs, batch_data_samples


@MODELS.register_module()
class MixupBlending(BaseMiniBatchBlending):
    """Implementing Mixup in a mini-batch.

    This module is proposed in `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`_.
    Code Reference https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/utils/mixup.py # noqa

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes: int, alpha: float = .2) -> None:
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    def do_blending(self, imgs: Tensor, label: Tensor, **kwargs) -> tuple:
        """Blending images with mixup.

        Args:
            imgs (Tensor): Model input images, float tensor with the
                shape of (B, N, C, H, W) or (B, N, C, T, H, W).
            label (Tensor): One hot labels, integer tensor with the shape
                of (B, num_classes).

        Returns:
            tuple: A tuple of blended images and labels.
        """
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label = lam * label + (1 - lam) * label[rand_index, :]

        return mixed_imgs, mixed_label


@MODELS.register_module()
class CutmixBlending(BaseMiniBatchBlending):
    """Implementing Cutmix in a mini-batch.

    This module is proposed in `CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features <https://arxiv.org/abs/1905.04899>`_.
    Code Reference https://github.com/clovaai/CutMix-PyTorch

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes: int, alpha: float = .2) -> None:
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    @staticmethod
    def rand_bbox(img_size: torch.Size, lam: Tensor) -> tuple:
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

    def do_blending(self, imgs: Tensor, label: Tensor, **kwargs) -> tuple:
        """Blending images with cutmix.

        Args:
            imgs (Tensor): Model input images, float tensor with the
                shape of (B, N, C, H, W) or (B, N, C, T, H, W).
            label (Tensor): One hot labels, integer tensor with the shape
                of (B, num_classes).

        Returns:
            tuple: A tuple of blended images and labels.
        """

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
