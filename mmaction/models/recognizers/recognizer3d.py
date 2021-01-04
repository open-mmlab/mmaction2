import torch

from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.test_batch is not None:
            tot = imgs.shape[0]
            assert num_segs == tot, ('test_batch is only compatible with '
                                     'batch_size == 1')
            ptr = 0
            cls_scores = []
            while ptr < tot:
                batch_imgs = imgs[ptr:ptr + self.test_batch]
                x = self.extract_feat(batch_imgs)
                if hasattr(self, 'neck'):
                    x, _ = self.neck(x)
                cls_score = self.cls_head(x)
                cls_scores.append(cls_score)
                ptr += self.test_batch
            cls_score = torch.cat(cls_scores)
        else:
            x = self.extract_feat(imgs)
            if hasattr(self, 'neck'):
                x, _ = self.neck(x)
            cls_score = self.cls_head(x)

        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if hasattr(self, 'neck'):
            x, _ = self.neck(x)

        outs = (self.cls_head(x), )
        return outs

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)
