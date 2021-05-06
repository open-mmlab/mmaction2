import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if self.with_neck:
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
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
            if self.with_neck:
                feat, _ = self.neck(feat)

        if self.feature_extraction:
            # perform spatio-temporal pooling
            avg_pool = nn.AdaptiveAvgPool3d(1)
            feat = avg_pool(feat)
            # squeeze dimensions
            x = x.reshape((batches, num_segs, -1))
            # temporal average pooling
            x = x.mean(axis=1)
            return x

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)
