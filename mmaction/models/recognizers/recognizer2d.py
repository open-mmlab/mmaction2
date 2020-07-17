from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, labels):
        """Defines the computation performed at every call when training."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x, num_segs)
        gt_labels = labels.squeeze()
        loss = self.cls_head.loss(cls_score, gt_labels)

        return loss

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x, num_segs)
        cls_score = self.average_clip(cls_score)

        return cls_score.cpu().numpy()

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``mmaction/tools/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        outs = (self.cls_head(x, num_segs), )
        return outs
