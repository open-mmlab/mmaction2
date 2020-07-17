from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels):
        """Defines the computation performed at every call when training."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss = self.cls_head.loss(cls_score, gt_labels)

        return loss

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)
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
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        outs = (self.cls_head(x), )
        return outs
