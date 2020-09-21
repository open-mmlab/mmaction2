from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class RecognizerAudio(BaseRecognizer):
    """Audio recognizer model framework."""

    def forward_train(self, imgs, labels):
        """Defines the computation performed at every call when training."""
        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss = self.cls_head.loss(cls_score, gt_labels)

        return loss

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score)

        return cls_score.cpu().numpy()
