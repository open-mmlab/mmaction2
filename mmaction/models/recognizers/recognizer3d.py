from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework.

    Attributes:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict): Classification head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self, backbone, cls_head, train_cfg=None, test_cfg=None):
        super(Recognizer3D, self).__init__(backbone, cls_head, train_cfg,
                                           test_cfg)

    def forward_train(self, imgs, labels):
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss = self.cls_head.loss(cls_score, gt_labels)

        return loss

    def forward_test(self, imgs):
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)

        return cls_score.cpu().numpy()
