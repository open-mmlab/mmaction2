from abc import ABCMeta, abstractmethod

import torch.nn as nn

from .. import builder


class BaseRecognizer(nn.Module, metaclass=ABCMeta):
    """Base class for recognizers

    All recognizers should subclass it.
    All subclass should overwrite:
        Methods:`forward_train`, supporting to forward when training.
        Methods:`forward_test`, supporting to forward when testing.

    Attributes:
        backbone (dict): backbone modules to extract feature.
        cls_head (dict): classification head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self, backbone, cls_head, train_cfg=None, test_cfg=None):
        super(BaseRecognizer, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.cls_head = builder.build_head(cls_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        self.cls_head.init_weights()

    def extract_feat(self, imgs):
        x = self.backbone(imgs)
        return x

    @abstractmethod
    def forward_train(self, imgs, labels):
        pass

    @abstractmethod
    def forward_test(self, imgs):
        pass

    def forward(self, imgs, label, return_loss=True):
        if return_loss:
            return self.forward_train(imgs, label)
        else:
            return self.forward_test(imgs)
