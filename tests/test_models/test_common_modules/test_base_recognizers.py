import pytest
import torch
import torch.nn.functional as F

from mmaction.models import BaseRecognizer
from ..base import generate_backbone_demo_inputs


class ExampleRecognizer(BaseRecognizer):

    def __init__(self, train_cfg, test_cfg):
        super(BaseRecognizer, self).__init__()
        # reconstruct `__init__()` method in BaseRecognizer to avoid building
        # backbone and head which are useless to ExampleRecognizer,
        # since ExampleRecognizer is only used for model-unrelated methods
        # (like `average_clip`) testing.
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_train(self, imgs, labels):
        return imgs, labels

    def forward_test(self, imgs):
        pass

    def forward_gradcam(self, imgs):
        pass


def test_base_recognizer():
    cls_score = torch.rand(5, 400)
    with pytest.raises(KeyError):
        # "average_clips" must defined in test_cfg keys
        wrong_test_cfg = dict(clip='score')
        recognizer = ExampleRecognizer(None, wrong_test_cfg)
        recognizer.average_clip(cls_score)

    with pytest.raises(ValueError):
        # unsupported average clips type
        wrong_test_cfg = dict(average_clips='softmax')
        recognizer = ExampleRecognizer(None, wrong_test_cfg)
        recognizer.average_clip(cls_score)

    with pytest.raises(ValueError):
        # Label should not be None
        recognizer = ExampleRecognizer(None, None)
        recognizer(torch.tensor(0))

    # average_clips='score'
    test_cfg = dict(average_clips='score')
    recognizer = ExampleRecognizer(None, test_cfg)
    score = recognizer.average_clip(cls_score, num_segs=5)
    assert torch.equal(score, cls_score.mean(dim=0, keepdim=True))

    # average_clips='prob'
    test_cfg = dict(average_clips='prob')
    recognizer = ExampleRecognizer(None, test_cfg)
    score = recognizer.average_clip(cls_score, num_segs=5)
    assert torch.equal(score,
                       F.softmax(cls_score, dim=1).mean(dim=0, keepdim=True))

    # alending
    num_classes = 10
    train_cfg = dict(
        blending=dict(type='MixupBlending', num_classes=num_classes, alpha=.2))
    recognizer = ExampleRecognizer(train_cfg, None)
    demo_inputs = generate_backbone_demo_inputs(input_shape=(4, 4, 3, 32, 32))
    demo_labels = torch.randint(0, num_classes, (4, ))
    imgs, label = recognizer(demo_inputs, demo_labels, return_loss=True)
    assert imgs.size() == demo_inputs.size()
    assert label.size() == torch.Size((4, num_classes))
