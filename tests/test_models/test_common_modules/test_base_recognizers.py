import pytest
import torch
import torch.nn.functional as F

from mmaction.models import BaseRecognizer


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
        pass

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

    # average_clips=None
    test_cfg = dict(average_clips=None)
    recognizer = ExampleRecognizer(None, test_cfg)
    score = recognizer.average_clip(cls_score, num_segs=5)
    assert torch.equal(score, cls_score)

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
