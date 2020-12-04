import numpy as np
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


def generate_demo_inputs(input_shape=(1, 3, 3, 224, 224), model_type='2D'):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 250, 3, 224, 224).
        model_type (str): Model type for data generation, from {'2D', '3D'}.
            Default:'2D'
    """
    if len(input_shape) == 5:
        (N, L, C, H, W) = input_shape
    elif len(input_shape) == 6:
        (N, M, C, L, H, W) = input_shape

    imgs = np.random.random(input_shape)

    if model_type == '2D':
        gt_labels = torch.LongTensor([2] * N)
    elif model_type == '3D':
        gt_labels = torch.LongTensor([2] * M)
    elif model_type == 'audio':
        gt_labels = torch.LongTensor([2] * L)
    else:
        raise ValueError(f'Data type {model_type} is not available')

    inputs = {'imgs': torch.FloatTensor(imgs), 'gt_labels': gt_labels}
    return inputs
