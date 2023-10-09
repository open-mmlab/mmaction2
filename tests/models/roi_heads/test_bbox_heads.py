# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.models import BBoxHeadAVA


def test_bbox_head_ava():
    """Test loss method, layer construction, attributes and forward function in
    bbox head."""
    with pytest.raises(TypeError):
        # topk must be None, int or tuple[int]
        BBoxHeadAVA(background_class=True, topk=0.1)

    with pytest.raises(AssertionError):
        # topk should be smaller than num_classes
        BBoxHeadAVA(background_class=True, num_classes=5, topk=(3, 5))

    bbox_head = BBoxHeadAVA(
        background_class=True, in_channels=10, num_classes=4, topk=1)
    input = torch.randn([3, 10, 2, 2, 2])
    ret = bbox_head(input)
    assert ret.shape == (3, 4)

    cls_score = torch.tensor(
        [[0.568, -0.162, 0.273, -0.390, 0.447, 0.102, -0.409],
         [2.388, 0.609, 0.369, 1.630, -0.808, -0.212, 0.296],
         [0.252, -0.533, -0.644, -0.591, 0.148, 0.963, -0.525],
         [0.134, -0.311, -0.764, -0.752, 0.656, -1.517, 0.185]])

    # Test topk_to_matrix()
    assert torch.equal(
        BBoxHeadAVA.topk_to_matrix(cls_score[:, 1:], 1),
        torch.tensor([[0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0]],
                     dtype=bool))
    assert torch.equal(
        BBoxHeadAVA.topk_to_matrix(cls_score[:, 1:], 2),
        torch.tensor([[0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]],
                     dtype=bool))
    assert torch.equal(
        BBoxHeadAVA.topk_to_matrix(cls_score[:, 1:], 3),
        torch.tensor([[0, 1, 0, 1, 1, 0], [1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 1]],
                     dtype=bool))
    assert torch.equal(
        BBoxHeadAVA.topk_to_matrix(cls_score[:, 1:], 6),
        torch.ones([4, 6], dtype=bool))

    # Test Multi-Label Loss
    bbox_head = BBoxHeadAVA(
        background_class=True)  # Why is this here? isn't this redundant?
    bbox_head.init_weights()
    bbox_head = BBoxHeadAVA(
        background_class=True,
        temporal_pool_type='max',
        spatial_pool_type='avg')
    bbox_head.init_weights()

    # test without background class
    """
    losses = bbox_head.loss(
        cls_score=cls_score,
        bbox_pred=None,
        rois=None,
        labels=labels,
        label_weights=label_weights)
    assert torch.isclose(losses['loss_action_cls'], torch.tensor(0.7162495))
    assert torch.isclose(losses['recall@thr=0.5'], torch.tensor(0.6666666))
    assert torch.isclose(losses['prec@thr=0.5'], torch.tensor(0.4791665))
    assert torch.isclose(losses['recall@top3'], torch.tensor(0.75))
    assert torch.isclose(losses['prec@top3'], torch.tensor(0.5))
    assert torch.isclose(losses['recall@top5'], torch.tensor(1.0))
    assert torch.isclose(losses['prec@top5'], torch.tensor(0.45))

    # Test Single-Label Loss
    bbox_head = BBoxHeadAVA(multilabel=False)
    losses = bbox_head.loss(
        cls_score=cls_score,
        bbox_pred=None,
        rois=None,
        labels=labels,
        label_weights=label_weights)
    assert torch.isclose(losses['loss_action_cls'], torch.tensor(1.639561))
    assert torch.isclose(losses['recall@thr=0.5'], torch.tensor(0.25))
    assert torch.isclose(losses['prec@thr=0.5'], torch.tensor(0.25))
    assert torch.isclose(losses['recall@top3'], torch.tensor(0.75))
    assert torch.isclose(losses['prec@top3'], torch.tensor(0.5))
    assert torch.isclose(losses['recall@top5'], torch.tensor(1.0))
    assert torch.isclose(losses['prec@top5'], torch.tensor(0.45))

    # Test ROI
    rois = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.5, 0.6, 0.7, 0.8]])
    rois[1::2] *= 380
    rois[2::2] *= 220
    crop_quadruple = np.array([0.1, 0.2, 0.8, 0.7])
    cls_score = torch.tensor([0.995, 0.728])
    img_shape = (320, 480)
    flip = True

    bbox_head = BBoxHeadAVA(multilabel=True)
    bboxes, scores = bbox_head.get_det_bboxes(
        rois=rois,
        cls_score=cls_score,
        img_shape=img_shape,
        flip=flip,
        crop_quadruple=crop_quadruple)
    assert torch.all(
        torch.isclose(
            bboxes,
            torch.tensor([[0.89783341, 0.20043750, 0.89816672, 0.20087500],
                          [0.45499998, 0.69875002, 0.58166665, 0.86499995]])))
    assert torch.all(
        torch.isclose(scores, torch.tensor([0.73007441, 0.67436624])))

    bbox_head = BBoxHeadAVA(multilabel=False)
    bboxes, scores = bbox_head.get_det_bboxes(
        rois=rois,
        cls_score=cls_score,
        img_shape=img_shape,
        flip=flip,
        crop_quadruple=crop_quadruple)
    assert torch.all(
        torch.isclose(
            bboxes,
            torch.tensor([[0.89783341, 0.20043750, 0.89816672, 0.20087500],
                          [0.45499998, 0.69875002, 0.58166665, 0.86499995]])))
    assert torch.all(torch.isclose(scores, torch.tensor([0.56636, 0.43364])))
    """
