# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

from mmaction.models import BMNLoss


def test_bmn_loss():
    bmn_loss = BMNLoss()

    # test tem_loss
    pred_start = torch.tensor([0.9, 0.1])
    pred_end = torch.tensor([0.1, 0.9])
    gt_start = torch.tensor([1., 0.])
    gt_end = torch.tensor([0., 1.])
    output_tem_loss = bmn_loss.tem_loss(pred_start, pred_end, gt_start, gt_end)

    # test pem_reg_loss
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    pred_bm_reg = torch.tensor([[0.1, 0.99], [0.5, 0.4]])
    gt_iou_map = torch.tensor([[0, 1.], [0, 1.]])
    mask = torch.tensor([[0.1, 0.4], [0.4, 0.1]])
    output_pem_reg_loss = bmn_loss.pem_reg_loss(pred_bm_reg, gt_iou_map, mask)
    assert_array_almost_equal(
        output_pem_reg_loss.numpy(), np.array([0.2140]), decimal=4)

    # test pem_cls_loss
    pred_bm_cls = torch.tensor([[0.1, 0.99], [0.95, 0.2]])
    gt_iou_map = torch.tensor([[0., 1.], [0., 1.]])
    mask = torch.tensor([[0.1, 0.4], [0.4, 0.1]])
    output_pem_cls_loss = bmn_loss.pem_cls_loss(pred_bm_cls, gt_iou_map, mask)
    assert_array_almost_equal(
        output_pem_cls_loss.numpy(), np.array([1.6137]), decimal=4)

    # test bmn_loss
    pred_bm = torch.tensor([[[[0.1, 0.99], [0.5, 0.4]],
                             [[0.1, 0.99], [0.95, 0.2]]]])
    pred_start = torch.tensor([[0.9, 0.1]])
    pred_end = torch.tensor([[0.1, 0.9]])
    gt_iou_map = torch.tensor([[[0., 2.5], [0., 10.]]])
    gt_start = torch.tensor([[1., 0.]])
    gt_end = torch.tensor([[0., 1.]])
    mask = torch.tensor([[0.1, 0.4], [0.4, 0.1]])
    output_loss = bmn_loss(pred_bm, pred_start, pred_end, gt_iou_map, gt_start,
                           gt_end, mask)
    assert_array_almost_equal(
        output_loss[0].numpy(),
        output_tem_loss + 10 * output_pem_reg_loss + output_pem_cls_loss)
    assert_array_almost_equal(output_loss[1].numpy(), output_tem_loss)
    assert_array_almost_equal(output_loss[2].numpy(), output_pem_reg_loss)
    assert_array_almost_equal(output_loss[3].numpy(), output_pem_cls_loss)
