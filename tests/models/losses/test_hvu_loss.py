# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmaction.models import HVULoss


def test_hvu_loss():
    pred = torch.tensor([[-1.0525, -0.7085, 0.1819, -0.8011],
                         [0.1555, -1.5550, 0.5586, 1.9746]])
    gt = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 1.]])
    mask = torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]])
    category_mask = torch.tensor([[1., 0.], [0., 1.]])
    categories = ['action', 'scene']
    category_nums = (2, 2)
    category_loss_weights = (1, 1)
    loss_all_nomask_sum = HVULoss(
        categories=categories,
        category_nums=category_nums,
        category_loss_weights=category_loss_weights,
        loss_type='all',
        with_mask=False,
        reduction='sum')
    loss = loss_all_nomask_sum(pred, gt, mask, category_mask)
    loss1 = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    loss1 = torch.sum(loss1, dim=1)
    assert torch.eq(loss['loss_cls'], torch.mean(loss1))

    loss_all_mask = HVULoss(
        categories=categories,
        category_nums=category_nums,
        category_loss_weights=category_loss_weights,
        loss_type='all',
        with_mask=True)
    loss = loss_all_mask(pred, gt, mask, category_mask)
    loss1 = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    loss1 = torch.sum(loss1 * mask, dim=1) / torch.sum(mask, dim=1)
    loss1 = torch.mean(loss1)
    assert torch.eq(loss['loss_cls'], loss1)

    loss_ind_mask = HVULoss(
        categories=categories,
        category_nums=category_nums,
        category_loss_weights=category_loss_weights,
        loss_type='individual',
        with_mask=True)
    loss = loss_ind_mask(pred, gt, mask, category_mask)
    action_loss = F.binary_cross_entropy_with_logits(pred[:1, :2], gt[:1, :2])
    scene_loss = F.binary_cross_entropy_with_logits(pred[1:, 2:], gt[1:, 2:])
    loss1 = (action_loss + scene_loss) / 2
    assert torch.eq(loss['loss_cls'], loss1)

    loss_ind_nomask_sum = HVULoss(
        categories=categories,
        category_nums=category_nums,
        category_loss_weights=category_loss_weights,
        loss_type='individual',
        with_mask=False,
        reduction='sum')
    loss = loss_ind_nomask_sum(pred, gt, mask, category_mask)
    action_loss = F.binary_cross_entropy_with_logits(
        pred[:, :2], gt[:, :2], reduction='none')
    action_loss = torch.sum(action_loss, dim=1)
    action_loss = torch.mean(action_loss)

    scene_loss = F.binary_cross_entropy_with_logits(
        pred[:, 2:], gt[:, 2:], reduction='none')
    scene_loss = torch.sum(scene_loss, dim=1)
    scene_loss = torch.mean(scene_loss)

    loss1 = (action_loss + scene_loss) / 2
    assert torch.eq(loss['loss_cls'], loss1)
