import torch
import torch.nn as nn

from .. import builder
from ..registry import LOCALIZERS
from .base import BaseLocalizer


@LOCALIZERS.register_module()
class SSN(BaseLocalizer):
    """Temporal Action Detection with Structured Segment Networks.

    Args:
        backbone (dict): Config for building backbone.
        cls_head (dict): Config for building classification head.
        in_channels (int): Number of channels for input data.
            Default: 3.
        spatial_type (str): Type of spatial pooling.
            Default: 'avg'.
        dropout_ratio (float): Ratio of dropout.
            Default: 0.5.
        loss_cls (dict): Config for building loss.
            Default: ``dict(type='SSNLoss')``.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
    """

    def __init__(self,
                 backbone,
                 cls_head,
                 in_channels=3,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 loss_cls=dict(type='SSNLoss'),
                 train_cfg=None,
                 test_cfg=None):

        super().__init__(backbone, cls_head, train_cfg, test_cfg)

        self.is_test_prepared = False
        self.in_channels = in_channels

        self.spatial_type = spatial_type
        if self.spatial_type == 'avg':
            self.pool = nn.AvgPool2d((7, 7), stride=1, padding=0)
        elif self.spatial_type == 'max':
            self.pool = nn.MaxPool2d((7, 7), stride=1, padding=0)
        else:
            self.pool = None

        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.loss_cls = builder.build_loss(loss_cls)

    def forward_train(self, imgs, proposal_scale_factor, proposal_type,
                      proposal_labels, reg_targets, **kwargs):
        """Define the computation performed at every call when training."""
        imgs = imgs.reshape((-1, self.in_channels) + imgs.shape[4:])

        x = self.extract_feat(imgs)

        if self.pool:
            x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)

        activity_scores, completeness_scores, bbox_preds = self.cls_head(
            (x, proposal_scale_factor))

        loss = self.loss_cls(activity_scores, completeness_scores, bbox_preds,
                             proposal_type, proposal_labels, reg_targets,
                             self.train_cfg)
        loss_dict = dict(**loss)

        return loss_dict

    def forward_test(self, imgs, relative_proposal_list, scale_factor_list,
                     proposal_tick_list, reg_norm_consts, **kwargs):
        """Define the computation performed at every call when testing."""
        num_crops = imgs.shape[0]
        imgs = imgs.reshape((num_crops, -1, self.in_channels) + imgs.shape[3:])
        num_ticks = imgs.shape[1]

        output = []
        minibatch_size = self.test_cfg.ssn.sampler.batch_size
        for idx in range(0, num_ticks, minibatch_size):
            chunk = imgs[:, idx:idx +
                         minibatch_size, :, :, :].view((-1, ) + imgs.shape[2:])
            x = self.extract_feat(chunk)
            if self.pool:
                x = self.pool(x)
            # Merge crop to save memory.
            x = x.reshape((num_crops, x.size(0) // num_crops, -1)).mean(dim=0)
            output.append(x)
        output = torch.cat(output, dim=0)

        relative_proposal_list = relative_proposal_list.squeeze(0)
        proposal_tick_list = proposal_tick_list.squeeze(0)
        scale_factor_list = scale_factor_list.squeeze(0)
        reg_norm_consts = reg_norm_consts.squeeze(0)

        if not self.is_test_prepared:
            self.is_test_prepared = self.cls_head.prepare_test_fc(
                self.cls_head.consensus.num_multipliers)

        (output, activity_scores, completeness_scores,
         bbox_preds) = self.cls_head(
             (output, proposal_tick_list, scale_factor_list), test_mode=True)

        relative_proposal_list = relative_proposal_list.cpu().numpy()
        activity_scores = activity_scores.cpu().numpy()
        completeness_scores = completeness_scores.cpu().numpy()
        if bbox_preds is not None:
            bbox_preds = bbox_preds.view(-1, self.cls_head.num_classes, 2)
            bbox_preds[:, :, 0] = (
                bbox_preds[:, :, 0] * reg_norm_consts[1, 0] +
                reg_norm_consts[0, 0])
            bbox_preds[:, :, 1] = (
                bbox_preds[:, :, 1] * reg_norm_consts[1, 1] +
                reg_norm_consts[0, 1])
            bbox_preds = bbox_preds.cpu().numpy()

        result = [
            dict(
                relative_proposal_list=relative_proposal_list,
                activity_scores=activity_scores,
                completeness_scores=completeness_scores,
                bbox_preds=bbox_preds)
        ]

        return result
