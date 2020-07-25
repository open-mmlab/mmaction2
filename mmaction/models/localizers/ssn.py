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
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
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
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.spatial_type == 'max':
            self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.avg_pool = None

        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.loss_cls = builder.build_loss(loss_cls)

    def forward_train(self, imgs, proposal_scale_factor, proposal_type,
                      proposal_labels, reg_targets, **kwargs):
        """Define the computation performed at every call when training."""
        imgs = imgs.reshape((-1, self.in_channels) + imgs.shape[4:])

        x = self.extract_feat(imgs)

        if self.avg_pool:
            x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)

        activity_score, completeness_score, bbox_pred = self.cls_head(
            (x, proposal_scale_factor))

        loss = self.loss_cls(activity_score, completeness_score, bbox_pred,
                             proposal_type, proposal_labels, reg_targets,
                             self.train_cfg)
        loss_dict = dict(**loss)

        return loss_dict

    def forward_test(self, imgs, relative_proposal_list, scale_factor_list,
                     proposal_tick_list, reg_stats, **kwargs):
        """Define the computation performed at every call when testing."""
        num_crops = imgs.shape[0]
        imgs = imgs.reshape((num_crops, -1, self.in_channels) + imgs.shape[3:])
        num_ticks = imgs.shape[1]

        output = []
        minibatch_size = self.test_cfg.ssn.sampler.batch_size
        for idx in range(0, num_ticks, minibatch_size):
            chunk = imgs[:, idx:idx +
                         minibatch_size, :, :, :].view((-1, ) + imgs.shape[2:])
            x = self.extract_feat(chunk.cuda())
            if self.avg_pool:
                x = self.avg_pool(x)
            # Merge crop to save memory.
            x = x.reshape((num_crops, x.size(0) // num_crops, -1)).mean(dim=0)
            output.append(x)
        output = torch.cat(output, dim=0)

        relative_proposal_list = relative_proposal_list.squeeze(0)
        proposal_tick_list = proposal_tick_list.squeeze(0)
        scale_factor_list = scale_factor_list.squeeze(0)
        reg_stats = reg_stats.squeeze(0)

        if not self.is_test_prepared:
            self.is_test_prepared = self.cls_head.prepare_test_fc(
                self.cls_head.consensus.feat_multiplier)

        (output, activity_scores, completeness_scores,
         bbox_preds) = self.cls_head(
             (output, proposal_tick_list, scale_factor_list), test_mode=True)

        if bbox_preds is not None:
            bbox_preds = bbox_preds.view(-1, self.cls_head.num_classes, 2)
            bbox_preds[:, :, 0] = (
                bbox_preds[:, :, 0] * reg_stats[1, 0] + reg_stats[0, 0])
            bbox_preds[:, :, 1] = (
                bbox_preds[:, :, 1] * reg_stats[1, 1] + reg_stats[0, 1])

        return (relative_proposal_list.cpu().numpy(),
                activity_scores.cpu().numpy(),
                completeness_scores.cpu().numpy(), bbox_preds.cpu().numpy())

    def forward(self, imgs, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(imgs, **kwargs)
        else:
            return self.forward_test(imgs, **kwargs)
