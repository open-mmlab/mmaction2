import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS as MMDET_HEADS

from mmaction.core.bbox import bbox_target


# TODO: we can add class_weight for 80 AVA classes
@MMDET_HEADS.register_module
class BBoxHeadAVA(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively.

    Args:
        temporal_pool_type (str): The temporal pool type. Choices are 'avg' or
            'max'. Default: 'avg'.
        spatial_pool_type (str): The spatial pool type. Choices are 'avg' or
            'max'. Default: 'max'.
        in_channels (int): The number of input channels. Default: 2048.
        num_classes (int): The number of classes. Default: 81.
        multilabel (bool): Whether used for a multilabel task. Default: True.
            (Only support multilabel == True now).
    """

    def __init__(
            self,
            temporal_pool_type='avg',
            spatial_pool_type='max',
            in_channels=2048,
            # The first class is reserved, to classify bbox as pos / neg
            num_classes=81,
            multilabel=True):

        super(BBoxHeadAVA, self).__init__()
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.multilabel = multilabel

        # Handle AVA first
        assert self.multilabel

        in_channels = self.in_channels
        # Pool by default
        if self.temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if self.spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.debug_imgs = None

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        x = self.temporal_pool(x)
        x = self.spatial_pool(x)

        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        # We do not predict bbox, so return None
        return cls_score, None

    def get_targets(self, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        cls_reg_targets = bbox_target(pos_proposals, neg_proposals,
                                      pos_gt_labels, rcnn_train_cfg)
        return cls_reg_targets

    def recall_prec(self, pred_vec, target_vec):
        """
        Args:
            pred_vec (tensor[N x C]): each element is either 0 or 1
            target_vec (tensor[N x C]): each element is either 0 or 1

        """
        correct = pred_vec & target_vec
        # Seems torch 1.5 has no auto type conversion
        recall = correct.sum(1) / target_vec.sum(1).float()
        prec = correct.sum(1) / (pred_vec.sum(1) + 1e-6)
        return recall.mean(), prec.mean()

    def multilabel_accuracy(self, pred, target, topk=1, thr=0.5):
        if topk is None:
            topk = ()
        elif isinstance(topk, int):
            topk = (topk, )

        pred = pred.sigmoid()
        pred_vec = pred > thr
        # Target is 0 or 1, so using 0.5 as the borderline is OK
        target_vec = target > 0.5
        recall_thr, prec_thr = self.recall_prec(pred_vec, target_vec)

        recalls, precs = [], []
        for k in topk:
            _, pred_label = pred.topk(k, 1, True, True)
            pred_vec = pred.new_full(pred.size(), 0, dtype=torch.bool)

            num_sample = pred.shape[0]
            for i in range(num_sample):
                pred_vec[i, pred_label[i]] = 1
            recall_k, prec_k = self.recall_prec(pred_vec, target_vec)
            recalls.append(recall_k)
            precs.append(prec_k)
        return recall_thr, prec_thr, recalls, precs

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets=None,
             bbox_weights=None,
             reduce=True):

        losses = dict()
        if cls_score is not None:
            # Only use the cls_score
            labels = labels[:, 1:]
            pos_inds = torch.sum(labels, dim=-1) > 0
            cls_score = cls_score[pos_inds, 1:]
            labels = labels[pos_inds]

            bce_loss = F.binary_cross_entropy_with_logits
            losses['loss_action_cls'] = bce_loss(cls_score, labels)
            recall_thr, prec_thr, recall_k, prec_k = self.multilabel_accuracy(
                cls_score, labels, topk=(3, 5), thr=0.5)
            losses['recall@thr=0.5'] = recall_thr
            losses['prec@thr=0.5'] = prec_thr
            losses['recall@top3'] = recall_k[0]
            losses['prec@top3'] = prec_k[0]
            losses['recall@top5'] = recall_k[1]
            losses['prec@top5'] = prec_k[1]
        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       img_shape,
                       flip=False,
                       crop_quadruple=None,
                       cfg=None):

        # might be used by testing w. augmentation
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        assert self.multilabel

        scores = cls_score.sigmoid() if cls_score is not None else None
        bboxes = rois[:, 1:]
        assert bboxes.shape[-1] == 4

        # First reverse the flip
        img_h, img_w = img_shape
        if flip:
            bboxes_ = bboxes.clone()
            bboxes_[:, 0] = img_w - 1 - bboxes[:, 2]
            bboxes_[:, 2] = img_w - 1 - bboxes[:, 0]
            bboxes = bboxes_

        # Then normalize the bbox to [0, 1]
        bboxes[:, 0::2] /= img_w
        bboxes[:, 1::2] /= img_h

        def _bbox_crop_undo(bboxes, crop_quadruple):
            decropped = bboxes.clone()

            if crop_quadruple is not None:
                x1, y1, tw, th = crop_quadruple
                decropped[:, 0::2] = bboxes[..., 0::2] * tw + x1
                decropped[:, 1::2] = bboxes[..., 1::2] * th + y1

            return decropped

        bboxes = _bbox_crop_undo(bboxes, crop_quadruple)
        return bboxes, scores
