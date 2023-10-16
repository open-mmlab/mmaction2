# Adapted from https://github.com/MCG-NJU/BasicTAD/
# https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection

import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmaction.registry import MODELS
from mmdet.models.dense_heads import RetinaHead
import warnings


@MODELS.register_module()
class RetinaHead1D(RetinaHead):
    r"""Modified RetinaHead to support 1D
    """

    def _init_layers(self):
        super()._init_layers()
        self.retina_cls = nn.Conv1d(  # ---------------
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        reg_dim = self.bbox_coder.encode_size // 2
        self.retina_reg = nn.Conv1d(  # ---------------
            self.feat_channels, self.num_base_priors * reg_dim, 3, padding=1)

    def forward_single(self, x):
        cls_score, bbox_pred = super().forward_single(x)
        # add pseudo H dimension
        cls_score, bbox_pred = cls_score.unsqueeze(-2), bbox_pred.unsqueeze(-2)
        # bbox_pred = [N, 2], where 2 is the x, w. Now adding pseudo y, h
        bbox_pred = bbox_pred.unflatten(1, (self.num_base_priors, -1))
        y, h = torch.split(torch.zeros_like(bbox_pred), 1, dim=2)
        bbox_pred = torch.cat((bbox_pred[:, :, :1, :, :], y, bbox_pred[:, :, 1:, :, :], h), dim=2)
        bbox_pred = bbox_pred.flatten(start_dim=1, end_dim=2)
        return cls_score, bbox_pred

    def predict_by_feat(self, *args, **kwargs):
        # As we predict sliding windows of untrimmed videos, we do not perform NMS inside each window but
        # leave the NMS performed globally on the whole video.
        if kwargs.get('with_nms', False):
            warnings.warn("with_nms is True, which is unexpected as we should perform NMS in Metric rather than in model")
        else:
            kwargs['with_nms'] = False
        return super().predict_by_feat(*args, **kwargs)

#     def get_anchors(self,
#                     featmap_sizes: List[tuple],
#                     batch_img_metas: List[dict],
#                     device: Union[torch.device, str] = 'cuda') \
#             -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
#         num_imgs = len(batch_img_metas)
#
#         # since feature map sizes of all images are the same, we only compute
#         # anchors for one time
#         multi_level_anchors = self.prior_generator.grid_priors(
#             featmap_sizes, device=device)
#         anchor_list = [multi_level_anchors for _ in range(num_imgs)]
#
#         # for each image, we compute valid flags of multi level anchors
#         valid_flag_list = []
#         for img_id, img_meta in enumerate(batch_img_metas):
#             multi_level_flags = self.prior_generator.valid_flags(
#                 featmap_sizes, img_meta['pad_tsize'], device)  # ---------------------------
#             valid_flag_list.append(multi_level_flags)
#
#         return anchor_list, valid_flag_list
#
#     def _get_targets_single(self,
#                             flat_anchors: Union[Tensor, BaseBoxes],
#                             valid_flags: Tensor,
#                             gt_instances: InstanceData,
#                             img_meta: dict,
#                             gt_instances_ignore: Optional[InstanceData] = None,
#                             unmap_outputs: bool = True) -> tuple:
#         inside_flags = anchor_inside_flags(flat_anchors, valid_flags,  # ---------------------
#                                            img_meta['tsize'],  # ---------------------
#                                            self.train_cfg['allowed_border'])
#         if not inside_flags.any():
#             raise ValueError(
#                 'There is no valid anchor inside the image boundary. Please '
#                 'check the image size and anchor sizes, or set '
#                 '``allowed_border`` to -1 to skip the condition.')
#         # assign gt and sample anchors
#         anchors = flat_anchors[inside_flags]
#
#         pred_instances = InstanceData(priors=anchors)
#         assign_result = self.assigner.assign(pred_instances, gt_instances,
#                                              gt_instances_ignore)
#         # No sampling is required except for RPN and
#         # Guided Anchoring algorithms
#         sampling_result = self.sampler.sample(assign_result, pred_instances,
#                                               gt_instances)
#
#         num_valid_anchors = anchors.shape[0]
#         target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox \
#             else self.bbox_coder.encode_size
#         bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
#         bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)
#
#         # TODO: Considering saving memory, is it necessary to be long?
#         labels = anchors.new_full((num_valid_anchors,),
#                                   self.num_classes,
#                                   dtype=torch.long)
#         label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
#
#         pos_inds = sampling_result.pos_inds
#         neg_inds = sampling_result.neg_inds
#         # `bbox_coder.encode` accepts tensor or box type inputs and generates
#         # tensor targets. If regressing decoded boxes, the code will convert
#         # box type `pos_bbox_targets` to tensor.
#         if len(pos_inds) > 0:
#             if not self.reg_decoded_bbox:
#                 pos_bbox_targets = self.bbox_coder.encode(
#                     sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
#             else:
#                 pos_bbox_targets = sampling_result.pos_gt_bboxes
#                 pos_bbox_targets = get_box_tensor(pos_bbox_targets)
#             bbox_targets[pos_inds, :] = pos_bbox_targets
#             bbox_weights[pos_inds, :] = 1.0
#
#             labels[pos_inds] = sampling_result.pos_gt_labels
#             if self.train_cfg['pos_weight'] <= 0:
#                 label_weights[pos_inds] = 1.0
#             else:
#                 label_weights[pos_inds] = self.train_cfg['pos_weight']
#         if len(neg_inds) > 0:
#             label_weights[neg_inds] = 1.0
#
#         # map up to original set of anchors
#         if unmap_outputs:
#             num_total_anchors = flat_anchors.size(0)
#             labels = unmap(
#                 labels, num_total_anchors, inside_flags,
#                 fill=self.num_classes)  # fill bg label
#             label_weights = unmap(label_weights, num_total_anchors,
#                                   inside_flags)
#             bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
#             bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
#
#         return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
#                 neg_inds, sampling_result)
#
#     def loss_by_feat(
#             self,
#             cls_scores: List[Tensor],
#             bbox_preds: List[Tensor],
#             batch_gt_instances: InstanceList,
#             batch_img_metas: List[dict],
#             batch_gt_instances_ignore: OptInstanceList = None) -> dict:
#         featmap_sizes = [featmap.size()[-1] for featmap in cls_scores]  # -----------------
#         assert len(featmap_sizes) == self.prior_generator.num_levels
#
#         device = cls_scores[0].device
#
#         anchor_list, valid_flag_list = self.get_anchors(
#             featmap_sizes, batch_img_metas, device=device)
#         cls_reg_targets = self.get_targets(
#             anchor_list,
#             valid_flag_list,
#             batch_gt_instances,
#             batch_img_metas,
#             batch_gt_instances_ignore=batch_gt_instances_ignore)
#         (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
#          avg_factor) = cls_reg_targets
#
#         # anchor number of multi levels
#         num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
#         # concat all level anchors and flags to a single tensor
#         concat_anchor_list = []
#         for i in range(len(anchor_list)):
#             concat_anchor_list.append(cat_boxes(anchor_list[i]))
#         all_anchor_list = images_to_levels(concat_anchor_list,
#                                            num_level_anchors)
#
#         losses_cls, losses_bbox = multi_apply(
#             self.loss_by_feat_single,
#             cls_scores,
#             bbox_preds,
#             all_anchor_list,
#             labels_list,
#             label_weights_list,
#             bbox_targets_list,
#             bbox_weights_list,
#             avg_factor=avg_factor)
#         return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
#
#     def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
#                             anchors: Tensor, labels: Tensor,
#                             label_weights: Tensor, bbox_targets: Tensor,
#                             bbox_weights: Tensor, avg_factor: int) -> tuple:
#         # classification loss
#         labels = labels.reshape(-1)
#         label_weights = label_weights.reshape(-1)
#         cls_score = cls_score.permute(0, 2, 1).reshape(-1, self.cls_out_channels)  # ---------------
#         loss_cls = self.loss_cls(
#             cls_score, labels, label_weights, avg_factor=avg_factor)
#         # regression loss
#         target_dim = bbox_targets.size(-1)
#         bbox_targets = bbox_targets.reshape(-1, target_dim)
#         bbox_weights = bbox_weights.reshape(-1, target_dim)
#         bbox_pred = bbox_pred.permute(0, 2, 1).reshape(-1, self.bbox_coder.encode_size)  # ---------------
#         if self.reg_decoded_bbox:
#             # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
#             # is applied directly on the decoded bounding boxes, it
#             # decodes the already encoded coordinates to absolute format.
#             anchors = anchors.reshape(-1, anchors.size(-1))
#             d_bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
#             d_bbox_pred = get_box_tensor(d_bbox_pred)
#         loss_bbox = self.loss_bbox(
#             d_bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
#         return loss_cls, loss_bbox
#
#     def predict_by_feat(self,
#                         cls_scores: List[Tensor],
#                         bbox_preds: List[Tensor],
#                         score_factors: Optional[List[Tensor]] = None,
#                         batch_img_metas: Optional[List[dict]] = None,
#                         cfg: Optional[ConfigDict] = None,
#                         rescale: bool = True,  # ------------------------------
#                         with_nms: bool = False) -> InstanceList:  # -----------------------
#         assert len(cls_scores) == len(bbox_preds)
#
#         if score_factors is None:
#             # e.g. Retina, FreeAnchor, Foveabox, etc.
#             with_score_factors = False
#         else:
#             # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
#             with_score_factors = True
#             assert len(cls_scores) == len(score_factors)
#
#         num_levels = len(cls_scores)
#
#         featmap_sizes = [cls_scores[i].shape[-1] for i in range(num_levels)]  # ------------------
#         mlvl_priors = self.prior_generator.grid_priors(
#             featmap_sizes,
#             dtype=cls_scores[0].dtype,
#             device=cls_scores[0].device)
#
#         result_list = []
#
#         for img_id in range(len(batch_img_metas)):
#             img_meta = batch_img_metas[img_id]
#             cls_score_list = select_single_mlvl(
#                 cls_scores, img_id, detach=True)
#             bbox_pred_list = select_single_mlvl(
#                 bbox_preds, img_id, detach=True)
#             if with_score_factors:
#                 score_factor_list = select_single_mlvl(
#                     score_factors, img_id, detach=True)
#             else:
#                 score_factor_list = [None for _ in range(num_levels)]
#
#             results = self._predict_by_feat_single(
#                 cls_score_list=cls_score_list,
#                 bbox_pred_list=bbox_pred_list,
#                 score_factor_list=score_factor_list,
#                 mlvl_priors=mlvl_priors,
#                 img_meta=img_meta,
#                 cfg=cfg,
#                 rescale=rescale,
#                 with_nms=with_nms)
#             result_list.append(results)
#         return result_list
#
#     def _predict_by_feat_single(self,
#                                 cls_score_list: List[Tensor],
#                                 bbox_pred_list: List[Tensor],
#                                 score_factor_list: List[Tensor],
#                                 mlvl_priors: List[Tensor],
#                                 img_meta: dict,
#                                 cfg: ConfigDict,
#                                 rescale: bool = True,  # ------------------
#                                 with_nms: bool = False) -> InstanceData:  # --------------------------
#         if score_factor_list[0] is None:
#             # e.g. Retina, FreeAnchor, etc.
#             with_score_factors = False
#         else:
#             # e.g. FCOS, PAA, ATSS, etc.
#             with_score_factors = True
#
#         cfg = self.test_cfg if cfg is None else cfg
#         cfg = copy.deepcopy(cfg)
#         tsize = img_meta['tsize']  # ---------------
#         nms_pre = cfg.get('nms_pre', -1)
#
#         mlvl_bbox_preds = []
#         mlvl_valid_priors = []
#         mlvl_scores = []
#         mlvl_labels = []
#         if with_score_factors:
#             mlvl_score_factors = []
#         else:
#             mlvl_score_factors = None
#         for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
#                 enumerate(zip(cls_score_list, bbox_pred_list,
#                               score_factor_list, mlvl_priors)):
#
#             assert cls_score.size()[-1] == bbox_pred.size()[-1]  # --------------
#
#             dim = self.bbox_coder.encode_size
#             bbox_pred = bbox_pred.permute(1, 0).reshape(-1, dim)  # --------------
#             if with_score_factors:
#                 score_factor = score_factor.permute(1, 0).reshape(-1).sigmoid()  # --------------
#             cls_score = cls_score.permute(1, 0).reshape(-1, self.cls_out_channels)  # ----------------
#             if self.use_sigmoid_cls:
#                 scores = cls_score.sigmoid()
#             else:
#                 # remind that we set FG labels to [0, num_class-1]
#                 # since mmdet v2.0
#                 # BG cat_id: num_class
#                 scores = cls_score.softmax(-1)[:, :-1]
#
#             # After https://github.com/open-mmlab/mmdetection/pull/6268/,
#             # this operation keeps fewer bboxes under the same `nms_pre`.
#             # There is no difference in performance for most models. If you
#             # find a slight drop in performance, you can set a larger
#             # `nms_pre` than before.
#             score_thr = cfg.get('score_thr', 0)
#
#             results = filter_scores_and_topk(
#                 scores, score_thr, nms_pre,
#                 dict(bbox_pred=bbox_pred, priors=priors))
#             scores, labels, keep_idxs, filtered_results = results
#
#             bbox_pred = filtered_results['bbox_pred']
#             priors = filtered_results['priors']
#
#             if with_score_factors:
#                 score_factor = score_factor[keep_idxs]
#
#             mlvl_bbox_preds.append(bbox_pred)
#             mlvl_valid_priors.append(priors)
#             mlvl_scores.append(scores)
#             mlvl_labels.append(labels)
#
#             if with_score_factors:
#                 mlvl_score_factors.append(score_factor)
#
#         bbox_pred = torch.cat(mlvl_bbox_preds)
#         priors = cat_boxes(mlvl_valid_priors)
#         bboxes = self.bbox_coder.decode(priors, bbox_pred, max_t=tsize)
#
#         results = InstanceData()
#         results.bboxes = bboxes
#         results.scores = torch.cat(mlvl_scores)
#         results.labels = torch.cat(mlvl_labels)
#         if with_score_factors:
#             results.score_factors = torch.cat(mlvl_score_factors)
#         results = self._bbox_post_process(
#             results=results,
#             cfg=cfg,
#             rescale=rescale,
#             with_nms=with_nms,
#             img_meta=img_meta)
#         return results
#
#     def _bbox_post_process(self,
#                            results: InstanceData,
#                            cfg: ConfigDict,
#                            rescale: bool = True,  # --------------------------
#                            with_nms: bool = False,  # ----------------------------
#                            img_meta: Optional[dict] = None) -> InstanceData:
#         if rescale:
#             assert img_meta.get('tscale_factor') is not None
#             tscale_factor = 1 / img_meta['tscale_factor']  # ------------
#             results.bboxes = scale_boxes(results.bboxes, tscale_factor)
#             # Convert the bboxes co-ordinate from the input video segment to the original video
#             results.bboxes += img_meta.get('tshift', 0)  # ---------------
#
#         if hasattr(results, 'score_factors'):
#             # TODO： Add sqrt operation in order to be consistent with
#             #  the paper.
#             score_factors = results.pop('score_factors')
#             results.scores = results.scores * score_factors
#
#         # filter small size bboxes
#         if cfg.get('min_bbox_size', 0) >= 0:
#             l = get_segment_len(results.bboxes)  # ---------------
#             valid_mask = l > cfg.get('min_bbox_size', 0)  # ---------------
#             if not valid_mask.all():
#                 results = results[valid_mask]
#
#         # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
#         if with_nms and results.bboxes.numel() > 0:
#             bboxes = get_box_tensor(results.bboxes)
#             det_bboxes, keep_idxs = batched_nms1d(bboxes, results.scores,
#                                                   results.labels,
#                                                   cfg.get('nms', dict(type='nms', iou_thr=0.5)))  # ------------------
#             results = results[keep_idxs]
#             # some nms would reweight the score, such as softnms
#             results.scores = det_bboxes[:, -1]
#             results = results[:cfg.get('max_per_video', 100)]  # ------------------
#
#         return results
#
#     def loss_and_predict(
#             self,
#             x: Tuple[Tensor],
#             batch_data_samples: SampleList,
#             with_nms: bool = False,  # ------------------
#             rescale: bool = True) -> Tuple[dict, InstanceList]:  # ------------------
#
#         outputs = unpack_gt_instances(batch_data_samples)
#         (batch_gt_instances, batch_gt_instances_ignore,
#          batch_img_metas) = outputs
#
#         outs = self(x)
#
#         loss_inputs = outs + (batch_gt_instances, batch_img_metas,
#                               batch_gt_instances_ignore)
#         losses = self.loss_by_feat(*loss_inputs)
#
#         predictions = self.predict_by_feat(
#             *outs, with_nms=with_nms, batch_img_metas=batch_img_metas, rescale=rescale)
#         return losses, predictions
#
#
# def get_segment_len(segments: Union[Tensor, BaseBoxes]) -> Tuple[Tensor, Tensor]:
#     """Get the width and height of boxes with type of tensor or box type.
#
#     Args:
#         segments (Tensor or :obj:`BaseBoxes`): boxes with type of tensor
#             or box type.
#
#     Returns:
#         Tuple[Tensor, Tensor]: the width and height of boxes.
#     """
#     if isinstance(segments, BaseBoxes):
#         l = segments.length
#     else:
#         # Tensor boxes will be treated as horizontal boxes by defaults
#         l = segments[:, 1] - segments[:, 0]
#     return l
