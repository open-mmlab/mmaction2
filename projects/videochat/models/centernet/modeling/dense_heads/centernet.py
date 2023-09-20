import torch
from detectron2.config import configurable
from detectron2.layers import cat
from detectron2.modeling.proposal_generator.build import \
    PROPOSAL_GENERATOR_REGISTRY
from detectron2.structures import Boxes, Instances
from detectron2.utils.comm import get_world_size
from torch import nn

from ..debug import debug_test, debug_train
from ..layers.heatmap_focal_loss import (binary_heatmap_focal_loss,
                                         heatmap_focal_loss_jit)
from ..layers.iou_loss import IOULoss
from ..layers.ml_nms import ml_nms
from .centernet_head import CenterNetHead
from .utils import _transpose, reduce_sum

__all__ = ['CenterNet']

INF = 100000000


@PROPOSAL_GENERATOR_REGISTRY.register()
class CenterNet(nn.Module):

    @configurable
    def __init__(
        self,
        # input_shape: Dict[str, ShapeSpec],
        in_channels=256,
        *,
        num_classes=80,
        in_features=('p3', 'p4', 'p5', 'p6', 'p7'),
        strides=(8, 16, 32, 64, 128),
        score_thresh=0.05,
        hm_min_overlap=0.8,
        loc_loss_type='giou',
        min_radius=4,
        hm_focal_alpha=0.25,
        hm_focal_beta=4,
        loss_gamma=2.0,
        reg_weight=2.0,
        not_norm_reg=True,
        with_agn_hm=False,
        only_proposal=False,
        as_proposal=False,
        not_nms=False,
        pos_weight=1.,
        neg_weight=1.,
        sigmoid_clamp=1e-4,
        ignore_high_fp=-1.,
        center_nms=False,
        sizes_of_interest=[[0, 80], [64, 160], [128, 320], [256, 640],
                           [512, 10000000]],
        more_pos=False,
        more_pos_thresh=0.2,
        more_pos_topk=9,
        pre_nms_topk_train=1000,
        pre_nms_topk_test=1000,
        post_nms_topk_train=100,
        post_nms_topk_test=100,
        nms_thresh_train=0.6,
        nms_thresh_test=0.6,
        no_reduce=False,
        debug=False,
        vis_thresh=0.5,
        pixel_mean=[103.530, 116.280, 123.675],
        pixel_std=[1.0, 1.0, 1.0],
        device='cuda',
        centernet_head=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.strides = strides
        self.score_thresh = score_thresh
        self.min_radius = min_radius
        self.hm_focal_alpha = hm_focal_alpha
        self.hm_focal_beta = hm_focal_beta
        self.loss_gamma = loss_gamma
        self.reg_weight = reg_weight
        self.not_norm_reg = not_norm_reg
        self.with_agn_hm = with_agn_hm
        self.only_proposal = only_proposal
        self.as_proposal = as_proposal
        self.not_nms = not_nms
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.sigmoid_clamp = sigmoid_clamp
        self.ignore_high_fp = ignore_high_fp
        self.center_nms = center_nms
        self.sizes_of_interest = sizes_of_interest
        self.more_pos = more_pos
        self.more_pos_thresh = more_pos_thresh
        self.more_pos_topk = more_pos_topk
        self.pre_nms_topk_train = pre_nms_topk_train
        self.pre_nms_topk_test = pre_nms_topk_test
        self.post_nms_topk_train = post_nms_topk_train
        self.post_nms_topk_test = post_nms_topk_test
        self.nms_thresh_train = nms_thresh_train
        self.nms_thresh_test = nms_thresh_test
        self.no_reduce = no_reduce
        self.debug = debug
        self.vis_thresh = vis_thresh
        if self.center_nms:
            self.not_nms = True
        self.iou_loss = IOULoss(loc_loss_type)
        assert (not self.only_proposal) or self.with_agn_hm
        # delta for rendering heatmap
        self.delta = (1 - hm_min_overlap) / (1 + hm_min_overlap)
        if centernet_head is None:
            self.centernet_head = CenterNetHead(
                in_channels=in_channels,
                num_levels=len(in_features),
                with_agn_hm=with_agn_hm,
                only_proposal=only_proposal)
        else:
            self.centernet_head = centernet_head
        if self.debug:
            pixel_mean = torch.Tensor(pixel_mean).to(
                torch.device(device)).view(3, 1, 1)
            pixel_std = torch.Tensor(pixel_std).to(torch.device(device)).view(
                3, 1, 1)
            self.denormalizer = lambda x: x * pixel_std + pixel_mean

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            # 'input_shape': input_shape,
            'in_channels':
            input_shape[cfg.MODEL.CENTERNET.IN_FEATURES[0]].channels,
            'num_classes':
            cfg.MODEL.CENTERNET.NUM_CLASSES,
            'in_features':
            cfg.MODEL.CENTERNET.IN_FEATURES,
            'strides':
            cfg.MODEL.CENTERNET.FPN_STRIDES,
            'score_thresh':
            cfg.MODEL.CENTERNET.INFERENCE_TH,
            'loc_loss_type':
            cfg.MODEL.CENTERNET.LOC_LOSS_TYPE,
            'hm_min_overlap':
            cfg.MODEL.CENTERNET.HM_MIN_OVERLAP,
            'min_radius':
            cfg.MODEL.CENTERNET.MIN_RADIUS,
            'hm_focal_alpha':
            cfg.MODEL.CENTERNET.HM_FOCAL_ALPHA,
            'hm_focal_beta':
            cfg.MODEL.CENTERNET.HM_FOCAL_BETA,
            'loss_gamma':
            cfg.MODEL.CENTERNET.LOSS_GAMMA,
            'reg_weight':
            cfg.MODEL.CENTERNET.REG_WEIGHT,
            'not_norm_reg':
            cfg.MODEL.CENTERNET.NOT_NORM_REG,
            'with_agn_hm':
            cfg.MODEL.CENTERNET.WITH_AGN_HM,
            'only_proposal':
            cfg.MODEL.CENTERNET.ONLY_PROPOSAL,
            'as_proposal':
            cfg.MODEL.CENTERNET.AS_PROPOSAL,
            'not_nms':
            cfg.MODEL.CENTERNET.NOT_NMS,
            'pos_weight':
            cfg.MODEL.CENTERNET.POS_WEIGHT,
            'neg_weight':
            cfg.MODEL.CENTERNET.NEG_WEIGHT,
            'sigmoid_clamp':
            cfg.MODEL.CENTERNET.SIGMOID_CLAMP,
            'ignore_high_fp':
            cfg.MODEL.CENTERNET.IGNORE_HIGH_FP,
            'center_nms':
            cfg.MODEL.CENTERNET.CENTER_NMS,
            'sizes_of_interest':
            cfg.MODEL.CENTERNET.SOI,
            'more_pos':
            cfg.MODEL.CENTERNET.MORE_POS,
            'more_pos_thresh':
            cfg.MODEL.CENTERNET.MORE_POS_THRESH,
            'more_pos_topk':
            cfg.MODEL.CENTERNET.MORE_POS_TOPK,
            'pre_nms_topk_train':
            cfg.MODEL.CENTERNET.PRE_NMS_TOPK_TRAIN,
            'pre_nms_topk_test':
            cfg.MODEL.CENTERNET.PRE_NMS_TOPK_TEST,
            'post_nms_topk_train':
            cfg.MODEL.CENTERNET.POST_NMS_TOPK_TRAIN,
            'post_nms_topk_test':
            cfg.MODEL.CENTERNET.POST_NMS_TOPK_TEST,
            'nms_thresh_train':
            cfg.MODEL.CENTERNET.NMS_TH_TRAIN,
            'nms_thresh_test':
            cfg.MODEL.CENTERNET.NMS_TH_TEST,
            'no_reduce':
            cfg.MODEL.CENTERNET.NO_REDUCE,
            'debug':
            cfg.DEBUG,
            'vis_thresh':
            cfg.VIS_THRESH,
            'pixel_mean':
            cfg.MODEL.PIXEL_MEAN,
            'pixel_std':
            cfg.MODEL.PIXEL_STD,
            'device':
            cfg.MODEL.DEVICE,
            'centernet_head':
            CenterNetHead(
                cfg,
                [input_shape[f] for f in cfg.MODEL.CENTERNET.IN_FEATURES]),
        }
        return ret

    def forward(self, images, features_dict, gt_instances):
        features = [features_dict[f] for f in self.in_features]
        clss_per_level, reg_pred_per_level, agn_hm_pred_per_level = \
            self.centernet_head(features)
        grids = self.compute_grids(features)
        shapes_per_level = grids[0].new_tensor([(x.shape[2], x.shape[3])
                                                for x in reg_pred_per_level])

        if not self.training:
            return self.inference(images, clss_per_level, reg_pred_per_level,
                                  agn_hm_pred_per_level, grids)
        else:
            pos_inds, labels, reg_targets, flattened_hms = \
                self._get_ground_truth(
                    grids, shapes_per_level, gt_instances)
            # logits_pred: M x F, reg_pred: M x 4, agn_hm_pred: M
            logits_pred, reg_pred, agn_hm_pred = self._flatten_outputs(
                clss_per_level, reg_pred_per_level, agn_hm_pred_per_level)

            if self.more_pos:
                # add more pixels as positive if \ 1. they are within the
                # center3x3 region of an object 2. their regression losses
                # are small (<self.more_pos_thresh)
                pos_inds, labels = self._add_more_pos(reg_pred, gt_instances,
                                                      shapes_per_level)

            losses = self.losses(pos_inds, labels, reg_targets, flattened_hms,
                                 logits_pred, reg_pred, agn_hm_pred)

            proposals = None
            if self.only_proposal:
                agn_hm_pred_per_level = [
                    x.sigmoid() for x in agn_hm_pred_per_level
                ]
                proposals = self.predict_instances(
                    grids, agn_hm_pred_per_level, reg_pred_per_level,
                    images.image_sizes, [None for _ in agn_hm_pred_per_level])
            elif self.as_proposal:  # category specific bbox as agnostic
                # proposals
                clss_per_level = [x.sigmoid() for x in clss_per_level]
                proposals = self.predict_instances(grids, clss_per_level,
                                                   reg_pred_per_level,
                                                   images.image_sizes,
                                                   agn_hm_pred_per_level)
            if self.only_proposal or self.as_proposal:
                for p in range(len(proposals)):
                    proposals[p].proposal_boxes = proposals[p].get(
                        'pred_boxes')
                    proposals[p].objectness_logits = proposals[p].get('scores')
                    proposals[p].remove('pred_boxes')
                    proposals[p].remove('scores')
                    proposals[p].remove('pred_classes')

            if self.debug:
                debug_train([self.denormalizer(x) for x in images],
                            gt_instances, flattened_hms, reg_targets, labels,
                            pos_inds, shapes_per_level, grids, self.strides)
            return proposals, losses

    def losses(self, pos_inds, labels, reg_targets, flattened_hms, logits_pred,
               reg_pred, agn_hm_pred):
        """
        Inputs:
            pos_inds: N
            labels: N
            reg_targets: M x 4
            flattened_hms: M x C
            logits_pred: M x C
            reg_pred: M x 4
            agn_hm_pred: M x 1 or None
            N: number of positive locations in all images
            M: number of pixels from all FPN levels
            C: number of classes
        """
        assert (torch.isfinite(reg_pred).all().item())
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        if self.no_reduce:
            total_num_pos = num_pos_local * num_gpus
        else:
            total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local
                                                            ])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        losses = {}
        if not self.only_proposal:
            pos_loss, neg_loss = heatmap_focal_loss_jit(
                logits_pred,
                flattened_hms,
                pos_inds,
                labels,
                alpha=self.hm_focal_alpha,
                beta=self.hm_focal_beta,
                gamma=self.loss_gamma,
                reduction='sum',
                sigmoid_clamp=self.sigmoid_clamp,
                ignore_high_fp=self.ignore_high_fp,
            )
            pos_loss = self.pos_weight * pos_loss / num_pos_avg
            neg_loss = self.neg_weight * neg_loss / num_pos_avg
            losses['loss_centernet_pos'] = pos_loss
            losses['loss_centernet_neg'] = neg_loss

        reg_inds = torch.nonzero(reg_targets.max(dim=1)[0] >= 0).squeeze(1)
        reg_pred = reg_pred[reg_inds]
        reg_targets_pos = reg_targets[reg_inds]
        reg_weight_map = flattened_hms.max(dim=1)[0]
        reg_weight_map = reg_weight_map[reg_inds]
        reg_weight_map = reg_weight_map * 0 + 1 \
            if self.not_norm_reg else reg_weight_map
        if self.no_reduce:
            reg_norm = max(reg_weight_map.sum(), 1)
        else:
            reg_norm = max(
                reduce_sum(reg_weight_map.sum()).item() / num_gpus, 1)

        reg_loss = self.reg_weight * self.iou_loss(
            reg_pred, reg_targets_pos, reg_weight_map,
            reduction='sum') / reg_norm
        losses['loss_centernet_loc'] = reg_loss

        if self.with_agn_hm:
            cat_agn_heatmap = flattened_hms.max(dim=1)[0]  # M
            agn_pos_loss, agn_neg_loss = binary_heatmap_focal_loss(
                agn_hm_pred,
                cat_agn_heatmap,
                pos_inds,
                alpha=self.hm_focal_alpha,
                beta=self.hm_focal_beta,
                gamma=self.loss_gamma,
                sigmoid_clamp=self.sigmoid_clamp,
                ignore_high_fp=self.ignore_high_fp,
            )
            agn_pos_loss = self.pos_weight * agn_pos_loss / num_pos_avg
            agn_neg_loss = self.neg_weight * agn_neg_loss / num_pos_avg
            losses['loss_centernet_agn_pos'] = agn_pos_loss
            losses['loss_centernet_agn_neg'] = agn_neg_loss

        if self.debug:
            print('losses', losses)
            print('total_num_pos', total_num_pos)
        return losses

    def compute_grids(self, features):
        grids = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            shifts_x = torch.arange(
                0,
                w * self.strides[level],
                step=self.strides[level],
                dtype=torch.float32,
                device=feature.device)
            shifts_y = torch.arange(
                0,
                h * self.strides[level],
                step=self.strides[level],
                dtype=torch.float32,
                device=feature.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
                self.strides[level] // 2
            grids.append(grids_per_level)
        return grids

    def _get_ground_truth(self, grids, shapes_per_level, gt_instances):
        """
        Input:
            grids: list of tensors [(hl x wl, 2)]_l
            shapes_per_level: list of tuples L x 2:
            gt_instances: gt instances
        Retuen:
            pos_inds: N
            labels: N
            reg_targets: M x 4
            flattened_hms: M x C or M x 1
            N: number of objects in all images
            M: number of pixels from all FPN levels
        """

        # get positive pixel index
        if not self.more_pos:
            pos_inds, labels = self._get_label_inds(gt_instances,
                                                    shapes_per_level)
        else:
            pos_inds, labels = None, None
        heatmap_channels = self.num_classes
        L = len(grids)
        num_loc_list = [len(loc) for loc in grids]
        strides = torch.cat([
            shapes_per_level.new_ones(num_loc_list[ll]) * self.strides[ll]
            for ll in range(L)
        ]).float()  # M
        reg_size_ranges = torch.cat([
            shapes_per_level.new_tensor(
                self.sizes_of_interest[ll]).float().view(1, 2).expand(
                    num_loc_list[ll], 2) for ll in range(L)
        ])  # M x 2
        grids = torch.cat(grids, dim=0)  # M x 2
        M = grids.shape[0]

        reg_targets = []
        flattened_hms = []
        for i in range(len(gt_instances)):  # images
            boxes = gt_instances[i].gt_boxes.tensor  # N x 4
            area = gt_instances[i].gt_boxes.area()  # N
            gt_classes = gt_instances[
                i].gt_classes  # N in [0, self.num_classes]

            N = boxes.shape[0]
            if N == 0:
                reg_targets.append(grids.new_zeros((M, 4)) - INF)
                flattened_hms.append(
                    grids.new_zeros(
                        (M, 1 if self.only_proposal else heatmap_channels)))
                continue

            ll = grids[:, 0].view(M, 1) - boxes[:, 0].view(1, N)  # M x N
            t = grids[:, 1].view(M, 1) - boxes[:, 1].view(1, N)  # M x N
            r = boxes[:, 2].view(1, N) - grids[:, 0].view(M, 1)  # M x N
            b = boxes[:, 3].view(1, N) - grids[:, 1].view(M, 1)  # M x N
            reg_target = torch.stack([ll, t, r, b], dim=2)  # M x N x 4

            centers = ((boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2)  # N x 2
            centers_expanded = centers.view(1, N, 2).expand(M, N,
                                                            2)  # M x N x 2
            strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)
            centers_discret = ((centers_expanded / strides_expanded).int() *
                               strides_expanded).float() + strides_expanded / 2
            # M x N x 2

            is_peak = (((grids.view(M, 1, 2).expand(M, N, 2) -
                         centers_discret)**2).sum(dim=2) == 0)  # M x N
            is_in_boxes = reg_target.min(dim=2)[0] > 0  # M x N
            is_center3x3 = self.get_center3x3(grids, centers,
                                              strides) & is_in_boxes  # M x N
            is_cared_in_the_level = self.assign_reg_fpn(
                reg_target, reg_size_ranges)  # M x N
            reg_mask = is_center3x3 & is_cared_in_the_level  # M x N

            dist2 = ((grids.view(M, 1, 2).expand(M, N, 2) -
                      centers_expanded)**2).sum(dim=2)  # M x N
            dist2[is_peak] = 0
            radius2 = self.delta**2 * 2 * area  # N
            radius2 = torch.clamp(radius2, min=self.min_radius**2)
            weighted_dist2 = dist2 / radius2.view(1, N).expand(M, N)  # M x N
            reg_target = self._get_reg_targets(reg_target,
                                               weighted_dist2.clone(),
                                               reg_mask, area)  # M x 4

            if self.only_proposal:
                flattened_hm = self._create_agn_heatmaps_from_dist(
                    weighted_dist2.clone())  # M x 1
            else:
                flattened_hm = self._create_heatmaps_from_dist(
                    weighted_dist2.clone(),
                    gt_classes,
                    channels=heatmap_channels)  # M x C

            reg_targets.append(reg_target)
            flattened_hms.append(flattened_hm)

        # transpose im first training_targets to level first ones
        reg_targets = _transpose(reg_targets, num_loc_list)
        flattened_hms = _transpose(flattened_hms, num_loc_list)
        for ll in range(len(reg_targets)):
            reg_targets[ll] = reg_targets[ll] / float(self.strides[ll])
        reg_targets = cat([x for x in reg_targets], dim=0)  # MB x 4
        flattened_hms = cat([x for x in flattened_hms], dim=0)  # MB x C

        return pos_inds, labels, reg_targets, flattened_hms

    def _get_label_inds(self, gt_instances, shapes_per_level):
        """
        Inputs:
            gt_instances: [n_i], sum n_i = N
            shapes_per_level: L x 2 [(h_l, w_l)]_L
        Returns:
            pos_inds: N'
            labels: N'
        """
        pos_inds = []
        labels = []
        L = len(self.strides)
        B = len(gt_instances)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] *
                         shapes_per_level[:, 1]).long()  # L
        level_bases = []
        s = 0
        for ll in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[ll]
        level_bases = shapes_per_level.new_tensor(level_bases).long()  # L
        strides_default = shapes_per_level.new_tensor(
            self.strides).float()  # L
        for im_i in range(B):
            targets_per_im = gt_instances[im_i]
            bboxes = targets_per_im.gt_boxes.tensor  # n x 4
            n = bboxes.shape[0]
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2)  # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2)
            strides = strides_default.view(1, L, 1).expand(n, L, 2)
            centers_inds = (centers / strides).long()  # n x L x 2
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            pos_ind = \
                level_bases.view(1, L).expand(n, L) + \
                im_i * loc_per_level.view(1, L).expand(n, L) + \
                centers_inds[:, :, 1] * Ws + \
                centers_inds[:, :, 0]  # n x L
            is_cared_in_the_level = self.assign_fpn_level(bboxes)
            pos_ind = pos_ind[is_cared_in_the_level].view(-1)
            label = targets_per_im.gt_classes.view(n, 1).expand(
                n, L)[is_cared_in_the_level].view(-1)

            pos_inds.append(pos_ind)  # n'
            labels.append(label)  # n'
        pos_inds = torch.cat(pos_inds, dim=0).long()
        labels = torch.cat(labels, dim=0)
        return pos_inds, labels  # N, N

    def assign_fpn_level(self, boxes):
        """
        Inputs:
            boxes: n x 4
            size_ranges: L x 2
        Return:
            is_cared_in_the_level: n x L
        """
        size_ranges = boxes.new_tensor(self.sizes_of_interest).view(
            len(self.sizes_of_interest), 2)  # L x 2
        crit = ((boxes[:, 2:] - boxes[:, :2])**2).sum(dim=1)**0.5 / 2  # n
        n, L = crit.shape[0], size_ranges.shape[0]
        crit = crit.view(n, 1).expand(n, L)
        size_ranges_expand = size_ranges.view(1, L, 2).expand(n, L, 2)
        is_cared_in_the_level = (crit >= size_ranges_expand[:, :, 0]) & \
                                (crit <= size_ranges_expand[:, :, 1])
        return is_cared_in_the_level

    def assign_reg_fpn(self, reg_targets_per_im, size_ranges):
        """
        Inputs:
            reg_targets_per_im: M x N x 4
            size_ranges: M x 2
        """
        crit = \
            ((reg_targets_per_im[:, :, :2] + reg_targets_per_im[:, :, 2:])
             ** 2).sum(dim=2) ** 0.5 / 2
        # M x N
        is_cared_in_the_level = (crit >= size_ranges[:, [0]]) & \
                                (crit <= size_ranges[:, [1]])
        return is_cared_in_the_level

    def _get_reg_targets(self, reg_targets, dist, mask, area):
        """
          reg_targets (M x N x 4): long tensor
          dist (M x N)
          is_*: M x N
        """
        dist[mask == 0] = INF * 1.0
        min_dist, min_inds = dist.min(dim=1)  # M
        reg_targets_per_im = reg_targets[range(len(reg_targets)),
                                         min_inds]  # M x N x 4 --> M x 4
        reg_targets_per_im[min_dist == INF] = -INF
        return reg_targets_per_im

    def _create_heatmaps_from_dist(self, dist, labels, channels):
        '''
        dist: M x N
        labels: N
        return:
          heatmaps: M x C
        '''
        heatmaps = dist.new_zeros((dist.shape[0], channels))
        for c in range(channels):
            inds = (labels == c)  # N
            if inds.int().sum() == 0:
                continue
            heatmaps[:, c] = torch.exp(-dist[:, inds].min(dim=1)[0])
            zeros = heatmaps[:, c] < 1e-4
            heatmaps[zeros, c] = 0
        return heatmaps

    def _create_agn_heatmaps_from_dist(self, dist):
        '''
        dist: M x N
        return:
          heatmaps: M x 1
        '''
        heatmaps = dist.new_zeros((dist.shape[0], 1))
        heatmaps[:, 0] = torch.exp(-dist.min(dim=1)[0])
        zeros = heatmaps < 1e-4
        heatmaps[zeros] = 0
        return heatmaps

    def _flatten_outputs(self, clss, reg_pred, agn_hm_pred):
        # Reshape: (N, F, Hl, Wl) -> (N, Hl, Wl, F) -> (sum_l N*Hl*Wl, F)
        clss = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]) for x in clss],
            dim=0) if clss[0] is not None else None
        reg_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred], dim=0)
        agn_hm_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1)
             for x in agn_hm_pred], dim=0) if self.with_agn_hm else None
        return clss, reg_pred, agn_hm_pred

    def get_center3x3(self, locations, centers, strides):
        '''
        Inputs:
            locations: M x 2
            centers: N x 2
            strides: M
        '''
        M, N = locations.shape[0], centers.shape[0]
        locations_expanded = locations.view(M, 1, 2).expand(M, N,
                                                            2)  # M x N x 2
        centers_expanded = centers.view(1, N, 2).expand(M, N, 2)  # M x N x 2
        strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)  # M x N
        centers_discret = ((centers_expanded / strides_expanded).int() *
                           strides_expanded).float() + strides_expanded / 2
        # M x N x 2
        dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
        dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
        return (dist_x <= strides_expanded[:, :, 0]) & \
               (dist_y <= strides_expanded[:, :, 0])

    def inference(self, images, clss_per_level, reg_pred_per_level,
                  agn_hm_pred_per_level, grids):
        logits_pred = [
            x.sigmoid() if x is not None else None for x in clss_per_level
        ]
        agn_hm_pred_per_level = [
            x.sigmoid() if x is not None else None
            for x in agn_hm_pred_per_level
        ]

        if self.only_proposal:
            proposals = self.predict_instances(
                grids, agn_hm_pred_per_level, reg_pred_per_level,
                images.image_sizes, [None for _ in agn_hm_pred_per_level])
        else:
            proposals = self.predict_instances(grids, logits_pred,
                                               reg_pred_per_level,
                                               images.image_sizes,
                                               agn_hm_pred_per_level)
        if self.as_proposal or self.only_proposal:
            for p in range(len(proposals)):
                proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
                proposals[p].objectness_logits = proposals[p].get('scores')
                proposals[p].remove('pred_boxes')

        if self.debug:
            debug_test([self.denormalizer(x) for x in images],
                       logits_pred,
                       reg_pred_per_level,
                       agn_hm_pred_per_level,
                       preds=proposals,
                       vis_thresh=self.vis_thresh,
                       debug_show_name=False)
        return proposals, {}

    def predict_instances(self,
                          grids,
                          logits_pred,
                          reg_pred,
                          image_sizes,
                          agn_hm_pred,
                          is_proposal=False):
        sampled_boxes = []
        for ll in range(len(grids)):
            sampled_boxes.append(
                self.predict_single_level(
                    grids[ll],
                    logits_pred[ll],
                    reg_pred[ll] * self.strides[ll],
                    image_sizes,
                    agn_hm_pred[ll],
                    ll,
                    is_proposal=is_proposal))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.nms_and_topK(boxlists, nms=not self.not_nms)
        return boxlists

    def predict_single_level(self,
                             grids,
                             heatmap,
                             reg_pred,
                             image_sizes,
                             agn_hm,
                             level,
                             is_proposal=False):
        N, C, H, W = heatmap.shape
        # put in the same format as grids
        if self.center_nms:
            heatmap_nms = nn.functional.max_pool2d(
                heatmap, (3, 3), stride=1, padding=1)
            heatmap = heatmap * (heatmap_nms == heatmap).float()
        heatmap = heatmap.permute(0, 2, 3, 1)  # N x H x W x C
        heatmap = heatmap.reshape(N, -1, C)  # N x HW x C
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3,
                                                           1)  # N x H x W x 4
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = heatmap > self.score_thresh  # 0.05
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)  # N
        pre_nms_topk = self.pre_nms_topk_train \
            if self.training else self.pre_nms_topk_test
        pre_nms_top_n = pre_nms_top_n.clamp(max=pre_nms_topk)  # N

        if agn_hm is not None:
            agn_hm = agn_hm.view(N, 1, H, W).permute(0, 2, 3, 1)
            agn_hm = agn_hm.reshape(N, -1)
            heatmap = heatmap * agn_hm[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = heatmap[i]  # HW x C
            per_candidate_inds = candidate_inds[i]  # n
            per_box_cls = per_box_cls[per_candidate_inds]  # n

            per_candidate_nonzeros = per_candidate_inds.nonzero()  # n
            per_box_loc = per_candidate_nonzeros[:, 0]  # n
            per_class = per_candidate_nonzeros[:, 1]  # n

            per_box_regression = box_regression[i]  # HW x 4
            per_box_regression = per_box_regression[per_box_loc]  # n x 4
            per_grids = grids[per_box_loc]  # n x 2

            per_pre_nms_top_n = pre_nms_top_n[i]  # 1

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_grids = per_grids[top_k_indices]

            detections = torch.stack([
                per_grids[:, 0] - per_box_regression[:, 0],
                per_grids[:, 1] - per_box_regression[:, 1],
                per_grids[:, 0] + per_box_regression[:, 2],
                per_grids[:, 1] + per_box_regression[:, 3],
            ],
                                     dim=1)  # n x 4

            # avoid invalid boxes in RoI heads
            detections[:, 2] = torch.max(detections[:, 2],
                                         detections[:, 0] + 0.01)
            detections[:, 3] = torch.max(detections[:, 3],
                                         detections[:, 1] + 0.01)
            boxlist = Instances(image_sizes[i])
            boxlist.scores = torch.sqrt(per_box_cls) \
                if self.with_agn_hm else per_box_cls  # n
            # import pdb; pdb.set_trace()
            boxlist.pred_boxes = Boxes(detections)
            boxlist.pred_classes = per_class
            results.append(boxlist)
        return results

    def nms_and_topK(self, boxlists, nms=True):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            nms_thresh = self.nms_thresh_train if self.training else \
                self.nms_thresh_test
            result = ml_nms(boxlists[i], nms_thresh) if nms else boxlists[i]
            if self.debug:
                print('#proposals before nms', len(boxlists[i]))
                print('#proposals after nms', len(result))
            num_dets = len(result)
            post_nms_topk = self.post_nms_topk_train if self.training else \
                self.post_nms_topk_test
            if num_dets > post_nms_topk:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(cls_scores.float().cpu(),
                                                 num_dets - post_nms_topk + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            if self.debug:
                print('#proposals after filter', len(result))
            results.append(result)
        return results

    def _add_more_pos(self, reg_pred, gt_instances, shapes_per_level):
        labels, level_masks, c33_inds, c33_masks, c33_regs = \
            self._get_c33_inds(gt_instances, shapes_per_level)
        N, L, K = labels.shape[0], len(self.strides), 9
        c33_inds[c33_masks == 0] = 0
        reg_pred_c33 = reg_pred[c33_inds].detach()  # N x L x K
        invalid_reg = c33_masks == 0
        c33_regs_expand = c33_regs.view(N * L * K, 4).clamp(min=0)
        if N > 0:
            with torch.no_grad():
                c33_reg_loss = self.iou_loss(
                    reg_pred_c33.view(N * L * K, 4),
                    c33_regs_expand,
                    None,
                    reduction='none').view(N, L, K).detach()  # N x L x K
        else:
            c33_reg_loss = reg_pred_c33.new_zeros((N, L, K)).detach()
        c33_reg_loss[invalid_reg] = INF  # N x L x K
        c33_reg_loss.view(N * L, K)[level_masks.view(N * L),
                                    4] = 0  # real center
        c33_reg_loss = c33_reg_loss.view(N, L * K)
        if N == 0:
            loss_thresh = c33_reg_loss.new_ones((N)).float()
        else:
            loss_thresh = torch.kthvalue(
                c33_reg_loss, self.more_pos_topk, dim=1)[0]  # N
        loss_thresh[
            loss_thresh > self.more_pos_thresh] = self.more_pos_thresh  # N
        new_pos = c33_reg_loss.view(N, L, K) < \
            loss_thresh.view(N, 1, 1).expand(N, L, K)
        pos_inds = c33_inds[new_pos].view(-1)  # P
        labels = labels.view(N, 1, 1).expand(N, L, K)[new_pos].view(-1)
        return pos_inds, labels

    def _get_c33_inds(self, gt_instances, shapes_per_level):
        '''
        Get the center (and the 3x3 region near center) locations of each
        objects Inputs: gt_instances: [n_i], sum n_i = N shapes_per_level: L
        x 2 [(h_l, w_l)]_L
        '''
        labels = []
        level_masks = []
        c33_inds = []
        c33_masks = []
        c33_regs = []
        L = len(self.strides)
        B = len(gt_instances)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] *
                         shapes_per_level[:, 1]).long()  # L
        level_bases = []
        s = 0
        for ll in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[ll]
        level_bases = shapes_per_level.new_tensor(level_bases).long()  # L
        strides_default = shapes_per_level.new_tensor(
            self.strides).float()  # L
        K = 9
        dx = shapes_per_level.new_tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1]).long()
        dy = shapes_per_level.new_tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1]).long()
        for im_i in range(B):
            targets_per_im = gt_instances[im_i]
            bboxes = targets_per_im.gt_boxes.tensor  # n x 4
            n = bboxes.shape[0]
            if n == 0:
                continue
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2)  # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2)

            strides = strides_default.view(1, L, 1).expand(n, L, 2)  #
            centers_inds = (centers / strides).long()  # n x L x 2
            center_grids = centers_inds * strides + strides // 2  # n x L x 2
            ll = center_grids[:, :, 0] - bboxes[:, 0].view(n, 1).expand(n, L)
            t = center_grids[:, :, 1] - bboxes[:, 1].view(n, 1).expand(n, L)
            r = bboxes[:, 2].view(n, 1).expand(n, L) - center_grids[:, :, 0]
            b = bboxes[:, 3].view(n, 1).expand(n, L) - center_grids[:, :, 1]
            # n x L
            reg = torch.stack([ll, t, r, b], dim=2)  # n x L x 4
            reg = reg / strides_default.view(1, L, 1).expand(n, L, 4).float()

            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            Hs = shapes_per_level[:, 0].view(1, L).expand(n, L)
            expand_Ws = Ws.view(n, L, 1).expand(n, L, K)
            expand_Hs = Hs.view(n, L, 1).expand(n, L, K)
            label = targets_per_im.gt_classes.view(n).clone()
            mask = reg.min(dim=2)[0] >= 0  # n x L
            mask = mask & self.assign_fpn_level(bboxes)
            labels.append(label)  # n
            level_masks.append(mask)  # n x L

            Dy = dy.view(1, 1, K).expand(n, L, K)
            Dx = dx.view(1, 1, K).expand(n, L, K)
            c33_ind = \
                level_bases.view(1, L, 1).expand(n, L, K) + \
                im_i * loc_per_level.view(1, L, 1).expand(n, L, K) + \
                (centers_inds[:, :, 1:2].expand(n, L, K) + Dy) * expand_Ws + \
                (centers_inds[:, :, 0:1].expand(n, L, K) + Dx)  # n x L x K

            c33_mask = \
                ((centers_inds[:, :, 1:2]
                  .expand(n, L, K) + dy) < expand_Hs) & \
                ((centers_inds[:, :, 1:2]
                  .expand(n, L, K) + dy) >= 0) & \
                ((centers_inds[:, :, 0:1]
                  .expand(n, L, K) + dx) < expand_Ws) & \
                ((centers_inds[:, :, 0:1]
                  .expand(n, L, K) + dx) >= 0)
            # Currently it hard codes the 3x3 region
            c33_reg = reg.view(n, L, 1, 4).expand(n, L, K, 4).clone()
            c33_reg[:, :, [0, 3, 6], 0] -= 1
            c33_reg[:, :, [0, 3, 6], 2] += 1
            c33_reg[:, :, [2, 5, 8], 0] += 1
            c33_reg[:, :, [2, 5, 8], 2] -= 1
            c33_reg[:, :, [0, 1, 2], 1] -= 1
            c33_reg[:, :, [0, 1, 2], 3] += 1
            c33_reg[:, :, [6, 7, 8], 1] += 1
            c33_reg[:, :, [6, 7, 8], 3] -= 1
            c33_mask = c33_mask & (c33_reg.min(dim=3)[0] >= 0)  # n x L x K
            c33_inds.append(c33_ind)
            c33_masks.append(c33_mask)
            c33_regs.append(c33_reg)

        if len(level_masks) > 0:
            labels = torch.cat(labels, dim=0)
            level_masks = torch.cat(level_masks, dim=0)
            c33_inds = torch.cat(c33_inds, dim=0).long()
            c33_regs = torch.cat(c33_regs, dim=0)
            c33_masks = torch.cat(c33_masks, dim=0)
        else:
            labels = shapes_per_level.new_zeros((0)).long()
            level_masks = shapes_per_level.new_zeros((0, L)).bool()
            c33_inds = shapes_per_level.new_zeros((0, L, K)).long()
            c33_regs = shapes_per_level.new_zeros((0, L, K, 4)).float()
            c33_masks = shapes_per_level.new_zeros((0, L, K)).bool()
        return labels, level_masks, c33_inds, c33_masks, c33_regs
        # N x L, N x L x K
