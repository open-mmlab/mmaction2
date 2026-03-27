import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from mmaction.models.builder import RECOGNIZERS, build_backbone, build_head, build_loss
from mmaction.models.recognizers.base import BaseRecognizer


@RECOGNIZERS.register_module()
class DistillPoseC3D(BaseRecognizer):
    """Pose C3D model with knowledge distillation.

    Args:
        backbone (dict): Backbone modules to extract features.
        teacher_backbone (dict): Teacher backbone modules.
        teacher_checkpoint (str): Path to teacher model checkpoint.
        cls_head (dict): Classification head to process features.
        teacher_cls_head (dict, optional): Teacher classification head.
            Default: None.
        train_cfg (dict, optional): Config for training. Default: None.
        test_cfg (dict, optional): Config for testing. Default: None.
    """

    def __init__(self,
                 backbone,
                 teacher_backbone,
                 teacher_checkpoint,
                 cls_head,
                 teacher_cls_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(backbone=backbone, cls_head=cls_head,
                         train_cfg=train_cfg, test_cfg=test_cfg)

        # Teacher model
        self.teacher_backbone = build_backbone(teacher_backbone)
        if teacher_cls_head is not None:
            self.teacher_cls_head = build_head(teacher_cls_head)
        else:
            # Use same head config as student but with teacher's feature size
            teacher_cls_head_cfg = copy.deepcopy(cls_head)
            teacher_cls_head_cfg['in_channels'] = self.teacher_backbone.feat_dim
            self.teacher_cls_head = build_head(teacher_cls_head_cfg)

        # Load teacher weights
        self.load_teacher(teacher_checkpoint)

        # Freeze teacher parameters
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_cls_head.parameters():
            param.requires_grad = False

    def load_teacher(self, checkpoint):
        """Load teacher model weights."""
        if not checkpoint:
            print("Warning: No teacher checkpoint provided. Using random weights.")
            return

        print(f"Loading teacher model from {checkpoint}")
        try:
            state_dict = torch.load(checkpoint, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Load backbone and head weights
            teacher_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    teacher_state_dict[k.replace('backbone.', '')] = v
                elif k.startswith('cls_head.'):
                    teacher_state_dict[k.replace('cls_head.', '')] = v

            # Load backbone weights
            backbone_state_dict = {k: v for k, v in teacher_state_dict.items()
                                   if k in self.teacher_backbone.state_dict()}
            if backbone_state_dict:
                self.teacher_backbone.load_state_dict(
                    backbone_state_dict, strict=False)
                print(
                    f"Loaded {len(backbone_state_dict)} keys into teacher backbone")
            else:
                print("Warning: No matching keys found for teacher backbone")

            # Load head weights
            head_state_dict = {k: v for k, v in teacher_state_dict.items()
                               if k in self.teacher_cls_head.state_dict()}
            if head_state_dict:
                self.teacher_cls_head.load_state_dict(
                    head_state_dict, strict=False)
                print(f"Loaded {len(head_state_dict)} keys into teacher head")
            else:
                print("Warning: No matching keys found for teacher head")

        except Exception as e:
            print(f"Error loading teacher checkpoint: {e}")

    def extract_feat(self, imgs):
        """Extract features through the backbone."""
        return self.backbone(imgs)

    def forward_train(self, imgs, labels, **kwargs):
        """Forward computation during training."""
        # Student forward pass
        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)

        # Teacher forward pass (no gradient computation)
        with torch.no_grad():
            teacher_x = self.teacher_backbone(imgs)
            teacher_cls_score = self.teacher_cls_head(teacher_x)

        # Compute distillation loss
        distill_loss_cfg = self.train_cfg.get('distill_loss',
                                              dict(type='KLDivLoss',
                                                   temperature=4.0,
                                                   alpha=0.5))

        # Build loss if it's a dict config
        if isinstance(distill_loss_cfg, dict):
            loss_type = distill_loss_cfg.pop('type')
            if loss_type == 'KLDivLoss':
                temperature = distill_loss_cfg.get('temperature', 4.0)
                alpha = distill_loss_cfg.get('alpha', 0.5)

                # Soft targets from teacher
                soft_targets = F.softmax(
                    teacher_cls_score / temperature, dim=1)
                # Softmax with temperature for student
                soft_prob = F.log_softmax(cls_score / temperature, dim=1)

                # KL divergence loss for soft targets
                kd_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * \
                    (temperature ** 2)

                # Hard target loss
                ce_loss = F.cross_entropy(cls_score, labels)

                # Combined loss
                loss = alpha * kd_loss + (1 - alpha) * ce_loss

                losses = {
                    'loss_cls': loss,
                    'loss_kd': kd_loss,
                    'loss_ce': ce_loss
                }
            else:
                # Use mmaction's loss builder for other loss types
                distill_loss = build_loss(
                    dict(type=loss_type, **distill_loss_cfg))
                loss = distill_loss(cls_score, teacher_cls_score, labels)
                losses = {'loss_cls': loss}
        else:
            # If not a dict, assume it's already a loss function
            loss = distill_loss_cfg(cls_score, teacher_cls_score, labels)
            losses = {'loss_cls': loss}

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        # Only use student model during testing
        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)

        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs)

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs."""
        return self._do_test(imgs)

    def forward(self, imgs, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get('gradcam', False):
            return self.forward_gradcam(imgs)
        if return_loss:
            if self.train_cfg is None:
                raise ValueError('Cannot train without train_cfg')
            return self.forward_train(imgs, **kwargs)

        return self.forward_test(imgs)
