import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.models.builder import LOSSES
from mmaction.registry import MODELS

@MODELS.register_module()
class KDLoss(nn.Module):
    """Standard KL-divergence based knowledge distillation loss."""

    def __init__(self, temperature=4.0, alpha=0.5, reduction='batchmean'):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction=reduction)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels=None):
        """
        Args:
            student_logits (Tensor): Logits from student model
            teacher_logits (Tensor): Logits from teacher model
            labels (Tensor, optional): Ground truth labels
        """
        # Apply temperature scaling
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        # KL divergence loss
        kd_loss = self.kl_div(soft_student, soft_teacher) * \
            (self.temperature ** 2)

        if labels is not None:
            # Hard label loss (cross-entropy)
            ce_loss = self.ce(student_logits, labels)
            # Combined loss
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            return total_loss, ce_loss, kd_loss
        else:
            return kd_loss


@MODELS.register_module()
class FeatureDistillationLoss(nn.Module):
    """Feature-based distillation loss with optional transformation."""

    def __init__(self,
                 student_channels=256,
                 teacher_channels=2048,
                 distill_type='mse',
                 transform_type='linear',
                 weight=1.0):
        super().__init__()
        self.weight = weight
        self.distill_type = distill_type

        # Feature transformation if dimensions differ
        if transform_type == 'linear' and student_channels != teacher_channels:
            self.transform = nn.Linear(student_channels, teacher_channels)
        elif transform_type == 'conv1x1':
            self.transform = nn.Conv3d(student_channels, teacher_channels,
                                       kernel_size=1, stride=1, padding=0)
        else:
            self.transform = None

    def forward(self, student_feat, teacher_feat):
        """Calculate feature distillation loss."""
        if self.transform is not None:
            if len(student_feat.shape) == 2:  # Linear features
                student_feat = self.transform(student_feat)
            else:  # 3D features
                student_feat = self.transform(student_feat)

        # Match shapes if needed (e.g., via pooling)
        if student_feat.shape != teacher_feat.shape:
            # Apply adaptive pooling to match spatial dimensions
            if len(student_feat.shape) == 5:  # 3D features (B,C,T,H,W)
                pool = nn.AdaptiveAvgPool3d(teacher_feat.shape[2:])
                student_feat = pool(student_feat)

        # Calculate loss based on type
        if self.distill_type == 'mse':
            loss = F.mse_loss(student_feat, teacher_feat)
        elif self.distill_type == 'l1':
            loss = F.l1_loss(student_feat, teacher_feat)
        elif self.distill_type == 'cosine':
            student_norm = F.normalize(student_feat, dim=1)
            teacher_norm = F.normalize(teacher_feat, dim=1)
            loss = 1 - (student_norm * teacher_norm).sum(dim=1).mean()

        return self.weight * loss
