import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.registry import MODELS

@MODELS.register_module()
class SelfAttentionDistillationLoss(nn.Module):
    """Distill self-attention knowledge from teacher to student."""
    
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        
    def _get_attention_map(self, feat):
        """Generate self-attention map: Q*K^T."""
        # Reshape from [B,C,T,H,W] to [B,C,THW]
        b, c = feat.shape[:2]
        feat_flat = feat.reshape(b, c, -1)
        
        # Normalize feature for numerical stability
        feat_norm = F.normalize(feat_flat, dim=1)
        
        # Compute self-attention: [B,THW,THW]
        attention = torch.bmm(feat_norm.transpose(1, 2), feat_norm)
        return attention
        
    def forward(self, student_feat, teacher_feat):
        """Calculate self-attention distillation loss."""
        # Generate attention maps
        student_attention = self._get_attention_map(student_feat)
        teacher_attention = self._get_attention_map(teacher_feat)
        
        # If dimensions don't match, downsample the larger one
        if student_attention.shape != teacher_attention.shape:
            # Get smaller dimension
            min_dim = min(student_attention.shape[-1], teacher_attention.shape[-1])
            
            # Adaptive pooling to smaller dimension
            if student_attention.shape[-1] > min_dim:
                pool = nn.AdaptiveAvgPool2d(min_dim)
                student_attention = pool(student_attention)
            if teacher_attention.shape[-1] > min_dim:
                pool = nn.AdaptiveAvgPool2d(min_dim)
                teacher_attention = pool(teacher_attention)
                
        # Calculate loss: MSE between attention maps
        loss = F.mse_loss(student_attention, teacher_attention)
        
        return self.weight * loss