import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.models.builder import LOSSES
from mmaction.registry import MODELS

@MODELS.register_module()
class AttentionTransferLoss(nn.Module):
    """Knowledge distillation via attention transfer."""
    
    def __init__(self, beta=1.0, normalize=True):
        super().__init__()
        self.beta = beta
        self.normalize = normalize
        
    def _attention_map(self, feat):
        """Convert feature maps to attention maps."""
        # Sum of absolute values for channel-wise attention (L2 norm along channel dimension)
        return F.normalize(feat.pow(2).sum(1), p=1, dim=(1, 2))
        
    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat (list[Tensor]): List of feature maps from student
            teacher_feat (list[Tensor]): List of feature maps from teacher
        """
        at_loss = 0
        for s, t in zip(student_feat, teacher_feat):
            # Handle different spatial dimensions with interpolation
            if s.shape[2:] != t.shape[2:]:
                s = F.interpolate(s, t.shape[2:], mode='trilinear', align_corners=False)
            
            # Generate attention maps
            s_attention = self._attention_map(s)
            t_attention = self._attention_map(t)
            
            # Calculate L2 distance between normalized attention maps
            at_loss += F.mse_loss(s_attention, t_attention)
            
        return self.beta * at_loss