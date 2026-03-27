import torch
import torch.nn as nn
from mmaction.registry import MODELS
from mmaction.models.recognizers.base import BaseRecognizer

@MODELS.register_module()
class ProgressiveDistillPoseRecognizer(BaseRecognizer):
    """Recognizer with progressive distillation strategy."""
    
    def __init__(self,
                 backbone,
                 cls_head,
                 teacher_config=None,
                 teacher_ckpt=None,
                 feature_dist_cfg=None,
                 attention_dist_cfg=None,
                 logit_dist_cfg=None,
                 progressive_cfg=dict(
                     phases=[
                         dict(epochs=(0, 10), distill_types=['feature']),
                         dict(epochs=(10, 20), distill_types=['feature', 'attention']),
                         dict(epochs=(20, -1), distill_types=['feature', 'attention', 'logit'])
                     ]
                 ),
                 train_cfg=None,
                 test_cfg=None):
        
        super().__init__(backbone, None, train_cfg, test_cfg)
        self.cls_head = cls_head
        self.teacher_model = None
        self.current_epoch = 0
        self.progressive_cfg = progressive_cfg
        
        # Initialize distillation losses
        self.feature_dist = None
        self.attention_dist = None
        self.logit_dist = None
        
        if feature_dist_cfg is not None:
            from mmaction.models.builder import build_loss
            self.feature_dist = build_loss(feature_dist_cfg)
            
        if attention_dist_cfg is not None:
            from mmaction.models.builder import build_loss
            self.attention_dist = build_loss(attention_dist_cfg)
            
        if logit_dist_cfg is not None:
            from mmaction.models.builder import build_loss
            self.logit_dist = build_loss(logit_dist_cfg)
        
        # Initialize teacher model if provided
        if teacher_config is not None and teacher_ckpt is not None:
            from mmaction.apis import init_recognizer
            self.teacher_model = init_recognizer(teacher_config, teacher_ckpt)
            # Freeze teacher model
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            
    def extract_feat(self, imgs):
        """Extract features through a backbone (student)."""
        x = self.backbone(imgs)
        return x
        
    def extract_teacher_feat(self, imgs):
        """Extract features from teacher model."""
        with torch.no_grad():
            if hasattr(self.teacher_model, 'backbone'):
                teacher_feats = self.teacher_model.backbone(imgs)
            else:
                teacher_feats = self.teacher_model.extract_feat(imgs)
        return teacher_feats
        
    def get_active_distill_types(self):
        """Get active distillation types based on current epoch."""
        active_types = []
        for phase in self.progressive_cfg['phases']:
            start_epoch, end_epoch = phase['epochs']
            if end_epoch == -1 or self.current_epoch < end_epoch:
                if self.current_epoch >= start_epoch:
                    active_types.extend(phase['distill_types'])
        return list(set(active_types))  # Remove duplicates
    
    def forward_train(self, imgs, labels, **kwargs):
        """Training forward function with progressive distillation."""
        # Update current epoch if provided
        if 'epoch' in kwargs:
            self.current_epoch = kwargs['epoch']
            
        # Get active distillation types
        active_types = self.get_active_distill_types()
        
        # Extract features from student model
        student_feat = self.extract_feat(imgs)
        
        # Get predictions from student
        cls_score = self.cls_head(student_feat)
        losses = dict()
        
        # Standard classification loss
        cls_loss = self.cls_head.loss(cls_score, labels)
        losses.update(cls_loss)
        
        # Skip distillation if teacher not available
        if self.teacher_model is None:
            return losses
            
        # Apply active distillation strategies
        teacher_logits = None
        teacher_feats = None
        
        if 'logit' in active_types and self.logit_dist is not None:
            if teacher_logits is None:
                # Extract teacher logits
                with torch.no_grad():
                    teacher_feat = self.extract_teacher_feat(imgs)
                    teacher_logits = self.teacher_model.cls_head(teacher_feat)
                    
            # Apply logit distillation
            kd_loss = self.logit_dist(cls_score, teacher_logits, labels)
            if isinstance(kd_loss, tuple):
                losses['loss_kd'] = kd_loss[0]
                losses['loss_ce_kd'] = kd_loss[1]
                losses['loss_kl_kd'] = kd_loss[2]
            else:
                losses['loss_kd'] = kd_loss
                
        if 'feature' in active_types and self.feature_dist is not None:
            if teacher_feats is None:
                # Extract teacher features
                with torch.no_grad():
                    teacher_feats = self.extract_teacher_feat(imgs)
            
            # Apply feature distillation
            feat_dist_loss = self.feature_dist(student_feat, teacher_feats)
            losses['loss_feat_dist'] = feat_dist_loss
            
        if 'attention' in active_types and self.attention_dist is not None:
            if teacher_feats is None:
                # Extract teacher features
                with torch.no_grad():
                    teacher_feats = self.extract_teacher_feat(imgs)
                    
            # Apply attention transfer
            at_loss = self.attention_dist(student_feat, teacher_feats)
            losses['loss_attention'] = at_loss
            
        return losses

    def forward_test(self, imgs):
        """Test function."""
        # During testing, only use student model
        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)
        
        return cls_score.cpu().numpy()