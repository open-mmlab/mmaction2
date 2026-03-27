from tools.data.video_retrieval.prepare_msvd import F
import torch
import torch.nn as nn
from mmcv.runner import Hook, HOOKS
from mmaction.models.builder import LOSSES
from mmaction.registry import MODELS

@MODELS.register_module()
class DynamicTemperatureKDLoss(nn.Module):
    """KD loss with dynamic temperature scheduling."""
    
    def __init__(self, 
                 init_temperature=4.0, 
                 final_temperature=1.0,
                 alpha=0.5):
        super().__init__()
        self.init_temp = init_temperature
        self.final_temp = final_temperature
        self.alpha = alpha
        self.current_temp = init_temperature
        self.ce = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels=None):
        """Forward function with current temperature."""
        # Temperature scaled logits
        soft_student = F.log_softmax(student_logits / self.current_temp, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.current_temp, dim=1)
        
        # KL divergence loss with temperature scaling
        kd_loss = self.kl_div(soft_student, soft_teacher) * (self.current_temp ** 2)
        
        if labels is not None:
            ce_loss = self.ce(student_logits, labels)
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            return total_loss, ce_loss, kd_loss
        else:
            return kd_loss

@MODELS.register_module()
class DynamicTemperatureHook(Hook):
    """Hook to update temperature during training."""
    
    def __init__(self, by_epoch=True):
        self.by_epoch = by_epoch
        
    def before_train_epoch(self, runner):
        """Update temperature at the beginning of each epoch."""
        if not self.by_epoch:
            return
            
        # Get current progress ratio
        progress_ratio = runner.epoch / runner.max_epochs
        
        # Find all KD loss modules
        for name, module in runner.model.named_modules():
            if isinstance(module, DynamicTemperatureKDLoss):
                # Linear annealing from init_temp to final_temp
                module.current_temp = module.init_temp + progress_ratio * (
                    module.final_temp - module.init_temp)
                
    def before_train_iter(self, runner):
        """Update temperature at each iteration if not by_epoch."""
        if self.by_epoch:
            return
            
        # Get current progress ratio
        progress_ratio = (runner.iter + 1) / runner.max_iters
        
        # Find all KD loss modules
        for name, module in runner.model.named_modules():
            if isinstance(module, DynamicTemperatureKDLoss):
                # Linear annealing from init_temp to final_temp
                module.current_temp = module.init_temp + progress_ratio * (
                    module.final_temp - module.init_temp)