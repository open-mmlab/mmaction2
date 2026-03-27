from mmcv.runner import HOOKS, Hook
from mmaction.registry import MODELS

@MODELS.register_module()
class DistillationHook(Hook):
    """Hook for handling knowledge distillation during training."""
    
    def __init__(self,
                alpha_scheduler=None,  # Schedule for alpha value
                temp_scheduler=None,   # Schedule for temperature
                ):
        self.alpha_scheduler = alpha_scheduler
        self.temp_scheduler = temp_scheduler
        
    def before_train_epoch(self, runner):
        """Called before each training epoch."""
        # Update alpha value if scheduler provided
        if self.alpha_scheduler is not None:
            alpha = self.alpha_scheduler(runner.epoch, runner.max_epochs)
            self._update_alpha(runner.model, alpha)
            
        # Update temperature if scheduler provided
        if self.temp_scheduler is not None:
            temp = self.temp_scheduler(runner.epoch, runner.max_epochs)
            self._update_temperature(runner.model, temp)
    
    def _update_alpha(self, model, alpha):
        """Update alpha value in KD loss modules."""
        for name, module in model.named_modules():
            if hasattr(module, 'alpha') and isinstance(module.alpha, float):
                module.alpha = alpha
                
    def _update_temperature(self, model, temp):
        """Update temperature in KD loss modules."""
        for name, module in model.named_modules():
            if hasattr(module, 'temperature') and not isinstance(module, DynamicTemperatureKDLoss):
                module.temperature = temp