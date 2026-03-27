import torch
import torch.nn as nn
from mmaction.registry import MODELS

@MODELS.register_module()
class MultiLevelFeatureDistillConnector(nn.Module):
    """Connector for multi-level feature distillation."""
    
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 out_channels=[256, 512, 1024, 2048],
                 kernel_sizes=[1, 1, 1, 1]):
        super().__init__()
        
        self.transformers = nn.ModuleList()
        for i, (in_ch, out_ch, k) in enumerate(zip(in_channels, out_channels, kernel_sizes)):
            self.transformers.append(
                nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=1, padding=k//2)
            )
            
    def forward(self, feats):
        """Transform student features to match teacher dimensions."""
        out_feats = []
        for i, feat in enumerate(feats):
            out_feats.append(self.transformers[i](feat))
        return out_feats