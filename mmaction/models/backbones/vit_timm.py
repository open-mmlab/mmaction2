# import torch
# import torch.nn.functional as F
# from einops import rearrange, repeat
from torch import nn

from ..registry import BACKBONES


@BACKBONES.register_module()
class ViT_Timm(nn.Module):

    def __init__(self, use_pretrained=True):
        super().__init__()
        import timm
        self.m = timm.create_model(
            'vit_base_patch16_224', pretrained=use_pretrained)

    def forward(self, img, mask=None):
        return self.m(img)
