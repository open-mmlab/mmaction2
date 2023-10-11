# Copyright (c) OpenMMLab. All rights reserved.
from .aagcn import AAGCN
from .c2d import C2D
from .c3d import C3D
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v2_tsm import MobileNetV2TSM
from .mvit import MViT
from .resnet import ResNet
from .resnet2plus1d import ResNet2Plus1d
from .resnet3d import ResNet3d, ResNet3dLayer
from .resnet3d_csn import ResNet3dCSN
from .resnet3d_slowfast import ResNet3dSlowFast
from .resnet3d_slowonly import ResNet3dSlowOnly
from .resnet_audio import ResNetAudio
from .resnet_omni import OmniResNet
from .resnet_tin import ResNetTIN
from .resnet_tsm import ResNetTSM
from .rgbposeconv3d import RGBPoseConv3D
from .stgcn import STGCN
from .swin import SwinTransformer3D
from .tanet import TANet
from .timesformer import TimeSformer
from .uniformer import UniFormer
from .uniformerv2 import UniFormerV2
from .vit_mae import VisionTransformer
from .x3d import X3D

__all__ = [
    'AAGCN', 'C2D', 'C3D', 'MViT', 'MobileNetV2', 'MobileNetV2TSM',
    'OmniResNet', 'ResNet', 'ResNet2Plus1d', 'ResNet3d', 'ResNet3dCSN',
    'ResNet3dLayer', 'ResNet3dSlowFast', 'ResNet3dSlowOnly', 'ResNetAudio',
    'ResNetTIN', 'ResNetTSM', 'STGCN', 'SwinTransformer3D', 'TANet',
    'TimeSformer', 'UniFormer', 'UniFormerV2', 'VisionTransformer', 'X3D',
    'RGBPoseConv3D'
]

try:
    from .mobileone_tsm import MobileOneTSM  # noqa: F401
    __all__.append('MobileOneTSM')

except (ImportError, ModuleNotFoundError):
    pass
