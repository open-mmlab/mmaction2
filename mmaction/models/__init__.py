# Copyright (c) OpenMMLab. All rights reserved.
<<<<<<< HEAD
from .backbones import (C2D_R50, C3D, STGCN, X3D, MobileNetV2, MobileNetV2TSM,
                        ResNet, ResNet2Plus1d, ResNet3d, ResNet3dCSN,
                        ResNet3dLayer, ResNet3dSlowFast, ResNet3dSlowOnly,
                        ResNetAudio, ResNetTIN, ResNetTSM, TANet, TimeSformer)
from .builder import (BACKBONES, DETECTORS, HEADS, LOCALIZERS, LOSSES, NECKS,
                      RECOGNIZERS, build_backbone, build_detector, build_head,
                      build_localizer, build_loss, build_model, build_neck,
                      build_recognizer)
from .common import (LFB, TAM, Conv2plus1d, ConvAudio,
                     DividedSpatialAttentionWithNorm,
=======
from .backbones import (AGCN, C3D, STGCN, X3D, MobileNetV2, MobileNetV2TSM,
                        ResNet, ResNet2Plus1d, ResNet3d, ResNet3dCSN,
                        ResNet3dLayer, ResNet3dSlowFast, ResNet3dSlowOnly,
                        ResNetTIN, ResNetTSM, TANet, TimeSformer)
from .common import (TAM, Conv2plus1d, DividedSpatialAttentionWithNorm,
>>>>>>> upstream/dev-1.x
                     DividedTemporalAttentionWithNorm, FFNWithNorm,
                     SubBatchNorm3D)
from .data_preprocessors import ActionDataPreprocessor
from .heads import (BaseHead, I3DHead, SlowFastHead, STGCNHead,
                    TimeSformerHead, TPNHead, TRNHead, TSMHead, TSNHead,
                    X3DHead)
from .localizers import PEM, TEM
from .losses import (BCELossWithLogits, BinaryLogisticRegressionLoss, BMNLoss,
                     CBFocalLoss, CrossEntropyLoss, HVULoss, NLLLoss,
                     OHEMHingeLoss, SSNLoss)
from .necks import TPN
from .recognizers import (BaseRecognizer, Recognizer2D, Recognizer3D,
                          RecognizerGCN)
from .roi_heads import ACRNHead, AVARoIHead, BBoxHeadAVA, SingleRoIExtractor3D
from .task_modules import MaxIoUAssignerAVA
from .utils import BaseMiniBatchBlending, CutmixBlending, MixupBlending

__all__ = [
<<<<<<< HEAD
    'BACKBONES', 'HEADS', 'RECOGNIZERS', 'build_recognizer', 'build_head',
    'build_backbone', 'Recognizer2D', 'Recognizer3D', 'C2D_R50', 'C3D',
    'ResNet', 'STGCN', 'ResNet3d', 'ResNet2Plus1d', 'I3DHead', 'TSNHead',
    'TSMHead', 'BaseHead', 'STGCNHead', 'BaseRecognizer', 'LOSSES',
    'CrossEntropyLoss', 'NLLLoss', 'HVULoss', 'ResNetTSM', 'ResNet3dSlowFast',
    'SlowFastHead', 'Conv2plus1d', 'ResNet3dSlowOnly', 'BCELossWithLogits',
    'LOCALIZERS', 'build_localizer', 'PEM', 'TAM', 'TEM',
    'BinaryLogisticRegressionLoss', 'BMN', 'BMNLoss', 'build_model',
    'OHEMHingeLoss', 'SSNLoss', 'ResNet3dCSN', 'ResNetTIN', 'TPN', 'TPNHead',
    'build_loss', 'build_neck', 'AudioRecognizer', 'AudioTSNHead', 'X3D',
    'X3DHead', 'ResNet3dLayer', 'DETECTORS', 'SingleRoIExtractor3D',
    'BBoxHeadAVA', 'ResNetAudio', 'build_detector', 'ConvAudio', 'AVARoIHead',
    'MobileNetV2', 'MobileNetV2TSM', 'TANet', 'LFB', 'FBOHead', 'LFBInferHead',
    'TRNHead', 'NECKS', 'TimeSformer', 'TimeSformerHead',
    'DividedSpatialAttentionWithNorm', 'DividedTemporalAttentionWithNorm',
    'FFNWithNorm', 'ACRNHead', 'BaseGCN', 'SkeletonGCN', 'CBFocalLoss',
    'SubBatchNorm3D'
=======
    'C3D', 'ResNet', 'STGCN', 'ResNet3d', 'ResNet2Plus1d', 'I3DHead',
    'TSNHead', 'TSMHead', 'BaseHead', 'STGCNHead', 'Recognizer3D',
    'Recognizer2D', 'RecognizerGCN', 'BaseRecognizer', 'CrossEntropyLoss',
    'NLLLoss', 'HVULoss', 'ResNetTSM', 'ResNet3dSlowFast', 'SlowFastHead',
    'Conv2plus1d', 'CBFocalLoss', 'SubBatchNorm3D', 'ResNet3dSlowOnly',
    'BCELossWithLogits', 'TAM', 'BinaryLogisticRegressionLoss', 'BMNLoss',
    'OHEMHingeLoss', 'SSNLoss', 'ResNet3dCSN', 'ResNetTIN', 'TPN', 'TPNHead',
    'X3D', 'X3DHead', 'ResNet3dLayer', 'SingleRoIExtractor3D', 'BBoxHeadAVA',
    'AVARoIHead', 'MobileNetV2', 'MobileNetV2TSM', 'TANet', 'TRNHead',
    'TimeSformer', 'TimeSformerHead', 'DividedSpatialAttentionWithNorm',
    'DividedTemporalAttentionWithNorm', 'FFNWithNorm', 'ACRNHead',
    'ActionDataPreprocessor', 'BaseMiniBatchBlending', 'CutmixBlending',
    'MixupBlending', 'AGCN', 'MaxIoUAssignerAVA', 'TEM', 'PEM'
>>>>>>> upstream/dev-1.x
]
