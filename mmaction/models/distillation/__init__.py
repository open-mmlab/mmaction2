# Add imports for distillation components
from .losses import (KDLoss, FeatureDistillationLoss, AttentionTransferLoss,
                    DynamicTemperatureKDLoss, HintLoss, MultiTeacherDistillationLoss,
                    SelfAttentionDistillationLoss)
from .recognizers import DistillPoseRecognizer, ProgressiveDistillPoseRecognizer
from .necks import MultiLevelFeatureDistillConnector
from .backbones import PoseC3DStudentBackbone
from .heads import PoseC3DDistillHead

__all__ = [
    # ... existing components
    'KDLoss', 'FeatureDistillationLoss', 'AttentionTransferLoss',
    'DynamicTemperatureKDLoss', 'HintLoss', 'MultiTeacherDistillationLoss', 
    'SelfAttentionDistillationLoss', 'DistillPoseRecognizer', 
    'ProgressiveDistillPoseRecognizer', 'MultiLevelFeatureDistillConnector',
    'PoseC3DStudentBackbone', 'PoseC3DDistillHead'
]