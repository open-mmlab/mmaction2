from mmaction.models.backbones.resnet3d_slowfast import ResNet3dSlowFast
from mmaction.models.data_preprocessors.data_preprocessor import \
    ActionDataPreprocessor
from mmaction.models.heads.slowfast_head import SlowFastHead
from mmaction.models.recognizers.recognizer3d import Recognizer3D

# model settings
model = dict(
    type=Recognizer3D,
    backbone=dict(
        type=ResNet3dSlowFast,
        pretrained=None,
        resample_rate=8,  # tau
        speed_ratio=8,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type=SlowFastHead,
        in_channels=2304,  # 2048+256
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
    data_preprocessor=dict(
        type=ActionDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))
