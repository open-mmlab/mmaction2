# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModel

from mmaction.registry import MODELS


@MODELS.register_module()
class RecognizerOmni(BaseModel):
    """Boundary Matching Network for temporal action proposal generation.

    Please refer `BMN: Boundary-Matching Network for Temporal Action Proposal
    Generation <https://arxiv.org/abs/1907.09702>`_.
    Code Reference https://github.com/JJBOY/BMN-Boundary-Matching-Network
    Args:
        temporal_dim (int): Total frames selected for each video.
        boundary_ratio (float): Ratio for determining video boundaries.
        num_samples (int): Number of samples for each proposal.
        num_samples_per_bin (int): Number of bin samples for each sample.
        feat_dim (int): Feature dimension.
        soft_nms_alpha (float): Soft NMS alpha.
        soft_nms_low_threshold (float): Soft NMS low threshold.
        soft_nms_high_threshold (float): Soft NMS high threshold.
        post_process_top_k (int): Top k proposals in post process.
        feature_extraction_interval (int):
            Interval used in feature extraction. Default: 16.
        loss_cls (dict): Config for building loss.
            Default: ``dict(type='BMNLoss')``.
        hidden_dim_1d (int): Hidden dim for 1d conv. Default: 256.
        hidden_dim_2d (int): Hidden dim for 2d conv. Default: 128.
        hidden_dim_3d (int): Hidden dim for 3d conv. Default: 512.
    """

    def __init__(self, backbone, cls_head):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(cls_head)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x, y, mode, **kwargs):
        print(type(x), len(x), 'fuckme')
        print(type(x), len(x), 'fuckyou')
        exit()

    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        pass

    def predict(self, batch_inputs, batch_data_samples, **kwargs):
        pass

    def _forward(self, x):
        pass
