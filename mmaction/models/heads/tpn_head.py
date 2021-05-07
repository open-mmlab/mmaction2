import torch.nn as nn

from ..builder import HEADS
from .tsn_head import TSNHead


@HEADS.register_module()
class TPNHead(TSNHead):
    """Class head for TPN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: https://arxiv.org/abs/1906.02629. Default: 0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool3d = None

        self.avg_pool2d = None
        self.new_cls = None

    def _init_new_cls(self):
        self.new_cls = nn.Conv3d(self.in_channels, self.num_classes, 1, 1, 0)
        if next(self.fc_cls.parameters()).is_cuda:
            self.new_cls = self.new_cls.cuda()
        self.new_cls.weight.copy_(self.fc_cls.weight[..., None, None, None])
        self.new_cls.bias.copy_(self.fc_cls.bias)

    def forward(self, x, num_segs=None, fcn_test=False):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int | None): Number of segments into which a video
                is divided. Default: None.
            fcn_test (bool): Whether to apply full convolution (fcn) testing.
                Default: False.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        if fcn_test:
            if self.avg_pool3d:
                x = self.avg_pool3d(x)
            if self.new_cls is None:
                self._init_new_cls()
            cls_score_feat_map = self.new_cls(x)
            return cls_score_feat_map

        if self.avg_pool2d is None:
            kernel_size = (1, x.shape[-2], x.shape[-1])
            self.avg_pool2d = nn.AvgPool3d(kernel_size, stride=1, padding=0)

        if num_segs is None:
            # [N, in_channels, 3, 7, 7]
            x = self.avg_pool3d(x)
        else:
            # [N * num_segs, in_channels, 7, 7]
            x = self.avg_pool2d(x)
            # [N * num_segs, in_channels, 1, 1]
            x = x.reshape((-1, num_segs) + x.shape[1:])
            # [N, num_segs, in_channels, 1, 1]
            x = self.consensus(x)
            # [N, 1, in_channels, 1, 1]
            x = x.squeeze(1)
            # [N, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        x = x.view(x.size(0), -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
