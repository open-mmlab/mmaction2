import torch

from ..registry import HEADS
from .slowfast_head import SlowFastHead


@HEADS.register_module()
class AVSlowFastHead(SlowFastHead):
    """The classification head for AVSlowFast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # ([N, channel_fast, T, H, W], [(N, channel_slow, T, H, W)], [(N, channel_audio, T, F)]) # noqa:E501
        x_slow, x_fast, x_audio = x
        if x_audio.dim() == 4:
            x_audio = x_audio.unsqueeze(4)
        # ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1], [N, channel_audio, 1, 1, 1]) # noqa:E501
        x_slow = self.avg_pool(x_slow)
        x_fast = self.avg_pool(x_fast)
        x_audio = self.avg_pool(x_audio)
        # [N, channel_fast + channel_slow, + channel_audio 1, 1, 1]
        x = torch.cat((x_slow, x_fast, x_audio), dim=1)

        if self.dropout is not None:
            x = self.dropout(x)

        # [N x C]
        x = x.view(x.size(0), -1)
        # [N x num_classes]
        cls_score = self.fc_cls(x)

        return cls_score
