import torch.nn as nn
from mmaction.registry import MODELS
from mmcv.cnn import ConvModule
from mmengine.model import kaiming_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd


# x: b,c,t,h,w


def n_tuple(x, num):
    return [x for i in range(num)]


@MODELS.register_module()
class VDM(nn.Module):
    """Temporal Down-Sampling Module."""

    def __init__(self,
                 in_channels=2048,
                 stage_layers=(1, 1, 1, 1),
                 kernel_sizes=(3, 1, 1),
                 strides=(2, 1, 1),
                 paddings=(1, 0, 0),
                 dilations=(1, 1, 1),
                 out_channels=512,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='SyncBN'),
                 act_cfg=dict(type='ReLU'),
                 out_indices=(0, 1, 2, 3, 4),
                 out_pooling=True,
                 ):
        super(VDM, self).__init__()

        self.in_channels = in_channels
        self.num_stages = len(stage_layers)
        self.stage_layers = stage_layers
        self.kernel_sizes = n_tuple(kernel_sizes, self.num_stages)
        self.strides = n_tuple(strides, self.num_stages)
        self.paddings = n_tuple(paddings, self.num_stages)
        self.dilations = n_tuple(dilations, self.num_stages)
        self.out_channels = n_tuple(out_channels, self.num_stages)
        self.out_pooling = out_pooling
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_indices = out_indices

        assert (len(self.stage_layers) == len(self.kernel_sizes) == len(
            self.strides) == len(self.paddings) == len(self.dilations) == len(
            self.out_channels))

        self.td_layers = []
        for i in range(self.num_stages):
            td_layer = self.make_td_layer(self.stage_layers[i], in_channels,
                                          self.out_channels[i],
                                          self.kernel_sizes[i],
                                          self.strides[i], self.paddings[i],
                                          self.dilations[i], self.conv_cfg,
                                          self.norm_cfg, self.act_cfg)
            in_channels = self.out_channels[i]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, td_layer)
            self.td_layers.append(layer_name)

        self.spatial_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))

    def sp(self, x):
        return self.spatial_pooling(x).squeeze(-1).squeeze(-1)

    @staticmethod
    def make_td_layer(num_layer, in_channels, out_channels, kernel_size,
                      stride, padding, dilation, conv_cfg, norm_cfg, act_cfg):
        layers = []
        layers.append(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        for _ in range(1, num_layer):
            layers.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initiate the parameters."""
        for m in self.modules():
            if isinstance(m, _ConvNd):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)

        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        if 0 in self.out_indices:
            outs.append(x)

        for i, layer_name in enumerate(self.td_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if (i + 1) in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        if self.out_pooling:
            for i in range(len(outs)):
                outs[i] = self.sp(outs[i])

        return tuple(outs)
