from ..registry import BACKBONES
from .resnet3d import ResNet3d


@BACKBONES.register_module
class ResNet2Plus1d(ResNet3d):
    """ResNet (2+1)d backbone.

    This model is proposed in
    "A Closer Look at Spatiotemporal Convolutions for Action Recognition".
    Link: https://arxiv.org/abs/1711.11248
    """

    def __init__(self, *args, **kwargs):
        super(ResNet2Plus1d, self).__init__(*args, **kwargs)
        assert self.pretrained2d is False
        assert self.conv_cfg['type'] == 'Conv2plus1d'

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            # no pool2 in R(2+1)d
            x = res_layer(x)

        return x
