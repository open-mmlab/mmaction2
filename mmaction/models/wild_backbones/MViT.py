import torch

from mmaction.models.builder import BACKBONES


@BACKBONES.register_module()
class MViT(torch.nn.Module):
    def __init__(self, model_type="mvit_base_16x4", pretrained=True, flow_input=False):
        super().__init__()
        assert model_type in ('mvit_base_16', 'mvit_base_16x4', 'mvit_base_32x3')
        model = torch.hub.load("facebookresearch/pytorchvideo", model=model_type, pretrained=pretrained)
        model.head = None
        self.model = model
        if flow_input:
            w = model.patch_embed.patch_model.weight
            ww = w.mean(dim=1, keepdim=True)
            ww = torch.cat([ww, ww], dim=1)
            conv_flow = torch.nn.Conv3d(96, 2, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
            conv_flow.weight = torch.nn.Parameter(ww, requires_grad=True)
            conv_flow.bias = model.patch_embed.patch_model.bias
            self.model.patch_embed.patch_model = conv_flow

    def forward(self, x):
        x = self.model(x)
        return x[:, 0, :]

    def init_weights(self):
        """The model is already initialized in the __init__ function"""
        pass
