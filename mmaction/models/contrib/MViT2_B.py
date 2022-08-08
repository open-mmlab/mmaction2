import torch

from ..builder import BACKBONES
from .slowfast.models import MODEL_REGISTRY
from .slowfast.config.defaults import get_cfg
from mmcv.runner import _load_checkpoint, load_state_dict


@BACKBONES.register_module()
class MViT2_B(torch.nn.Module):
    def __init__(self, pretrained=True, flow_input=False):
        super().__init__()
        cfg = get_cfg()
        cfg.merge_from_file("mmaction/models/contrib/slowfast/config/configs/Kinetics/MVITv2_B_32x3.yaml")
        model = MODEL_REGISTRY.get(cfg.MODEL.MODEL_NAME)(cfg)
        if pretrained:
            state_dict = _load_checkpoint("https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth")
            load_state_dict(model, state_dict['model_state'])
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
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        pass
