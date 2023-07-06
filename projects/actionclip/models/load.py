import torch
from mmengine.dataset import Compose
from mmengine.runner.checkpoint import _load_checkpoint
from torchvision.transforms import Normalize

from .actionclip import ActionClip

_MODELS = {
    'ViT-B/32-8':
    'https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p32-res224-clip-pre_1x1x8_k400-rgb/vit-b-32-8f.pth',  # noqa: E501
    'ViT-B/16-8':
    'https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_1x1x8_k400-rgb/vit-b-16-8f.pth',  # noqa: E501
    'ViT-B/16-16':
    'https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_1x1x16_k400-rgb/vit-b-16-16f.pth',  # noqa: E501
    'ViT-B/16-32':
    'https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_1x1x32_k400-rgb/vit-b-16-32f.pth',  # noqa: E501
}


def available_models():
    """Returns the names of available ActionCLIP models."""
    return list(_MODELS.keys())


def _transform(num_segs):
    pipeline = [
        dict(type='DecordInit'),
        dict(
            type='SampleFrames',
            clip_len=1,
            frame_interval=1,
            num_clips=num_segs,
            test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='FormatShape', input_format='NCHW'),
        lambda x: torch.tensor(x['imgs']).div(255),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ]
    return Compose(pipeline)


def init_actionclip(name, device):
    assert name in _MODELS, \
        f'Model {name} not found; available models = {available_models()}'
    model_path = _MODELS[name]

    checkpoint = _load_checkpoint(model_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    clip_arch = name.split('-')[0] + '-' + name.split('-')[1]

    num_adapter_segs = int(name.split('-')[2])
    assert num_adapter_segs == \
           state_dict['adapter.frame_position_embeddings.weight'].shape[0]
    num_adapter_layers = len([
        k for k in state_dict.keys()
        if k.startswith('adapter.') and k.endswith('.attn.in_proj_weight')
    ])

    model = ActionClip(
        clip_arch=clip_arch,
        num_adapter_segs=num_adapter_segs,
        num_adapter_layers=num_adapter_layers)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, _transform(num_adapter_segs)
