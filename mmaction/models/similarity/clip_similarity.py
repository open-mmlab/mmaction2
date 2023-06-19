# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Tuple

import torch
from mmengine.dist import all_gather, get_rank
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmaction.registry import MODELS
from mmaction.utils import ForwardResults, OptSampleList


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> Tuple[List]:
        ctx.save_for_backward(input)
        output = all_gather(input)
        return tuple(output)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[get_rank()]
        return grad_out


@MODELS.register_module()
class CLIPSimilarity(BaseModel):
    """CLIP-based similarity model.

    Args:
        clip_arch (str): The architecture of the clip model.
            Supported choices are `'ViT-B/32'`, `'ViT-B/16'`,
            `'ViT-L/14'` and `'ViT-L/14@336px'`.
        data_preprocessor (dict): The pre-process config.
        adapter (dict): The 3D adapter config.
        to_float32 (bool): Whether to convert the dtype of params of clip
            model to float32.
        frozen_layers: Layers to be frozen (all params fixed). -1 means
            not freezing any parameters. Defaults to -1.
        loss (dict): The config of loss. Defaults to
            `dict(type='CrossEntropyLoss', loss_weight=0.5)`.
    """

    def __init__(
        self,
        clip_arch: str,
        data_preprocessor: Dict[str, Dict],
        adapter: Dict,
        to_float32: bool = False,
        frozen_layers: int = -1,
        loss: Dict = dict(type='CrossEntropyLoss', loss_weight=0.5)
    ) -> None:
        super(CLIPSimilarity,
              self).__init__(data_preprocessor=data_preprocessor)

        try:
            import clip
        except ImportError:
            raise ImportError('Please run `pip install '
                              'git+https://github.com/openai/CLIP.git` '
                              'to install clip first. ')

        self.clip = clip.load(clip_arch, device='cpu')[0]
        if to_float32:
            self.clip.float()
        self.loss = MODELS.build(loss)
        self.adapter = MODELS.build(adapter)
        self.frozen_layers = frozen_layers
        self._freeze_stages()

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video."""
        b, n, c, h, w = video.shape
        video = video.view(-1, c, h, w)
        frames_features = self.encode_image(video)
        frames_features = frames_features.view(b, n, -1)
        video_features = self.adapter(frames_features)
        return video_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image."""
        return self.clip.encode_image(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text."""
        return self.clip.encode_text(text)

    def extract_feat(self,
                     inputs: Dict[str, torch.Tensor],
                     norm: bool = True) -> Tuple:
        """Extract features."""
        text_inputs = inputs['text']
        video_inputs = inputs['imgs']
        text_features = self.encode_text(text_inputs)
        video_features = self.encode_video(video_inputs)

        if norm:
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True)
            video_features = video_features / video_features.norm(
                dim=-1, keepdim=True)

        return video_features, text_features

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """Forward function."""

        if mode == 'tensor':
            return self.extract_feat(inputs, norm=False)

        elif mode == 'loss':
            video_features, text_features = self.extract_feat(inputs)
            video_features = torch.cat(
                GatherLayer.apply(video_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

            logit_scale = self.clip.logit_scale.exp()
            logits_per_video = logit_scale * video_features @ text_features.t()
            logits_per_text = logits_per_video.t()

            labels = torch.arange(logits_per_video.shape[0]).to(
                logit_scale.device)

            sim_loss_v2t = self.loss(logits_per_video, labels)
            sim_loss_t2v = self.loss(logits_per_text, labels)

            losses = dict()
            losses['sim_loss_v2t'] = sim_loss_v2t
            losses['sim_loss_t2v'] = sim_loss_t2v
            return losses

        elif mode == 'predict':
            video_features, text_features = self.extract_feat(inputs)
            for ds, vf, tf in zip(data_samples, video_features, text_features):
                features = InstanceData(video_feature=vf, text_feature=tf)
                ds.features = features
            return data_samples

        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def train(self, mode: bool = True) -> None:
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_layers``."""

        if self.frozen_layers >= 0:
            top_layers = [
                'ln_final', 'text_projection', 'logit_scale', 'visual.ln_post',
                'visual.proj'
            ]
            mid_layers = [
                'visual.transformer.resblocks', 'transformer.resblocks'
            ]

            for name, param in self.clip.named_parameters():
                if any(name.find(n) == 0 for n in top_layers):
                    continue
                elif any(name.find(n) == 0 for n in mid_layers):
                    layer_n = int(name.split('.resblocks.')[1].split('.')[0])
                    if layer_n >= self.frozen_layers:
                        continue
                param.requires_grad = False
