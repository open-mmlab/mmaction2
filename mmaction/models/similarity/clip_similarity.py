# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, Any, List

import clip
import torch
from mmengine.model import BaseModel
<<<<<<< Updated upstream
=======
from mmengine.dist import all_gather, get_rank
>>>>>>> Stashed changes
from mmengine.structures import InstanceData

from mmaction.registry import MODELS
from mmaction.utils.typing import ForwardResults, OptSampleList


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

    def __init__(
        self,
        clip_arch: str,
        data_preprocessor: Dict[str, Dict],
        to_float32: bool = False,
        loss: Dict = dict(type='CrossEntropyLoss', loss_weight=0.5)
    ) -> None:
        super(CLIPSimilarity,
              self).__init__(data_preprocessor=data_preprocessor)
        self.clip = clip.load(clip_arch)[0]
        if to_float32:
            self.clip.float()
        self.loss = MODELS.build(loss)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video."""
        b, n, c, h, w = video.shape
        video = video.view(-1, c, h, w)
        frames_features = self.encode_image(video)
        video_features = frames_features.view(b, n, -1).mean(1)
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
            video_features = torch.cat(GatherLayer.apply(video_features), dim=0)
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
