import clip
import torch
from typing import Dict
from mmengine.model import BaseModel
from mmaction.utils.typing import OptSampleList, ForwardResults
from mmaction.registry import MODELS


@MODELS.register_module()
class CLIPSimilarity(BaseModel):
    def __init__(self, clip_arch: str,
                 data_preprocessor: Dict[str, Dict],
                 loss: Dict = dict(type='CrossEntropyLoss', loss_weight=0.5)) -> None:
        super(CLIPSimilarity, self).__init__(
            data_preprocessor=data_preprocessor)
        self.clip = clip.load(clip_arch)[0]
        self.loss = MODELS.build(loss)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video. """
        b, n, c, h, w = video.shape
        video = video.view(-1, c, h, w)
        frames_features = self.encode_image(video)
        video_features = frames_features.view(b, n, -1).mean(1)
        return video_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image. """
        return self.clip.encode_image(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text. """
        return self.clip.encode_text(text)

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """Forward function. """

        if mode == 'loss':
            text_inputs = inputs['text']
            video_inputs = inputs['imgs']

            text_features = self.encode_text(text_inputs)
            video_features = self.encode_video(video_inputs)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

            logit_scale = self.clip.logit_scale.exp()
            logits_per_video = logit_scale * video_features @ text_features.t()
            logits_per_text = logits_per_video.t()

            labels = torch.arange(logits_per_video.shape[0]).to(logit_scale.device)

            sim_loss_v2t = self.loss(logits_per_video, labels)
            sim_loss_t2v = self.loss(logits_per_text, labels)

            losses = dict()
            losses['sim_loss_v2t'] = sim_loss_v2t
            losses['sim_loss_t2v'] = sim_loss_t2v
            return losses

        else:
            raise ValueError('TODO')
