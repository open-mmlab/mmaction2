from typing import Dict, List, Optional, Union

import clip
import mmengine
import torch
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.structures import LabelData

from mmaction.registry import MODELS
from .adapter import TransformerAdapter


def text_prompt(labels_or_label_file, template=None):
    if isinstance(labels_or_label_file, str):
        labels = mmengine.list_from_file(labels_or_label_file)
    elif isinstance(labels_or_label_file, list):
        labels = labels_or_label_file
    else:
        raise ValueError(f'`labels_or_label_file` must be `list` or `str`, '
                         f'but got {type(labels_or_label_file)}')

    if template is None:
        template = [
            'a photo of action {}', 'a picture of action {}',
            'Human action of {}', '{}, an action', '{} this is an action',
            '{}, a video of action', 'Playing action of {}', '{}',
            'Playing a kind of action, {}', 'Doing a kind of action, {}',
            'Look, the human is {}', 'Can you recognize the action of {}?',
            'Video classification of {}', 'A video of {}', 'The man is {}',
            'The woman is {}'
        ]
    elif isinstance(template, str):
        template = [template]
    elif not mmengine.is_seq_of(template, str):
        raise ValueError(f'`template` must be list of `str`, `str` or `None`, '
                         f'but got {type(template)}')

    num_prompt = len(template)
    prompt = torch.cat(
        [clip.tokenize(t.format(c)) for t in template for c in labels])
    return prompt, num_prompt


@MODELS.register_module()
class ActionClip(BaseModel):

    def __init__(self,
                 clip_arch: str,
                 num_adapter_segs: int,
                 num_adapter_layers: int = 6,
                 labels_or_label_file: Optional[Union[List[str], str]] = None,
                 template: Optional[Union[List[str], str]] = None,
                 data_preprocessor: Optional[Dict] = None):
        super(ActionClip, self).__init__(data_preprocessor=data_preprocessor)
        self.clip = clip.load(clip_arch)[0]
        self.adapter = TransformerAdapter(self.clip, num_adapter_segs,
                                          num_adapter_layers)

        if labels_or_label_file is not None:
            self.prompt, self.num_prompt = text_prompt(labels_or_label_file,
                                                       template)
            self.text_features = None

    def encode_video(self, video):
        b, n, c, h, w = video.shape
        video = video.view(-1, c, h, w)
        frames_features = self.encode_image(video)
        frames_features = frames_features.view(b, n, -1)
        video_features = self.adapter(frames_features)
        return video_features

    def encode_image(self, image):
        return self.clip.encode_image(image)

    def encode_text(self, text):
        return self.clip.encode_text(text)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List] = None,
                mode: str = 'tensor'):

        if mode == 'tensor':
            return self.encode_video(inputs)

        elif mode == 'predict':
            assert hasattr(self, 'prompt'),\
                '`labels_or_label_file` is required to perform prediction. '

            video_features = self.encode_video(inputs)
            video_features = video_features / video_features.norm(
                dim=-1, keepdim=True)

            bsz = len(data_samples)
            num_views = video_features.shape[0] // bsz

            if self.text_features is None:
                text_features = self.encode_text(self.prompt.to(inputs.device))
                self.text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True)

            # (bsz*num_views, num_prompt, num_classes) ->
            # (bsz, num_views*num_prompt, num_classes)
            similarity = (100.0 * video_features @ self.text_features.T). \
                view(bsz, num_views * self.num_prompt, -1)

            cls_scores = F.softmax(similarity, dim=2).mean(dim=1)

            for data_sample, score in zip(data_samples, cls_scores):
                data_sample.pred_scores = LabelData(item=score)

            return data_samples

        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports `predict` and `tensor` mode. ')
