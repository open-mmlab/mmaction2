import mmengine
from typing import List, Optional, Dict, Union
import torch
import torch.nn.functional as F
import clip
from mmengine.model import BaseModel
from mmaction.registry import MODELS
from .adapter import TransformerAdapter
from mmengine.structures import LabelData


def text_prompt(labels_or_label_file):
    if isinstance(labels_or_label_file, str):
        labels = mmengine.list_from_file(labels_or_label_file)
    elif isinstance(labels_or_label_file, list):
        labels = labels_or_label_file
    else:
        raise ValueError('`labels_or_label_file` must be `list` or `str`. ')

    template = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]

    num_prompt = len(template)
    prompt = torch.cat([clip.tokenize(t.format(c)) for t in template for c in labels])
    return prompt, num_prompt


@MODELS.register_module()
class ActionClip(BaseModel):

    def __init__(self, clip_arch: str,
                 num_adapter_segs: int,
                 num_adapter_layers: int = 6,
                 labels_or_label_file: Optional[Union[List[str], str]] = None,
                 data_preprocessor: Optional[Dict] = None):

        if data_preprocessor is None:
            data_preprocessor = dict(
                type='ActionDataPreprocessor',
                mean=[122.771, 116.746, 104.093],
                std=[68.500, 66.632, 70.323],
                format_shape='NCHW')

        super(ActionClip, self).__init__(data_preprocessor=data_preprocessor)

        self.clip = clip.load(clip_arch, device='cpu')[0]
        self.adapter = TransformerAdapter(self.clip, num_adapter_segs, num_adapter_layers)

        if labels_or_label_file is not None:
            self.prompt, self.num_prompt = text_prompt(labels_or_label_file)
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

    def forward(self, inputs: torch.Tensor,
                data_samples: Optional[List] = None,
                mode: str = 'tensor'):

        if mode == 'tensor':
            return self.encode_video(inputs)

        elif mode == 'predict':
            assert hasattr(self, 'prompt'),\
                "`labels_or_label_file` is required to perform prediction. "

            video_features = self.encode_video(inputs)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

            bsz = len(data_samples)
            num_views = video_features.shape[0] // bsz

            if self.text_features is None:
                text_features = self.encode_text(self.prompt.to(inputs.device))
                self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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
