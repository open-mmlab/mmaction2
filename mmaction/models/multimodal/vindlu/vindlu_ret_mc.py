# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from einops import rearrange

from mmaction.registry import MODELS
from .vindlu_ret import VindLURetrieval


@MODELS.register_module()
class VindLURetrievalMC(VindLURetrieval):
    """VindLU VQA retrieval multiple choice.

    score_weight (float): Weight coefficient for itm_head score to compute the
    choice score. similarity_weight (float): Weight coefficient for similarity
    score to compute the     choice score.
    """

    def __init__(self, score_weight=0.7, similarity_weight=0.3, **kwargs):
        kwargs.pop('text_decoder')
        super().__init__(**kwargs)
        self.score_weight = score_weight
        self.similarity_weight = similarity_weight

    def predict(self, inputs, data_samples, **kwargs):
        """Predict captions from a batch of inputs.

        Args:
            images (torch.Tensor): The input images tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``

        Returns:
            List[ActionDataSample]: Return list of data samples.
        """
        num_options_per_q = len(data_samples[0].caption_options)
        for sample in data_samples:
            sample.text = sample.caption_options

        output = self.extract_feat(inputs, data_samples)

        text_embeds = output['text_embeds']
        text_attn_mask = output['text_attn_mask']
        image_embeds = output['image_embeds']
        image_feat = output['image_feat']
        text_feat = output['text_feat']

        # compute similarity between vision feat and caption feat
        text_feat = rearrange(
            text_feat, '(b n) c -> b c n', n=num_options_per_q)
        sim = torch.matmul(image_feat.mean(1, keepdim=True),
                           text_feat).squeeze(1) / self.temp
        sim = F.softmax(sim, dim=1).flatten()

        # cross-modal encode
        encoder_output = image_embeds.repeat_interleave(
            num_options_per_q, dim=0)
        image_atts = torch.ones(
            encoder_output.size()[:-1], dtype=torch.long).to(inputs.device)
        output = self.text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text_attn_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode='fusion',
        )
        itm_embeds = output.last_hidden_state[:, 0]  # [CLS]

        itm_score = F.softmax(self.itm_head(itm_embeds), dim=1)[:, 1]  # [bs*5]
        score = itm_score * self.score_weight + sim * self.similarity_weight

        pred_answers = score.view(-1, num_options_per_q).max(1)[1].cpu()

        # assemble predictions
        ensemble_scores = score.view(-1, num_options_per_q).cpu()  # (bsz, 5)

        out_data_samples = []
        for data_sample, ensemble_score, pred_ans in \
                zip(data_samples, ensemble_scores, pred_answers):
            data_sample.pred_label = pred_ans.item()
            data_sample.score = ensemble_score.numpy()
            out_data_samples.append(data_sample)

        return out_data_samples
