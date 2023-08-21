# Copyright (c) OpenMMLab. All rights reserved.
import logging

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from mmengine.logging import MMLogger

from mmaction.registry import MODELS
from .bert_builder import build_bert_decoder
from .vindlu import VindLU


@MODELS.register_module()
class VindLURetMC(VindLU):
    """docstring for VindLU retrieval multiple choice."""

    def __init__(self, 
                 max_txt_len,
                 eval_frame_ensemble=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.max_txt_len = max_txt_len
        self.eval_frame_ensemble = eval_frame_ensemble

    def forward(self, inputs, data_samples, mode: str = 'loss'):
        """
        Args:
        k: number of answers for each question
        weights: weight for each answer
        """

        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def predict(self, inputs, data_samples, **kwargs):

        # image_feats, pooled_image_feat, captions_output, num_options_per_q \
        #     = self.forward_encoder(inputs, data_samples)

        image_feats, pooled_image_feat = self.encode_vision(inputs)
        image_feats = rearrange(image_feats, 'b t l c -> b (t l) c')
        image_feats = image_feats.unsqueeze(1)  # (bsz, 1, #frm*L, d)
        pooled_image_feat = pooled_image_feat.unsqueeze(1)  # (bsz, 1, #frm, d)
        image_atts = torch.ones(
            image_feats.size()[:-1], dtype=torch.long).to(inputs.device)

        # forward text encoder
        captions = [sample.caption_options for sample in data_samples]
        num_options_per_q = len(captions[0])
        captions = [cap for cap_list in captions for cap in cap_list]
        captions = self.tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors='pt').to(inputs.device)

        captions_output = self.text_encoder(
            captions.input_ids,
            attention_mask=captions.attention_mask,
            return_dict=True,
            mode='text')

        text_feat = captions_output.last_hidden_state
        pooled_text_feat = text_feat[:, 0]

    
        n_clip_per_video = image_feats.shape[1]
        clip_scores = []
        for clip_idx in range(n_clip_per_video):
            image_feat = image_feats[:, clip_idx]
            pooled_image_feat = pooled_image_feat[:, clip_idx]
            image_feat = tile(image_feat, 0, num_options_per_q)
            image_mask = torch.ones(image_feat.size()[:-1], dtype=torch.long).to(
                inputs.device, non_blocking=True
            )

            # contrastive score
            pooled_text_feat = rearrange(
                pooled_text_feat, "(b n) c -> b n c", n=num_options_per_q
            )
            _image_feat = F.normalize(self.vision_proj(pooled_image_feat), dim=-1)
            _text_feat = F.normalize(self.text_proj(pooled_text_feat), dim=-1)
            sim = torch.matmul(_image_feat, rearrange(_text_feat, "b n c -> b c n"))  # [b, t, n]
            sim = sim.mean(1) / self.temp  # [b,n]
            sim = F.softmax(sim, dim=1)  # [b, n]
            sim = sim.flatten()  # [b*n,]

            # cross-modal encode
            output = self.text_encoder(
                encoder_embeds=text_feat,
                attention_mask=captions.attention_mask,
                encoder_hidden_states=image_feat,
                encoder_attention_mask=image_mask,
                return_dict=True,
                mode="fusion",
            )
            itm_embeds = output.last_hidden_state[:, 0]  # [CLS]

            score = F.softmax(self.itm_head(itm_embeds), dim=1)[:, 1]  # [bs*5]
            score = score * 0.7 + sim * 0.3

            clip_scores.append(score)

        clip_scores = torch.stack(clip_scores)  # (#clips, k)
        if len(clip_scores) == 1:
            score = clip_scores[0]
        else:
            assert self.eval_frame_ensemble in ["mean", "max", "lse"]
            if self.eval_frame_ensemble == "mean":
                score = clip_scores.mean(0)
            elif self.eval_frame_ensemble == "max":
                score = clip_scores.max(0)[0]
            elif self.eval_frame_ensemble == "lse":  # LogSumExp
                score = torch.logsumexp(clip_scores, dim=0)
            else:
                raise ValueError(
                    "self.eval_frame_ensemble must in [mean, max, lse] when #clip > 1."
                )

        pred_answers = score.view(-1, num_options_per_q).max(1)[1].cpu()
        # all_pred_answers.append(pred_ans)

        # assemble predictions
        ensemble_scores = score.view(-1, num_options_per_q).cpu()  # (bsz, 5)
        clip_scores = clip_scores.view(
                -1, n_clip_per_video, num_options_per_q).cpu()  # (bsz, #clips, 5)
            
        out_data_samples = []
        for data_sample, ensemble_score, pred_ans, clip_score in \
                zip(data_samples, ensemble_scores, pred_answers, clip_scores):
            data_sample.pred_label = pred_ans.item()
            data_sample.score = ensemble_score.numpy()
            data_sample.clip_scores = clip_score.numpy()
            data_sample.clip_pred_answer = clip_score.max(1)[1].numpy()
            out_data_samples.append(data_sample)

        return out_data_samples


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*repeat_idx)
    order_index = torch.LongTensor(
        np.concatenate([
            init_dim * np.arange(n_tile) + i for i in range(init_dim)
        ]))
    return torch.index_select(x, dim, order_index.to(x.device))
