# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import mmengine
import torch
import torch.nn.functional as F
from einops import rearrange

from mmaction.registry import MODELS
from .vindlu import VindLUBase


@MODELS.register_module()
class VindLUVQA(VindLUBase):
    """VindLU VQA.

    Args:
        text_decoder (dict): Backbone for extracting
            multi-modal features. We apply this part as VQA fusion module.
        answer_list_path (str, optional): Path to `answer_list.json`.
        max_question_len (int): Max text length of question text.
            Defaults to 25.
        max_answer_len (int): Max text length of answer text. Defaults to 5.
        num_ans_candidates (int): Number of answer candidates, used to filter
            out answers with low probability. Defaults to 128.
        **kwargs: Other keyword arguments accepted by the VindLUBase.
    """

    def __init__(self,
                 text_decoder: dict,
                 answer_list_path: Optional[str] = None,
                 max_question_len: int = 25,
                 max_answer_len: int = 5,
                 num_ans_candidates: int = 128,
                 **kwargs):
        super().__init__(**kwargs)

        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.num_ans_candidates = num_ans_candidates
        self.answer_list_path = answer_list_path
        self.text_decoder_cfg = text_decoder

        # for inference only
        if answer_list_path:
            self.answer_list = mmengine.load(answer_list_path)

        # delete extra/unnecessary modules inherited from VindLUBase
        extra_attributes = ['vision_proj', 'text_proj', 'temp', 'itm_head']
        for attr in extra_attributes:
            delattr(self, attr)

        self.text_decoder_cfg.gradient_checkpointing = \
            self.gradient_checkpointing
        self.text_decoder = MODELS.build(self.text_decoder_cfg)

    def forward_encoder(self, inputs, data_samples):
        # forward vision encoder
        image_embeds, _ = self.encode_vision(inputs)
        image_embeds = rearrange(image_embeds, 'b t l c -> b (t l) c')
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(inputs.device)

        # forward text encoder
        questions = [sample.question for sample in data_samples]
        questions = self.tokenizer(
            questions,
            padding='max_length',
            truncation=True,
            max_length=self.max_question_len,
            return_tensors='pt').to(inputs.device)

        question_output = self.text_encoder(
            questions.input_ids,
            attention_mask=questions.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True)

        return questions, question_output

    def loss(self, inputs, data_samples):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (dict): A batch of inputs. The input tensor with of
                at least one modality. For image, the value is a tensor
                of shape (N, C, ...) in general.
                For text, the value is a dict of tokenized text inputs.
            data_samples (Optional[List[DataSample]]):
                The annotation data of every samples. Defaults to None.

        Returns:
            Dict[str, torch.tensor]: a dictionary of loss components of
        """

        questions, question_output = self.forward_encoder(inputs, data_samples)

        weights = torch.cat(
            [torch.tensor(sample.gt_answer_weight) for sample in data_samples],
            dim=0).to(inputs.device)
        raw_answers = []
        for sample in data_samples:
            raw_answers.extend(sample.gt_answer)
        answer_count = torch.tensor([
            len(sample.gt_answer) for sample in data_samples
        ]).to(inputs.device)
        answers = [a + ' ' + '[SEP]' for a in raw_answers]
        answers = self.tokenizer(
            answers,
            padding='max_length',
            truncation=True,
            max_length=self.max_answer_len,
            return_tensors='pt').to(inputs.device)

        answer_targets = answers.input_ids.masked_fill(
            answers.input_ids == self.tokenizer.pad_token_id, -100)

        question_states = []
        question_atts = []
        for b, n in enumerate(answer_count):
            question_states += [question_output.last_hidden_state[b]] * n
            question_atts += [questions.attention_mask[b]] * n
        question_states = torch.stack(question_states, 0).to(inputs.device)
        question_atts = torch.stack(question_atts, 0).to(inputs.device)

        answer_output = self.text_decoder(
            answers.input_ids,
            attention_mask=answers.attention_mask,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=answer_targets,
            return_dict=True,
            reduction='none',
        )
        loss = weights * answer_output.loss
        loss = loss.sum() / inputs.size(0)

        return dict(loss=loss)

    def predict(self, inputs, data_samples, **kwargs):

        questions, question_output = self.forward_encoder(inputs, data_samples)

        raw_answers = self.answer_list
        answers = [a + ' ' + '[SEP]' for a in raw_answers]
        answers = self.tokenizer(
            answers,
            padding='max_length',
            truncation=True,
            max_length=self.max_answer_len,
            return_tensors='pt',
        ).to(inputs.device)

        topk_ids, topk_probs = self.rank_answer(
            question_output.last_hidden_state, questions.attention_mask,
            answers.input_ids, answers.attention_mask, self.num_ans_candidates)

        out_data_samples = []
        for data_sample, topk_id, topk_prob in zip(data_samples, topk_ids,
                                                   topk_probs):
            _, pred = topk_prob.max(dim=0)
            data_sample.pred_answer = raw_answers[topk_id[pred]]
            out_data_samples.append(data_sample)

        return out_data_samples

    def rank_answer(self, question_states, question_atts, answer_ids,
                    answer_atts, k):
        """
        question_states: (bsz, Lq, d)
        answer_ids: answer input id after tokenization, (#answers, La)
        """
        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(
            start_ids,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            return_dict=True,
            reduction='none',
        )
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(
            logits, dim=1).index_select(
                dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100)

        question_states = question_states.repeat_interleave(k, dim=0)
        question_atts = question_atts.repeat_interleave(k, dim=0)

        output = self.text_decoder(
            input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=targets_ids,
            return_dict=True,
            reduction='none',
        )

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences
        # using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs

    def preprocess_state_dict(self, state_dict):
        """Preprocess pretrained checkpoint for text_encoder and
        text_decoder."""
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]

            # init text decoder as multimodal encoder
            # (last 6 layers of model.text_encoder)
            # only for generation tasks like VQA
            if self.text_decoder_cfg and 'text_encoder' in key:
                if 'layer' in key:
                    encoder_keys = key.split('.')
                    layer_num = int(encoder_keys[4])
                    if layer_num < self.text_encoder_cfg.fusion_layer:
                        del state_dict[key]
                        continue
                    else:
                        decoder_layer_num = layer_num - 9
                        encoder_keys[4] = str(decoder_layer_num)
                        encoder_key = '.'.join(encoder_keys)
                else:
                    encoder_key = key
                decoder_key = encoder_key.replace('text_encoder',
                                                  'text_decoder')
                state_dict[decoder_key] = state_dict[key]
                del state_dict[key]
        return state_dict
