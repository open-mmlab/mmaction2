# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
import mmengine.dist as dist
from torch.distributed.nn import all_gather as all_gather_with_grad


from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.utils import track_on_main_process
from .vindlu import VindLU

def all_gather_concat(data: torch.Tensor) -> torch.Tensor:
    """Gather tensors with different first-dimension size and concat to one
    tenosr.

    Note:
        Only the first dimension should be different.

    Args:
        data (Tensor): Tensor to be gathered.

    Returns:
        torch.Tensor: The concatenated tenosr.
    """
    if dist.get_world_size() == 1:
        return data

    data_size = torch.tensor(data.size(0), device=data.device)
    sizes_list = dist.all_gather(data_size)

    total_length = sum(sizes_list)
    max_length = max(sizes_list)
    size_diff = max_length.item() - data_size.item()
    if size_diff:
        padding = torch.zeros(
            size_diff, *data.size()[1:], device=data.device, dtype=data.dtype)
        data = torch.cat((data, padding))

    gather_list = dist.all_gather(data)

    # gather all data according to the default DDP sampler. For instance, 
    # 8 samples on 2 GPUs, GPU0: [0,2,4,8], GPU1: [1,3,5,7], will be gathered
    # as [0,1,2,3,4,5,6,7]
    all_data = []
    for gather_batch in zip(*gather_list):
        all_data.extend(gather_batch)

    return torch.stack(all_data)[:total_length]


@MODELS.register_module()
class VindLURet(VindLU):
    """docstring for VindLU retrieval."""

    def __init__(self, 
                 max_txt_len,
                 topk=128,
                 eval_frame_ensemble=None,
                 negative_all_rank=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.max_txt_len = max_txt_len
        self.topk = topk
        self.eval_frame_ensemble = eval_frame_ensemble
        self.negative_all_rank = negative_all_rank

    def forward(self, inputs, data_samples, mode: str = 'loss'):
        """
        Args:
        k: number of answers for each question
        weights: weight for each answer
        """

        if mode == 'tensor':
            return self.extract_feat(inputs, data_samples)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def loss(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[ActionDataSample]] = None,
    ) -> Dict[str, torch.tensor]:
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
        output = self.extract_feat(inputs, data_samples)

        text_embeds = output['text_embeds']
        text_attn_mask = output['text_attn_mask']
        image_embeds = output['image_embeds']
        image_feat = output['image_feat']
        text_feat = output['text_feat']

        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        # ITC Loss
        # B*world_size, D
        image_feat_all = torch.cat(dist.all_gather(image_feat))
        # B*world_size, D
        text_feat_all = torch.cat(dist.all_gather(text_feat))

        # image to text similarity
        # B, B*world_size
        sim_i2t = image_feat @ text_feat_all.t()
        sim_i2t = sim_i2t / self.temp


        # text-image similarity
        # B, B*world_size
        sim_t2i = text_feat @ image_feat_all.t()
        sim_t2i = sim_t2i / self.temp


        rank = dist.get_rank()
        bs = inputs.size(0)
        itc_targets = torch.linspace(
            rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        itc_loss = (F.cross_entropy(sim_i2t, itc_targets) +
                    F.cross_entropy(sim_t2i, itc_targets)) / 2

        # prepare for itm
        output_pos = self.text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text_attn_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="fusion",
        )

        idx = torch.tensor([i.gt_video_id for i in data_samples]).view(-1, 1)
        bs = idx.size(0)
        if self.negative_all_rank:
            idxs = torch.cat(dist.all_gather(idx))
            image_feat_world = torch.cat(dist.all_gather(image_feat))
            text_feat_world = torch.cat(dist.all_gather(text_feat))
            att_mask_world = torch.cat(dist.all_gather(text_attn_mask))
            text_embeds_world = torch.cat(all_gather_with_grad(text_embeds))
            image_embeds_world = torch.cat(all_gather_with_grad(image_embeds))
        else:
            idxs = idx
            image_feat_world = image_feat.detach()
            text_feat_world = text_feat.detach()
            image_embeds_world = image_embeds
            text_embeds_world = text_embeds
            att_mask_world = text_attn_mask

        with torch.no_grad():
            # compute sample similarity
            sim_i2t = image_feat @ text_feat_world.t() / self.temp
            sim_t2i = text_feat @ image_feat_world.t() / self.temp

            mask = torch.eq(idx, idxs.t()).to(self.device)
            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_i2t.masked_fill_(mask, 0)

            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_t2i.masked_fill_(mask, 0)

        # select a negative image (from all ranks) for each text
        neg_idx = torch.multinomial(weights_t2i, 1).squeeze()
        image_embeds_neg = image_embeds_world[neg_idx]

        # select a negative text (from all ranks) for each image
        neg_idx = torch.multinomial(weights_i2t, 1).squeeze()
        text_embeds_neg = text_embeds_world[neg_idx]
        text_atts_neg = att_mask_world[neg_idx]


        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_attn_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
            mode="fusion",
        )

        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :],
                output_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )

        itm_targets = torch.ones((3 * bs, ), dtype=torch.long, 
                                 device=inputs.device)
        itm_targets[bs:] = 0
        itm_logit = self.itm_head(vl_embeddings)
        itm_loss = F.cross_entropy(itm_logit, itm_targets)

        return dict(itc_loss=itc_loss, itm_loss=itm_loss)


    def preprocess_text(self, data_samples):
        sample_item = data_samples[0]

        if sample_item is not None and 'text' in sample_item:
            if isinstance(sample_item.get('text'), (list, tuple)):
                texts = []
                for sample in data_samples:
                    texts.extend(sample.get('text'))
            elif isinstance(sample_item.get('text'), str):
                texts = [sample.get('text') for sample in data_samples]
            else:
                raise TypeError('text must be a string or a list of strings')
        else:
            return None

        # perform tokenize first if satisfied conditions
        texts = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors='pt',
        ).to(self.device)

        return texts

    def extract_feat(
        self,
        images: torch.Tensor = None,
        data_samples: List[ActionDataSample] = None,
        return_texts=True,
    ) -> Dict[str, torch.Tensor]:
        """Extract features from the input dict.

        Args:
            images (tensor, optional): The images to extract features.
                Defaults to None.
            data_samples (list, optional): The data samples containing texts
                to extract features. Defaults to None.
            return_texts (bool): Whether to return the tokenized text and the
                corresponding attention masks. Defaults to True.

        Returns:
            Tuple[torch.Tensor]: The output features.
                If multimodal_backbone is not exist, tuple of torch.Tensor
                will be returned.
        """
        if data_samples is not None:
            texts = self.preprocess_text(data_samples)
        else:
            texts = None

        assert images is not None or texts is not None, \
            'At least single modality should be passed as inputs.'

        results = {}
        if texts is not None and return_texts:
            results.update({
                'text_ids': texts.input_ids,
                'text_attn_mask': texts.attention_mask,
            })

        # extract image features
        if images is not None:
            # images = torch.ones_like(images)
            image_embeds, pooled_image_embeds = self.encode_vision(images)
            # concat temporal embeds 
            image_embeds = rearrange(image_embeds, "b t l c -> b (t l) c").contiguous()
            # image_embeds = image_embeds.unsqueeze(1)  # (bsz, 1, #frm*L, d)
            results['image_embeds'] = image_embeds
            results['image_feat'] = F.normalize(self.vision_proj(pooled_image_embeds).mean(1), dim=-1)

        # extract text features
        if texts is not None:
            texts_output = self.text_encoder(
                texts.input_ids,
                attention_mask=texts.attention_mask,
                return_dict=True,
                mode='text')

            text_embeds = texts_output.last_hidden_state
            pooled_text_feat = text_embeds[:, 0]
            results['text_embeds'] = text_embeds
            results['text_feat'] = F.normalize(self.text_proj(pooled_text_feat), dim=-1)

        return results

    # def predict(self, images, data_samples, cal_i2t=True, cal_t2i=True):
    #     feats = self.extract_feat(images, data_samples)

    #     return self.predict_all(
    #         feats, data_samples, cal_i2t=cal_i2t, cal_t2i=cal_t2i)

    def predict_all(self,
                    feats,
                    data_samples,
                    num_images=None,
                    num_texts=None,
                    cal_i2t=True,
                    cal_t2i=True):
        text_attn_mask = feats['text_attn_mask']
        image_embeds = feats.get('image_embeds', None)
        image_feat = feats['image_feat']
        text_embeds = feats['text_embeds']
        text_feat = feats['text_feat']

        num_images = num_images or image_feat.size(0)
        num_texts = num_texts or text_feat.size(0)

        image_embeds_all = all_gather_concat(image_embeds)[:num_images]
        image_feat_all = all_gather_concat(image_feat)[:num_images]
        text_feat_all = all_gather_concat(text_feat)[:num_texts]
        text_embeds_all = all_gather_concat(text_embeds)[:num_texts]
        text_attn_mask_all = all_gather_concat(text_attn_mask)[:num_texts]

        results = []
        if cal_i2t:
            result_i2t = self.compute_score_matrix_i2t(
                image_feat,
                image_embeds,
                text_feat_all,
                text_embeds_all,
                text_attn_mask_all,
            )
            results.append(
                self._get_predictions(result_i2t, data_samples, mode='i2t'))
        if cal_t2i:
            result_t2i = self.compute_score_matrix_t2i(
                image_feat_all,
                image_embeds_all,
                text_feat,
                text_embeds,
                text_attn_mask,
            )
            results.append(
                self._get_predictions(result_t2i, data_samples, mode='t2i'))
        return tuple(results)

    def compute_score_matrix_i2t(self, img_feats, img_embeds, text_feats,
                                 text_embeds, text_atts):
        """Compare the score matrix for image-to-text retrieval. Every image
        should compare to all the text features.

        Args:
            img_feats (torch.Tensor): The input img feats tensor with shape
                (M, C). M stands for numbers of samples on a single GPU.
            img_embeds (torch.Tensor): The input img embeds tensor with shape
                (M, C). M stands for numbers of samples on a single GPU.
            text_feats (torch.Tensor): The input text feats tensor with shape
                (N, C). N stands for numbers of all samples on all GPUs.
            text_ids (torch.Tensor): The input tensor with shape (N, C).
            text_atts (torch.Tensor): The input tensor with shape (N, C).

        Returns:
            torch.Tensor: Score matrix of image-to-text retrieval.
        """
        use_sim = False
        # compute i2t sim matrix
        sim_matrix_i2t = img_feats @ text_feats.t()

        score_matrix_i2t = torch.full((img_feats.size(0), text_feats.size(0)),
                                      -100.0).to(self.device)
        for i in track_on_main_process(
                range(img_feats.size(0)), 'Compute I2T scores...'):
            sims = sim_matrix_i2t[i]
            topk_sim, topk_idx = sims.topk(k=self.topk, dim=0)
            if use_sim:
                score_matrix_i2t[i, topk_idx] = topk_sim
            else:
                topk_bz = 32
                encoder_output = img_embeds[i].repeat(topk_bz, 1, 1)
                encoder_att = torch.ones(
                    encoder_output.size()[:-1], dtype=torch.long).to(self.device)
                for j in range(0, self.topk // topk_bz):
                    batch_topk = topk_idx[j*topk_bz: (j+1)*topk_bz]
                    output = self.text_encoder(
                        encoder_embeds=text_embeds[batch_topk],
                        attention_mask=text_atts[batch_topk],
                        encoder_hidden_states=encoder_output,
                        encoder_attention_mask=encoder_att,
                        return_dict=True,
                        mode='fusion'
                    )
                    score = self.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
                    score_matrix_i2t[i, batch_topk] = score
        return score_matrix_i2t

    def compute_score_matrix_t2i(self, img_feats, img_embeds, text_feats,
                                 text_embeds, text_atts):
        """Compare the score matrix for text-to-image retrieval. Every text
        should compare to all the image features.

        Args:
            img_feats (torch.Tensor): The input img feats tensor with shape
                (M, C). M stands for numbers of samples on a single GPU.
            img_embeds (torch.Tensor): The input img embeds tensor with shape
                (M, C). M stands for numbers of samples on a single GPU.
            text_feats (torch.Tensor): The input text feats tensor with shape
                (N, C). N stands for numbers of all samples on all GPUs.
            text_ids (torch.Tensor): The input tensor with shape (M, C).
            text_atts (torch.Tensor): The input tensor with shape (M, C).

        Returns:
            torch.Tensor: Score matrix of text-to-image retrieval.
        """
        use_sim = False
        # compute t2i sim matrix
        sim_matrix_t2i = text_feats @ img_feats.t()

        score_matrix_t2i = torch.full((text_feats.size(0), img_feats.size(0)),
                                      -100.0).to(self.device)
        for i in track_on_main_process(
                range(text_feats.size(0)), 'Compute T2I scores...'):
            sims = sim_matrix_t2i[i]
            topk_sim, topk_idx = sims.topk(k=self.topk, dim=0)
            if use_sim:
                score_matrix_t2i[i, topk_idx] = topk_sim
            else:
                topk_bz = 32
                for j in range(0, self.topk // topk_bz):
                    batch_topk = topk_idx[j*topk_bz: (j+1)*topk_bz]
                    encoder_output = img_embeds[batch_topk]
                    encoder_att = torch.ones(
                        encoder_output.size()[:-1], dtype=torch.long).to(self.device)
                    output = self.text_encoder(
                        encoder_embeds=text_embeds[i].repeat(topk_bz, 1, 1),
                        attention_mask=text_atts[i].repeat(topk_bz, 1),
                        encoder_hidden_states=encoder_output,
                        encoder_attention_mask=encoder_att,
                        return_dict=True,
                        mode='fusion'
                    )
                    score = self.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
                    score_matrix_t2i[i, batch_topk] = score
        return score_matrix_t2i

    def _get_predictions(self,
                         result: torch.Tensor,
                         data_samples: List[ActionDataSample],
                         mode: str = 'i2t'):
        """Post-process the output of retriever.

        Args:
            result (torch.Tensor): Score matrix of single retrieve,
                either from image or text.
            data_samples (List[ActionDataSample], optional): The annotation
                data of every samples.
            mode (str): Retrieve mode, either `i2t` for image to text, or `t2i`
                text to image. Defaults to `i2t`.

        Returns:
            List[ActionDataSample]: the raw data_samples with
                the predicted results.
        """

        # create data sample if not exists
        if data_samples is None:
            data_samples = [ActionDataSample() for _ in range(result.size(0))]
        elif mode == 't2i':
            # Process data samples to align with the num of texts.
            new_data_samples = []
            for sample in data_samples:
                if isinstance(sample.text, (list, tuple)):
                    texts = sample.text
                else:
                    texts = [sample.text]
                for i, text in enumerate(texts):
                    new_sample = ActionDataSample(text=text)
                    if 'gt_video_id' in sample:
                        new_sample.gt_label = sample.gt_video_id[i]
                    new_data_samples.append(new_sample)
            assert len(new_data_samples) == result.size(0)
            data_samples = new_data_samples
        elif mode == 'i2t':
            for sample in data_samples:
                if 'gt_text_id' in sample:
                    sample.gt_label = sample.gt_text_id
        else:
            raise ValueError(f'Type {mode} is not supported.')

        for data_sample, score in zip(data_samples, result):
            idx = score.argmax(keepdim=True).detach()

            data_sample.set_pred_score(score)
            data_sample.set_pred_label(idx)
        return data_samples
