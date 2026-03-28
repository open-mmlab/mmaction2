"""
 * Tag2Text
 * Written by Xinyu Huang
"""
import json
import math
import os
import warnings
from typing import List
from urllib.parse import urlparse

import numpy as np
import torch
from timm.models.hub import download_cached_file
from torch import nn
from transformers import BertTokenizer

from projects.videochat.models.med import (BertConfig, BertLMHeadModel,
                                           BertModel, logger)
from projects.videochat.models.swin_transformer import (
    SwinTransformer, interpolate_relative_pos_embed)
from projects.videochat.models.vit import (VisionTransformer,
                                           interpolate_pos_embed)
from projects.videochat.tag_class.tag_class import tra_array

warnings.filterwarnings('ignore')


def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)


# delete some tags that may disturb captioning
delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]

# adjust thresholds for some tags
# default threshold: 0.68
# 2701: "person"; 2828: "man"; 1167: "woman";
tag_thrshold = {2701: 0.7, 2828: 0.7, 1167: 0.7}


class Tag2Text_Caption(nn.Module):

    def __init__(
        self,
        med_config='configs/med_config.json',
        image_size=384,
        vit='base',
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        prompt='a picture of ',
        threshold=0.68,
    ):
        """
        Args: med_config (str): path for the mixture of encoder-decoder
        model's configuration file image_size (int): input image size vit (
        str): model size of vision transformer
        """
        super().__init__()

        if vit == 'swin_b':
            if image_size == 224:
                vision_config_path = 'configs/swin/config_swinB_224.json'
            elif image_size == 384:
                vision_config_path = 'configs/swin/config_swinB_384.json'
            vision_config = read_json(vision_config_path)
            assert image_size == vision_config['image_res']
            # assert config['patch_size'] == 32
            vision_width = vision_config['vision_width']

            self.visual_encoder = SwinTransformer(
                img_size=vision_config['image_res'],
                patch_size=4,
                in_chans=3,
                embed_dim=vision_config['embed_dim'],
                depths=vision_config['depths'],
                num_heads=vision_config['num_heads'],
                window_size=vision_config['window_size'],
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.0,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True,
                use_checkpoint=False)

        else:
            self.visual_encoder, vision_width = create_vit(
                vit, image_size, vit_grad_ckpt, vit_ckpt_layer)

        self.tokenizer = init_tokenizer()

        # create the decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = 768
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        # create encoder
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.tag_encoder = BertModel(
            config=encoder_config, add_pooling_layer=False)

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        self.threshold = threshold
        num_features = 768
        self.num_class = 3429

        q2l_config = BertConfig.from_json_file('configs/q2l_config.json')
        q2l_config.encoder_width = vision_width
        self.vision_multi = BertModel(
            config=q2l_config, add_pooling_layer=False)
        self.vision_multi.resize_token_embeddings(len(self.tokenizer))
        self.label_embed = nn.Embedding(self.num_class, q2l_config.hidden_size)
        self.fc = GroupWiseLinear(self.num_class, num_features, bias=True)
        self.del_selfattention()

        tie_encoder_decoder_weights(self.tag_encoder, self.vision_multi, '',
                                    ' ')
        self.tag_array = tra_array

        self.class_threshold = torch.ones(self.num_class) * self.threshold
        for key, value in tag_thrshold.items():
            self.class_threshold[key] = value

    def del_selfattention(self):
        del self.vision_multi.embeddings
        for layer in self.vision_multi.encoder.layer:
            del layer.attention

    def generate(self,
                 image,
                 sample=False,
                 num_beams=3,
                 max_length=30,
                 min_length=10,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 tag_input=None,
                 return_tag_predict=False):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # ==============generate tag==============#
        if tag_input is None:
            image_spatial_embeds = image_embeds[:, 1:, :]
            # image_cls_embeds = image_embeds[:, 0, :]

            bs = image_spatial_embeds.shape[0]
            label_embed = self.label_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            mlr_tagembedding = self.vision_multi(
                encoder_embeds=label_embed,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
                mode='mlr',
            )

            logits = self.fc(mlr_tagembedding[0])

            # targets = torch.where(torch.sigmoid(logits) > self.threshold ,
            # torch.tensor(1.0).to(image.device), torch.zeros(
            # self.num_class).to(image.device))
            targets = torch.where(
                torch.sigmoid(logits) > self.class_threshold.to(image.device),
                torch.tensor(1.0).to(image.device),
                torch.zeros(self.num_class).to(image.device))

            tag = targets.cpu().numpy()
            tag[:, delete_tag_index] = 0
            bs = image.size(0)
            tag_input = []
            for b in range(bs):
                index = np.argwhere(tag[b] == 1)
                token = self.tag_array[index].squeeze(axis=1)
                tag_input.append(' | '.join(token))
        # ========================================#

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
            image_atts = image_atts.repeat_interleave(num_beams, dim=0)
            tag_input_temp = []
            for tag in tag_input:
                for i in range(num_beams):
                    tag_input_temp.append(tag)
            tag_input = tag_input_temp

        tag_input_tokenzier = self.tokenizer(
            tag_input,
            padding='max_length',
            truncation=True,
            max_length=40,
            return_tensors='pt').to(image.device)
        encoder_input_ids = tag_input_tokenzier.input_ids
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        output_tagembedding = self.tag_encoder(
            encoder_input_ids,
            attention_mask=tag_input_tokenzier.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(
            prompt, return_tensors='pt').input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            model_kwargs = {
                'encoder_hidden_states': output_tagembedding.last_hidden_state,
                'encoder_attention_mask': None
            }
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                **model_kwargs)
        else:
            # beam search
            model_kwargs = {
                'encoder_hidden_states': output_tagembedding.last_hidden_state,
                'encoder_attention_mask': None
            }
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        if return_tag_predict:
            if sample:
                return captions, tag_input
            else:
                return captions, tag_input[0:int(len(tag_input) / num_beams)]
        return captions

    def generate_sublists(self,
                          image,
                          sample=False,
                          num_beams=3,
                          max_length=30,
                          min_length=10,
                          top_p=0.9,
                          repetition_penalty=1.0,
                          tag_input=None,
                          return_tag_predict=False):
        n = image.shape[0]  # total number of images
        captions = []
        tags = []
        # iterate over image sub-tensors of size 10 or less
        for i in range(0, n, 50):
            image_subtensor = image[i:i + 50]
            # get subtensor of size 10 or less
            # call original generate function on image subtensor
            sublist_captions, sublist_tags = self.generate(
                image_subtensor,
                sample=sample,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                tag_input=tag_input,
                return_tag_predict=return_tag_predict)

            # append sublist captions to overall captions
            captions.extend(sublist_captions)
            tags.extend(sublist_tags)

        return captions, tags


def tag2text_caption(pretrained='', **kwargs):
    model = Tag2Text_Caption(**kwargs)
    if pretrained:
        if kwargs['vit'] == 'swin_b':
            model, msg = load_checkpoint_swinbase(model, pretrained, kwargs)
        else:
            model, msg = load_checkpoint(model, pretrained)
        # print('vit:',kwargs['vit'])
        # print('msg_v2',msg)
    return model


def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module,
                                base_model_prefix: str, skip_key: str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f'{decoder.__class__} and {encoder.__class__} are not equal. In '
            f'this case make sure that all encoder weights are correctly '
            f'initialized. ')

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f'{decoder_pointer} and {encoder_pointer} have to be of type ' \
           f'torch.nn.Module '
        if hasattr(decoder_pointer, 'weight') and skip_key not in module_name:
            assert hasattr(encoder_pointer, 'weight')
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, 'bias'):
                assert hasattr(encoder_pointer, 'bias')
                encoder_pointer.bias = decoder_pointer.bias
            # print(module_name+' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                    len(encoder_modules) > 0
            ), f'Encoder module {encoder_pointer} does not match decoder ' \
               f'module {decoder_pointer} '

            all_encoder_weights = set([
                module_name + '/' + sub_name
                for sub_name in encoder_modules.keys()
            ])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(
                            decoder_modules[decoder_name],
                            type(encoder_modules[encoder_name])) and \
                            len(encoder_modules) != len(decoder_modules):
                        # this can happen if the name corresponds to the
                        # position in a list module list of layers in this
                        # case the decoder has added a cross-attention that
                        # the encoder does not have thus skip this step and
                        # subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        'Max depth of recursive function '
                        '`tie_encoder_to_decoder` reached. It seems that '
                        'there is a circular dependency between two or more '
                        '`nn.Modules` of your model. ')
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + '/' + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + '/' + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix,
                                       uninitialized_encoder_weights, skip_key)


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(
        '/mnt/data.coronaryct.1/ZhuYichen/Ask-Anything/model/bert-base'
        '-uncased/',
        local_files_only=True)
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit,
               image_size,
               use_grad_checkpointing=False,
               ckpt_layer=0,
               drop_path_rate=0):
    assert vit in ['base', 'large'], 'vit parameter must be base or large'
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=12,
            num_heads=12,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0 or drop_path_rate)
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=24,
            num_heads=16,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0.1 or drop_path_rate)
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ('http', 'https')


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(
            url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(
        state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(
            state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg


def load_checkpoint_swinbase(model, url_or_filename, kwargs):
    if kwargs['image_size'] == 224:
        vision_config_path = 'configs/swin/config_swinB_224.json'
    elif kwargs['image_size'] == 384:
        vision_config_path = 'configs/swin/config_swinB_384.json'
    elif kwargs['image_size'] == 480:
        vision_config_path = 'configs/swin/config_swinB_480.json'
    elif kwargs['image_size'] == 576:
        vision_config_path = 'configs/swin/config_swinB_576.json'
    elif kwargs['image_size'] == 608:
        vision_config_path = 'configs/swin/config_swinB_608.json'
    window_size = read_json(vision_config_path)['window_size']
    # print('--------------')
    # print(url_or_filename)
    # print('--------------')
    if is_url(url_or_filename):
        cached_file = download_cached_file(
            url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    for k in list(state_dict.keys()):
        if 'relative_position_bias_table' in k:
            dst_num_pos = (2 * window_size - 1)**2
            state_dict[k] = interpolate_relative_pos_embed(
                state_dict[k], dst_num_pos, param_name=k)
        elif ('relative_position_index' in k) or ('attn_mask' in k):
            del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg
