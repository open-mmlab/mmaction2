# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/OpenGVLab/efficient-video-recognition

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger
from mmengine.model.weight_init import constant_init, kaiming_init, normal_init
from mmengine.runner import load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmaction.registry import MODELS


def load_weights_clip(load_path: str) -> Dict[str, torch.Tensor]:
    clip_model = torch.jit.load(load_path, map_location='cpu')
    clip_model = clip_model.visual
    src_state_dict = clip_model.state_dict()
    src_state_dict = dict((k, v.float()) for k, v in src_state_dict.items())

    dst_state_dict = {}

    dst_state_dict['cls_token'] = src_state_dict['class_embedding']
    dst_state_dict['pos_embed'] = src_state_dict['positional_embedding']
    dst_state_dict['patch_embed.proj.weight'] = src_state_dict[
        'conv1.weight'].flatten(1)
    dst_state_dict['patch_embed.proj.bias'] = torch.zeros(
        [src_state_dict['conv1.weight'].size(0)])

    dst_state_dict['ln_pre.weight'] = src_state_dict['ln_pre.weight']
    dst_state_dict['ln_pre.bias'] = src_state_dict['ln_pre.bias']

    block_idx = 0
    while True:
        src_prefix = 'transformer.resblocks.%d.' % block_idx
        dst_prefix = 'blocks.%d.' % block_idx

        src_block_state_dict = dict((k[len(src_prefix):], v)
                                    for k, v in src_state_dict.items()
                                    if k.startswith(src_prefix))
        if len(src_block_state_dict) == 0:
            break

        dst_block_state_dict = {}
        feat_dim = src_block_state_dict['ln_1.weight'].size(0)

        for i, dst_name in enumerate(('q', 'k', 'v')):
            dst_block_state_dict['attn.%s_proj.weight' %
                                 dst_name] = src_block_state_dict[
                                     'attn.in_proj_weight'][feat_dim *
                                                            i:feat_dim *
                                                            (i + 1)]
            dst_block_state_dict['attn.%s_proj.bias' %
                                 dst_name] = src_block_state_dict[
                                     'attn.in_proj_bias'][feat_dim *
                                                          i:feat_dim * (i + 1)]

        dst_block_state_dict['attn.out_proj.weight'] = src_block_state_dict[
            'attn.out_proj.weight']
        dst_block_state_dict['attn.out_proj.bias'] = src_block_state_dict[
            'attn.out_proj.bias']

        dst_block_state_dict['mlp.fc1.weight'] = src_block_state_dict[
            'mlp.c_fc.weight']
        dst_block_state_dict['mlp.fc1.bias'] = src_block_state_dict[
            'mlp.c_fc.bias']
        dst_block_state_dict['mlp.fc2.weight'] = src_block_state_dict[
            'mlp.c_proj.weight']
        dst_block_state_dict['mlp.fc2.bias'] = src_block_state_dict[
            'mlp.c_proj.bias']

        dst_block_state_dict['norm1.weight'] = src_block_state_dict[
            'ln_1.weight']
        dst_block_state_dict['norm1.bias'] = src_block_state_dict['ln_1.bias']
        dst_block_state_dict['norm2.weight'] = src_block_state_dict[
            'ln_2.weight']
        dst_block_state_dict['norm2.bias'] = src_block_state_dict['ln_2.bias']

        dst_state_dict.update(
            dict((dst_prefix + k, v) for k, v in dst_block_state_dict.items()))
        block_idx += 1

    return dst_state_dict


weight_loader_fn_dict = {
    'clip': load_weights_clip,
}


class QuickGELU(nn.Module):
    """from official CLIP repo"""

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16, from official CLIP repo."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Attention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
        self,
        q_in_dim: int,
        k_in_dim: int,
        v_in_dim: int,
        qk_proj_dim: int,
        v_proj_dim: int,
        num_heads: int,
        out_dim: int,
        return_all_features: bool = False,
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        self.return_all_features = return_all_features
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0)
        assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1)
        assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk**0.5), k)
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        if self.return_all_features:
            return dict(q=q, k=k, v=v, aff=aff, out=out)
        else:
            return out


class PatchEmbed2D(nn.Module):

    def __init__(
        self,
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels

        self.proj = nn.Linear(np.prod(patch_size) * in_channels, embed_dim)

    def _initialize_weights(self, x):
        nn.init.kaiming_normal_(self.proj.weight, 0.)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        pH, pW = self.patch_size

        assert C == self.in_channels and H % pH == 0 and W % pW == 0

        x = x.view(B, C, H // pH, pH, W // pW,
                   pW).permute(0, 2, 4, 1, 3, 5).flatten(3).flatten(1, 2)
        x = self.proj(x)

        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
        return_all_features: bool = False,
    ):
        super().__init__()

        self.return_all_features = return_all_features

        self.attn = Attention(
            q_in_dim=in_feature_dim,
            k_in_dim=in_feature_dim,
            v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim,
            v_proj_dim=qkv_dim,
            num_heads=num_heads,
            out_dim=in_feature_dim,
            return_all_features=return_all_features,
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
                ('act', act()),
                ('dropout', nn.Dropout(mlp_dropout)),
                ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
            ]))

        self.norm1 = LayerNorm(in_feature_dim)
        self.norm2 = LayerNorm(in_feature_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor):
        if self.return_all_features:
            ret_dict = {}

            x_norm = self.norm1(x)
            attn_out = self.attn(x_norm, x_norm, x_norm)
            ret_dict['q'] = attn_out['q']
            ret_dict['k'] = attn_out['k']
            ret_dict['v'] = attn_out['v']
            ret_dict['attn_out'] = attn_out['out']
            x = x + attn_out['out']

            x = x + self.mlp(self.norm2(x))
            ret_dict['out'] = x

            return ret_dict

        else:
            x_norm = self.norm1(x)
            x = x + self.attn(x_norm, x_norm, x_norm)
            x = x + self.mlp(self.norm2(x))

            return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
    ):
        super().__init__()

        self.attn = Attention(
            q_in_dim=in_feature_dim,
            k_in_dim=in_feature_dim,
            v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim,
            v_proj_dim=qkv_dim,
            num_heads=num_heads,
            out_dim=in_feature_dim,
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
                ('act', act()),
                ('dropout', nn.Dropout(mlp_dropout)),
                ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
            ]))

        self.norm1 = LayerNorm(in_feature_dim)
        self.norm2 = LayerNorm(in_feature_dim)
        self.norm3 = LayerNorm(in_feature_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        y_norm = self.norm3(y)
        x = x + self.attn(self.norm1(x), y_norm, y_norm)
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer2D(nn.Module):

    def __init__(
        self,
        feature_dim: int = 768,
        input_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,
        act: nn.Module = QuickGELU,
        return_all_features: bool = False,
        ln_pre: bool = False,
    ):
        super().__init__()

        self.return_all_features = return_all_features

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, embed_dim=feature_dim)
        self.num_patches = np.prod(
            [x // y for x, y in zip(input_size, patch_size)]) + 1

        self.cls_token = nn.Parameter(torch.zeros([feature_dim]))
        self.pos_embed = nn.Parameter(
            torch.zeros([self.num_patches, feature_dim]))

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                in_feature_dim=feature_dim,
                qkv_dim=feature_dim,
                num_heads=num_heads,
                mlp_factor=mlp_factor,
                act=act,
                return_all_features=return_all_features,
            ) for _ in range(num_layers)
        ])

        if ln_pre:
            self.ln_pre = LayerNorm(feature_dim)
        else:
            self.ln_pre = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor):
        dtype = self.patch_embed.proj.weight.dtype
        x = x.to(dtype)

        x = self.patch_embed(x)
        x = torch.cat(
            [self.cls_token.view(1, 1, -1).repeat(x.size(0), 1, 1), x], dim=1)
        x = x + self.pos_embed

        x = self.ln_pre(x)

        if self.return_all_features:
            all_features = []
            for blk in self.blocks:
                x = blk(x)
                all_features.append(x)
                x = x['out']
            return all_features

        else:
            for blk in self.blocks:
                x = blk(x)
            return x


def model_to_fp16(model: VisionTransformer2D):

    def _module_to_fp16(m: nn.Module):
        if isinstance(m, (nn.Linear, )):
            m.half()

    model.apply(_module_to_fp16)

    model.pos_embed.data = model.pos_embed.data.half()
    model.cls_token.data = model.cls_token.data.half()


vit_presets = {
    'ViT-B/16':
    dict(
        feature_dim=768,
        input_size=(224, 224),
        patch_size=(16, 16),
        num_heads=12,
        num_layers=12,
        mlp_factor=4.0,
        ln_pre=True,
    ),
    'ViT-L/14':
    dict(
        feature_dim=1024,
        input_size=(224, 224),
        patch_size=(14, 14),
        num_heads=16,
        num_layers=24,
        mlp_factor=4.0,
        ln_pre=True,
    ),
}


class TemporalCrossAttention(nn.Module):

    def __init__(
            self,
            spatial_size: Tuple[int, int] = (14, 14),
            feature_dim: int = 768,
    ):
        super().__init__()

        self.spatial_size = spatial_size

        w_size = np.prod([x * 2 - 1 for x in spatial_size])
        self.w1 = nn.Parameter(torch.zeros([w_size, feature_dim]))
        self.w2 = nn.Parameter(torch.zeros([w_size, feature_dim]))

        idx_tensor = torch.zeros([np.prod(spatial_size) for _ in (0, 1)],
                                 dtype=torch.long)
        for q in range(np.prod(spatial_size)):
            qi, qj = q // spatial_size[1], q % spatial_size[1]
            for k in range(np.prod(spatial_size)):
                ki, kj = k // spatial_size[1], k % spatial_size[1]
                i_offs = qi - ki + spatial_size[0] - 1
                j_offs = qj - kj + spatial_size[1] - 1
                idx_tensor[q, k] = i_offs * (spatial_size[1] * 2 - 1) + j_offs
        self.idx_tensor = idx_tensor

    def forward_half(self, q: torch.Tensor, k: torch.Tensor,
                     w: torch.Tensor) -> torch.Tensor:
        q, k = q[:, :, 1:], k[:, :, 1:]  # remove cls token

        assert q.size() == k.size()
        assert q.size(2) == np.prod(self.spatial_size)

        attn = torch.einsum('ntqhd,ntkhd->ntqkh', q / (q.size(-1)**0.5), k)
        attn = attn.softmax(dim=-2).mean(dim=-1)  # L, L, N, T

        self.idx_tensor = self.idx_tensor.to(w.device)
        w_unroll = w[self.idx_tensor]  # L, L, C
        ret = torch.einsum('ntqk,qkc->ntqc', attn, w_unroll)

        return ret

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        N, T, L, H, D = q.size()
        assert L == np.prod(self.spatial_size) + 1

        ret = torch.zeros([N, T, L, self.w1.size(-1)], device='cuda')
        ret[:, 1:, 1:, :] += self.forward_half(q[:, 1:, :, :, :],
                                               k[:, :-1, :, :, :], self.w1)
        ret[:, :-1, 1:, :] += self.forward_half(q[:, :-1, :, :, :],
                                                k[:, 1:, :, :, :], self.w2)

        return ret


class EVLDecoder(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        spatial_size: Tuple[int, int] = (14, 14),
        num_layers: int = 4,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.enable_temporal_conv = enable_temporal_conv
        self.enable_temporal_pos_embed = enable_temporal_pos_embed
        self.enable_temporal_cross_attention = enable_temporal_cross_attention
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads,
                                    mlp_factor, mlp_dropout)
            for _ in range(num_layers)
        ])

        if enable_temporal_conv:
            self.temporal_conv = nn.ModuleList([
                nn.Conv1d(
                    in_feature_dim,
                    in_feature_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=in_feature_dim) for _ in range(num_layers)
            ])
        if enable_temporal_pos_embed:
            self.temporal_pos_embed = nn.ParameterList([
                nn.Parameter(torch.zeros([num_frames, in_feature_dim]))
                for _ in range(num_layers)
            ])
        if enable_temporal_cross_attention:
            self.cross_attention = nn.ModuleList([
                TemporalCrossAttention(spatial_size, in_feature_dim)
                for _ in range(num_layers)
            ])

        self.cls_token = nn.Parameter(torch.zeros([in_feature_dim]))

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, in_features: List[Dict[str, torch.Tensor]]):
        N, T, L, C = in_features[0]['out'].size()
        assert len(in_features) == self.num_layers
        x = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)

        for i in range(self.num_layers):
            frame_features = in_features[i]['out']

            if self.enable_temporal_conv:
                feat = in_features[i]['out']
                feat = feat.permute(0, 2, 3,
                                    1).contiguous().flatten(0,
                                                            1)  # N * L, C, T
                feat = self.temporal_conv[i](feat)
                feat = feat.view(N, L, C,
                                 T).permute(0, 3, 1,
                                            2).contiguous()  # N, T, L, C
                frame_features += feat

            if self.enable_temporal_pos_embed:
                frame_features += self.temporal_pos_embed[i].view(1, T, 1, C)

            # 复杂度TxLxL
            if self.enable_temporal_cross_attention:
                frame_features += self.cross_attention[i](in_features[i]['q'],
                                                          in_features[i]['k'])

            frame_features = frame_features.flatten(1, 2)  # N, T * L, C
            # 先用T个X在L的维度与feature frame做cross-attention, TxL复杂度
            # 再在T上SA, TxT复杂度
            # 整体复杂度Tx(L+T)
            # X: Nx1xTxC 在 L 上做SA

            # 先试一下T上mean做ssv2分类？

            # 复杂度1xTxL
            x = self.decoder_layers[i](x, frame_features)

        return x


@MODELS.register_module()
class EVL(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        backbone_name: str = 'ViT-B/16',
        backbone_type: str = 'clip',
        backbone_path: str = '',
        backbone_mode: str = 'frozen_fp16',
        decoder_num_layers: int = 4,
        decoder_qkv_dim: int = 768,
        decoder_num_heads: int = 12,
        decoder_mlp_factor: float = 4.0,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        decoder_mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.decoder_num_layers = decoder_num_layers
        self.num_frames = num_frames

        backbone_config = self._create_backbone(backbone_name, backbone_type,
                                                backbone_path, backbone_mode)
        backbone_feature_dim = backbone_config['feature_dim']
        backbone_spatial_size = tuple(x // y for x, y in zip(
            backbone_config['input_size'], backbone_config['patch_size']))

        self.decoder = EVLDecoder(
            num_frames=num_frames,
            spatial_size=backbone_spatial_size,
            num_layers=decoder_num_layers,
            in_feature_dim=backbone_feature_dim,
            qkv_dim=decoder_qkv_dim,
            num_heads=decoder_num_heads,
            mlp_factor=decoder_mlp_factor,
            enable_temporal_conv=enable_temporal_conv,
            enable_temporal_pos_embed=enable_temporal_pos_embed,
            enable_temporal_cross_attention=enable_temporal_cross_attention,
            mlp_dropout=decoder_mlp_dropout,
        )
        self.layer_norm = nn.LayerNorm(backbone_feature_dim)

    def init_weights(self):
        pass

    def _create_backbone(
        self,
        backbone_name: str,
        backbone_type: str,
        backbone_path: str,
        backbone_mode: str,
    ) -> dict:
        weight_loader_fn = weight_loader_fn_dict[backbone_type]
        state_dict = weight_loader_fn(backbone_path)

        backbone = VisionTransformer2D(
            return_all_features=True, **vit_presets[backbone_name])
        backbone.load_state_dict(
            state_dict, strict=True
        )  # weight_loader_fn is expected to strip unused parameters

        assert backbone_mode in ['finetune', 'freeze_fp16', 'freeze_fp32']

        if backbone_mode == 'finetune':
            self.backbone = backbone
        else:
            backbone.eval().requires_grad_(False)
            if backbone_mode == 'freeze_fp16':
                model_to_fp16(backbone)
            self.backbone = [backbone]  # avoid backbone parameter registration

        return vit_presets[backbone_name]

    def _get_backbone(self, x):
        if isinstance(self.backbone, list):
            # freeze backbone
            self.backbone[0] = self.backbone[0].to(x.device)
            return self.backbone[0]
        else:
            # finetune bakbone
            return self.backbone

    def forward(self, x: torch.Tensor):
        backbone = self._get_backbone(x)

        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        T = self.num_frames
        B = x.size(0) // T
        features = backbone(x)[-self.decoder_num_layers:]
        features = [
            dict((k, v.float().view(B, T,
                                    *v.size()[1:])) for k, v in x.items())
            for x in features
        ]

        x = self.decoder(features).squeeze(1)
        return x
