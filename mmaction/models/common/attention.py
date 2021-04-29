import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import (ATTENTION, BaseModule, MultiheadAttention,
                      build_norm_layer)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """From TimeSformer.

    For temporary use only.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """From TimeSformer.

    For temporary use only.
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


@ATTENTION.register_module()
class _MultiheadAttention(MultiheadAttention):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 drop_path=0.,
                 init_cfg=None,
                 **kwargs):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop=attn_drop,
            proj_drop=drop_path,
            init_cfg=init_cfg,
            **kwargs)
        self.dropout = DropPath(drop_path)


@ATTENTION.register_module()
class DividedTemporalAttentionWithNorm(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 drop_path=0.1,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.drop_path = DropPath(drop_path)
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)
        self.temporal_fc = nn.Linear(self.embed_dims, self.embed_dims)

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        assert residual is None, (
            'Cannot apply pre-norm with DividedTemporalAttentionWithNorm')

        init_cls_token = query[:, 0, :].unsqueeze(1)
        identity = query_t = query[:, 1:, :]

        # query_t [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_t.size()
        p, t = pt // self.num_frames, self.num_frames

        # res_temporal [batch_size * num_patches, num_frames, embed_dims]
        query_t = self.norm(query_t.view(b * p, t, m))
        res_temporal = self.drop_path(self.attn(query_t, query_t, query_t)[0])
        res_temporal = self.temporal_fc(res_temporal)

        # res_temporal [batch_size, num_patches * num_frames, embed_dims]
        res_temporal = res_temporal.view(b, p * t, m)

        # ret_value [batch_size, num_patches * num_frames + 1, embed_dims]
        return torch.cat((init_cls_token, identity + res_temporal), 1)


@ATTENTION.register_module()
class DividedSpatialAttentionWithNorm(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 drop_path=0.1,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.drop_path = DropPath(drop_path)
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        assert residual is None, (
            'Cannot apply pre-norm with DividedTemporalAttentionWithNorm')

        identity = query
        init_cls_token = query[:, 0, :].unsqueeze(1)
        query_s = query[:, 1:, :]

        # query_s [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_s.size()
        p, t = pt // self.num_frames, self.num_frames

        # cls_token [batch_size * num_frames, 1, embed_dims]
        cls_token = init_cls_token.repeat(1, t, 1).view(b * t, m).unsqueeze(1)

        # query_s [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)
        query_s = torch.cat((cls_token, query_s), 1)
        query_s = self.norm(query_s)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = self.drop_path(self.attn(query_s, query_s, query_s)[0])

        # cls_token [batch_size, 1, embed_dims]
        cls_token = res_spatial[:, 0, :].view(b, t, m)
        cls_token = torch.mean(cls_token, 1, True)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = rearrange(
            res_spatial[:, 0:, :], '(b t) p m -> b (p t) m', p=p, t=t)
        res_spatial = torch.cat((cls_token, res_spatial), 1)

        return identity + res_spatial
