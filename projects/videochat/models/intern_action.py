#!/usr/bin/env python
import os
from collections import OrderedDict

import torch
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
from torch import nn
from torch.nn import MultiheadAttention

MODEL_PATH = '../'
_MODELS = {
    'ViT-B/16': os.path.join(MODEL_PATH, 'vit_b16.pth'),
    'ViT-L/14': os.path.join(MODEL_PATH, 'vit_l14.pth'),
    'ViT-L/14_336': os.path.join(MODEL_PATH, 'vit_l14_336.pth'),
}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Local_MHRA(nn.Module):

    def __init__(self, d_model, dw_reduction=1.5, pos_kernel_size=3):
        super().__init__()

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(
                re_d_model,
                re_d_model,
                kernel_size=(pos_kernel_size, 1, 1),
                stride=(1, 1, 1),
                padding=(padding, 0, 0),
                groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        print('Init zero for Conv in pos_emb')
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x):
        return self.pos_embed(x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self,
                 d_model,
                 n_head,
                 attn_mask=None,
                 drop_path=0.0,
                 dw_reduction=1.5,
                 no_lmhra=False,
                 double_lmhra=True):
        super().__init__()

        self.n_head = n_head
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        print(f'Drop path rate: {drop_path}')

        self.no_lmhra = no_lmhra
        self.double_lmhra = double_lmhra
        print(f'No L_MHRA: {no_lmhra}')
        print(f'Double L_MHRA: {double_lmhra}')
        if not no_lmhra:
            self.lmhra1 = Local_MHRA(d_model, dw_reduction=dw_reduction)
            if double_lmhra:
                self.lmhra2 = Local_MHRA(d_model, dw_reduction=dw_reduction)

        # spatial
        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, T=8, use_checkpoint=False):
        # x: 1+HW, NT, C
        if not self.no_lmhra:
            # Local MHRA
            tmp_x = x[1:, :, :]
            L, NT, C = tmp_x.shape
            N = NT // T
            H = W = int(L**0.5)
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0,
                                                      1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x))
            tmp_x = tmp_x.view(N, C, T,
                               L).permute(3, 0, 2,
                                          1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # MHSA
        if use_checkpoint:
            attn_out = checkpoint.checkpoint(self.attention, self.ln_1(x))
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attention(self.ln_1(x)))
        # Local MHRA
        if not self.no_lmhra and self.double_lmhra:
            tmp_x = x[1:, :, :]
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0,
                                                      1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra2(tmp_x))
            tmp_x = tmp_x.view(N, C, T,
                               L).permute(3, 0, 2,
                                          1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # FFN
        if use_checkpoint:
            mlp_out = checkpoint.checkpoint(self.mlp, self.ln_2(x))
            x = x + self.drop_path(mlp_out)
        else:
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Extractor(nn.Module):

    def __init__(
        self,
        d_model,
        n_head,
        attn_mask=None,
        mlp_factor=4.0,
        dropout=0.0,
        drop_path=0.0,
    ):
        super().__init__()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        print(f'Drop path rate: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_mlp)),
                         ('gelu', QuickGELU()),
                         ('dropout', nn.Dropout(dropout)),
                         ('c_proj', nn.Linear(d_mlp, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

        # zero init
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.out_proj.weight, 0.)
        nn.init.constant_(self.attn.out_proj.bias, 0.)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[-1].weight, 0.)
        nn.init.constant_(self.mlp[-1].bias, 0.)

    def attention(self, x, y):
        d_model = self.ln_1.weight.size(0)
        q = (x @ self.attn.in_proj_weight[:d_model].T
             ) + self.attn.in_proj_bias[:d_model]

        k = (y @ self.attn.in_proj_weight[d_model:-d_model].T
             ) + self.attn.in_proj_bias[d_model:-d_model]
        v = (y @ self.attn.in_proj_weight[-d_model:].T
             ) + self.attn.in_proj_bias[-d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        k = k.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        v = v.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5))

        aff = aff.softmax(dim=-1)
        out = aff @ v
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        return out

    def forward(self, x, y):
        x = x + self.drop_path(self.attention(self.ln_1(x), self.ln_3(y)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        width,
        layers,
        heads,
        attn_mask=None,
        backbone_drop_path_rate=0.,
        use_checkpoint=False,
        checkpoint_num=[0],
        t_size=8,
        dw_reduction=2,
        no_lmhra=False,
        double_lmhra=True,
        return_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        n_layers=12,
        n_dim=768,
        n_head=12,
        mlp_factor=4.0,
        drop_path_rate=0.,
        mlp_dropout=[
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        ],
        cls_dropout=0.5,
        num_classes=400,
    ):
        super().__init__()
        self.T = t_size
        self.return_list = return_list
        # backbone
        b_dpr = [
            x.item()
            for x in torch.linspace(0, backbone_drop_path_rate, layers)
        ]
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                attn_mask,
                drop_path=b_dpr[i],
                dw_reduction=dw_reduction,
                no_lmhra=no_lmhra,
                double_lmhra=double_lmhra,
            ) for i in range(layers)
        ])
        # checkpoint
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.n_layers = n_layers
        print(f'Use checkpoint: {self.use_checkpoint}')
        print(f'Checkpoint number: {self.checkpoint_num}')

        # global block
        assert n_layers == len(return_list)
        if n_layers > 0:
            self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, n_dim))
            self.dpe = nn.ModuleList([
                nn.Conv3d(
                    n_dim,
                    n_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    groups=n_dim) for i in range(n_layers)
            ])
            for m in self.dpe:
                nn.init.constant_(m.bias, 0.)
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, n_layers)
            ]
            self.dec = nn.ModuleList([
                Extractor(
                    n_dim,
                    n_head,
                    mlp_factor=mlp_factor,
                    dropout=mlp_dropout[i],
                    drop_path=dpr[i],
                ) for i in range(n_layers)
            ])
            self.balance = nn.Parameter(torch.zeros((n_dim)))
            self.sigmoid = nn.Sigmoid()
        # projection
        self.proj = nn.Sequential(
            nn.LayerNorm(n_dim),
            nn.Dropout(cls_dropout),
            nn.Linear(n_dim, num_classes),
        )

    def forward(self, x):
        T_down = self.T
        L, NT, C = x.shape
        N = NT // T_down
        H = W = int((L - 1)**0.5)

        if self.n_layers > 0:
            cls_token = self.temporal_cls_token.repeat(1, N, 1)

        j = -1
        for i, resblock in enumerate(self.resblocks):
            if self.use_checkpoint and i < self.checkpoint_num[0]:
                x = resblock(x, self.T, use_checkpoint=True)
            else:
                x = resblock(x, T_down)
            if i in self.return_list:
                j += 1
                tmp_x = x.clone()
                tmp_x = tmp_x.view(L, N, T_down, C)
                # dpe
                _, tmp_feats = tmp_x[:1], tmp_x[1:]
                tmp_feats = tmp_feats.permute(1, 3, 2,
                                              0).reshape(N, C, T_down, H, W)
                tmp_feats = self.dpe[j](tmp_feats).view(
                    N, C, T_down, L - 1).permute(3, 0, 2, 1).contiguous()
                tmp_x[1:] = tmp_x[1:] + tmp_feats
                # global block
                tmp_x = tmp_x.permute(2, 0, 1, 3).flatten(0, 1)  # T * L, N, C
                cls_token = self.dec[j](cls_token, tmp_x)

        if self.n_layers > 0:
            weight = self.sigmoid(self.balance)
            residual = x.view(L, N, T_down, C)[0].mean(1)  # L, N, T, C
            return self.proj((1 - weight) * cls_token[0, :, :] +
                             weight * residual)
        else:
            residual = x.view(L, N, T_down, C)[0].mean(1)  # L, N, T, C
            return self.proj(residual)


class VisionTransformer(nn.Module):

    def __init__(
        self,
        # backbone
        input_resolution,
        patch_size,
        width,
        layers,
        heads,
        output_dim,
        backbone_drop_path_rate=0.,
        use_checkpoint=False,
        checkpoint_num=[0],
        t_size=8,
        kernel_size=3,
        dw_reduction=1.5,
        temporal_downsample=True,
        no_lmhra=- False,
        double_lmhra=True,
        # global block
        return_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        n_layers=12,
        n_dim=768,
        n_head=12,
        mlp_factor=4.0,
        drop_path_rate=0.,
        mlp_dropout=[
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        ],
        cls_dropout=0.5,
        num_classes=400,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        padding = (kernel_size - 1) // 2
        if temporal_downsample:
            self.conv1 = nn.Conv3d(
                3,
                width, (kernel_size, patch_size, patch_size),
                (2, patch_size, patch_size), (padding, 0, 0),
                bias=False)
            t_size = t_size // 2
        else:
            self.conv1 = nn.Conv3d(
                3,
                width, (1, patch_size, patch_size),
                (1, patch_size, patch_size), (0, 0, 0),
                bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width,
            layers,
            heads,
            dw_reduction=dw_reduction,
            backbone_drop_path_rate=backbone_drop_path_rate,
            use_checkpoint=use_checkpoint,
            checkpoint_num=checkpoint_num,
            t_size=t_size,
            no_lmhra=no_lmhra,
            double_lmhra=double_lmhra,
            return_list=return_list,
            n_layers=n_layers,
            n_dim=n_dim,
            n_head=n_head,
            mlp_factor=mlp_factor,
            drop_path_rate=drop_path_rate,
            mlp_dropout=mlp_dropout,
            cls_dropout=cls_dropout,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C)

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        out = self.transformer(x)
        return out


def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 2:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: '
                  f'{k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim)
    model.load_state_dict(state_dict, strict=False)


def intern_action_b16(
    pretrained=True,
    use_checkpoint=False,
    checkpoint_num=[0],
    t_size=16,
    dw_reduction=1.5,
    backbone_drop_path_rate=0.,
    temporal_downsample=True,
    no_lmhra=False,
    double_lmhra=True,
    return_list=[8, 9, 10, 11],
    n_layers=4,
    n_dim=768,
    n_head=12,
    mlp_factor=4.0,
    drop_path_rate=0.,
    mlp_dropout=[0.5, 0.5, 0.5, 0.5],
    cls_dropout=0.5,
    num_classes=400,
):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
        t_size=t_size,
        dw_reduction=dw_reduction,
        backbone_drop_path_rate=backbone_drop_path_rate,
        temporal_downsample=temporal_downsample,
        no_lmhra=no_lmhra,
        double_lmhra=double_lmhra,
        return_list=return_list,
        n_layers=n_layers,
        n_dim=n_dim,
        n_head=n_head,
        mlp_factor=mlp_factor,
        drop_path_rate=drop_path_rate,
        mlp_dropout=mlp_dropout,
        cls_dropout=cls_dropout,
        num_classes=num_classes,
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS['ViT-B/16'], map_location='cpu')
        load_state_dict(model, state_dict)
    return model.eval()


def intern_action_l14(
    pretrained=True,
    use_checkpoint=False,
    checkpoint_num=[0],
    t_size=16,
    dw_reduction=1.5,
    backbone_drop_path_rate=0.,
    temporal_downsample=True,
    no_lmhra=False,
    double_lmhra=True,
    return_list=[20, 21, 22, 23],
    n_layers=4,
    n_dim=1024,
    n_head=16,
    mlp_factor=4.0,
    drop_path_rate=0.,
    mlp_dropout=[0.5, 0.5, 0.5, 0.5],
    cls_dropout=0.5,
    num_classes=400,
):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
        t_size=t_size,
        dw_reduction=dw_reduction,
        backbone_drop_path_rate=backbone_drop_path_rate,
        temporal_downsample=temporal_downsample,
        no_lmhra=no_lmhra,
        double_lmhra=double_lmhra,
        return_list=return_list,
        n_layers=n_layers,
        n_dim=n_dim,
        n_head=n_head,
        mlp_factor=mlp_factor,
        drop_path_rate=drop_path_rate,
        mlp_dropout=mlp_dropout,
        cls_dropout=cls_dropout,
        num_classes=num_classes,
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS['ViT-L/14'], map_location='cpu')
        load_state_dict(model, state_dict)
    return model.eval()


def intern_action_l14_336(
    pretrained=True,
    use_checkpoint=False,
    checkpoint_num=[0],
    t_size=16,
    dw_reduction=1.5,
    backbone_drop_path_rate=0.,
    no_temporal_downsample=True,
    no_lmhra=False,
    double_lmhra=True,
    return_list=[20, 21, 22, 23],
    n_layers=4,
    n_dim=1024,
    n_head=16,
    mlp_factor=4.0,
    drop_path_rate=0.,
    mlp_dropout=[0.5, 0.5, 0.5, 0.5],
    cls_dropout=0.5,
    num_classes=400,
):
    model = VisionTransformer(
        input_resolution=336,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
        t_size=t_size,
        dw_reduction=dw_reduction,
        backbone_drop_path_rate=backbone_drop_path_rate,
        no_temporal_downsample=no_temporal_downsample,
        no_lmhra=no_lmhra,
        double_lmhra=double_lmhra,
        return_list=return_list,
        n_layers=n_layers,
        n_dim=n_dim,
        n_head=n_head,
        mlp_factor=mlp_factor,
        drop_path_rate=drop_path_rate,
        mlp_dropout=mlp_dropout,
        cls_dropout=cls_dropout,
        num_classes=num_classes,
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS['ViT-L/14_336'], map_location='cpu')
        load_state_dict(model, state_dict)
    return model.eval()


if __name__ == '__main__':
    import time

    import numpy as np
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 16

    model = intern_action_l14(
        pretrained=False,
        t_size=num_frames,
        backbone_drop_path_rate=0.,
        drop_path_rate=0.,
        dw_reduction=1.5,
        no_lmhra=False,
        temporal_downsample=True,
        return_list=[
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
        ],
        mlp_dropout=[0.5] * 16,
        n_layers=16)
    print(model)

    flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, 224, 224))
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time() - s)
