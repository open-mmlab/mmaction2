import copy
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import auto_fp16, load_checkpoint

from mmaction.models.common import LFB
from mmaction.utils import get_root_logger

try:
    from mmdet.models.builder import SHARED_HEADS as MMDET_SHARED_HEADS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    warnings.warn('Please install mmdet to SHARED_HEADS')
    mmdet_imported = False


class NonLocalLayer(nn.Module):

    def __init__(self,
                 num_st_feat_channels,
                 num_lt_feat_channels,
                 num_latent_channels,
                 use_scale=True,
                 act_before_upsample=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 dropout_ratio=0.2):
        self.num_st_feat_channels = num_st_feat_channels
        self.num_lt_feat_channels = num_lt_feat_channels
        self.num_latent_channels = num_latent_channels
        self.use_scale = use_scale
        self.act_before_upsample = act_before_upsample
        self.dropout_ratio = dropout_ratio

        self.st_feat_conv = ConvModule(
            self.num_st_feat_channels,
            self.num_latent_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.lt_feat_conv = ConvModule(
            self.num_lt_feat_channels,
            self.num_latent_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.global_conv = ConvModule(
            self.num_lt_feat_channels,
            self.num_latent_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)

        if act_before_upsample:
            self.ln = nn.LayerNorm([num_latent_channels, 1, 1, 1])
        else:
            self.ln = nn.LayerNorm([num_st_feat_channels, 1, 1, 1])

        self.relu = nn.ReLU()

        self.out_conv = ConvModule(
            self.num_latent_channels,
            self.num_lt_feat_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, st_feat, lt_feat):
        n = st_feat.size(0)

        theta = self.st_feat_conv(st_feat)
        theta = theta.view(n, self.num_latent_channels, -1, 1,
                           1).permute(0, 2, 1).contiguous()

        phi = self.lt_feat_conv(lt_feat)
        phi = phi.view(n, self.num_latent_channels, -1, 1, 1)

        g = self.global_conv(lt_feat)
        g = g.view(n, self.num_latent_channels, -1, 1, 1)

        theta_phi = torch.matmul(theta, phi)

        if self.use_scale:
            theta_phi /= theta.shape[-1]**0.5
        p = theta_phi.softmax(dim=-1).permute(0, 2, 1)

        out = torch.matmul(g, p)
        out = out.view(n, self.num_latent_channels, -1, 1, 1)

        if self.act_before_upsample:
            out = self.relu(self.ln(out))

        out = self.out_conv(out)

        if not self.act_before_upsample:
            out = self.ln(out)

        if self.dropout_ratio > 0:
            out = self.dropout(out)
        return out


class FBONonLocal(nn.Module):

    def __init__(self,
                 num_non_local_layers=2,
                 num_st_feat_channels=2048,
                 num_lt_feat_channels=2048,
                 num_latent_channels=512,
                 st_feat_dropout_ratio=0.2,
                 lt_feat_dropout_ratio=0.2,
                 with_relu_after_nl=False):
        super().__init__()
        self.num_non_local_layers = num_non_local_layers
        self.num_st_feat_channels = num_st_feat_channels
        self.num_lt_feat_channels = num_lt_feat_channels
        self.num_latent_channels = num_latent_channels
        self.st_feat_dropout_ratio = st_feat_dropout_ratio
        self.lt_feat_dropout_ratio = lt_feat_dropout_ratio
        self.with_relu_after_nl = with_relu_after_nl

        self.st_feat_conv = nn.Conv1d(
            num_st_feat_channels,
            num_latent_channels,
            kernel_size=1,
            bias=False)
        self.lt_feat_conv = nn.Conv1d(
            num_lt_feat_channels,
            num_latent_channels,
            kernel_size=1,
            bias=False)

        if self.st_feat_dropout_ratio > 0:
            self.st_feat_dropout = nn.Dropout(self.st_feat_dropout_ratio)

        if self.lt_feat_dropout_ratio > 0:
            self.lt_feat_dropout = nn.Dropout(self.lt_feat_dropout_ratio)

        if self.with_relu_after_nl:
            self.relu = nn.ReLU()

        self.non_local_layers = []
        for idx in range(self.num_non_local_layers):
            layer_name = f'non_local_layer{idx + 1}'
            self.add_module(
                layer_name,
                NonLocalLayer(num_latent_channels, num_latent_channels,
                              num_latent_channels))
            self.non_local_layers.append(layer_name)

    def forward(self, st_feat, lt_feat):
        # prepare input
        st_feat = self.st_feat_conv(st_feat)
        if self.st_feat_dropout_ratio > 0:
            st_feat = self.st_feat_dropout(st_feat)

        lt_feat = self.lt_feat_conv(lt_feat)
        if self.lt_feat_dropout_ratio > 0:
            lt_feat = self.lt_feat_dropout(lt_feat)

        for layer_name in self.non_local_layers:
            non_local_layer = getattr(self, layer_name)
            nl_out = non_local_layer(st_feat, lt_feat)
            nl_out = st_feat + nl_out
            if self.with_relu_after_nl:
                nl_out = self.relu(nl_out)
            st_feat = nl_out
        return nl_out


class FBOAvg(nn.Module):
    pass


class FBOMax(nn.Module):
    pass


@MMDET_SHARED_HEADS.register_module()
class FBOHead(nn.Module):

    fbo_dict = {'non_local': FBONonLocal, 'avg': FBOAvg, 'max': FBOMax}

    def __init__(self,
                 lfb_cfg,
                 fbo_cfg,
                 temporal_pool_type='avg',
                 spatial_pool_type='max'):
        fbo_type = self.fbo_cfg.pop('type', 'non_local')
        assert fbo_type in self.fbo_dict
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']

        self.lfb_cfg = copy.deepcopy(lfb_cfg)
        self.fbo_cfg = copy.deepcopy(fbo_cfg)

        self.lfb = LFB(**self.lfb_cfg)
        self.fbo = self.fbo_dict[fbo_type](**self.fbo_cfg)

        # Pool by default
        if temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

    def init_weights(self, pretrained=None):
        """Initialize the weights in the module.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    @staticmethod
    def sample_lfb(self, rois, img_metas):
        inds = rois[:, 0]
        lt_feat_list = []
        for ind in inds:
            lt_feat_list.append(self.lfb[img_metas[ind]['img_key']])
        lt_feat = torch.stack(lt_feat_list, dim=0)
        # [N, num_lfb_channels, window_size * max_num_feat_per_step]
        lt_feat = lt_feat.permute(0, 2, 1).contiguous()
        return lt_feat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    @auto_fp16()
    def forward(self, x, rois, img_metas):
        # feature size [N, C, T, H, W]
        n, c, _, _, _ = x.shape

        # [N, C, 1, 1, 1]
        x = self.temporal_pool(x)
        st_feat = self.spatial_pool(x)
        identity = st_feat

        # [N, C, window_size * num_feat_per_step, 1, 1, 1]
        lt_feat = self.sample_lfb(rois, img_metas)

        fbo_feat = self.fbo(st_feat, lt_feat)

        out = torch.cat([identity, fbo_feat], dim=1)
        return out
