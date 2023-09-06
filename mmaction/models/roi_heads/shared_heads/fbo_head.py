# Copyright (c) OpenMMLab. All rights reserved.
# Copyrigho (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.runner import load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from .lfb import LFB


class NonLocalLayer(nn.Module):
    """Non-local layer used in `FBONonLocal` is a variation of the vanilla non-
    local block.

    Args:
        st_feat_channels (int): Channels of short-term features.
        lt_feat_channels (int): Channels of long-term features.
        latent_channels (int): Channels of latent features.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(latent_channels)`. Default: True.
        pre_activate (bool): Whether to use the activation function before
            upsampling. Default: False.
        conv_cfg (Dict | None): The config dict for convolution layers. If
            not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (Dict | None): he config dict for normalization layers.
            Default: None.
        dropout_ratio (float, optional): Probability of dropout layer.
            Default: 0.2.
        zero_init_out_conv (bool): Whether to use zero initialization for
            out_conv. Default: False.
    """

    def __init__(self,
                 st_feat_channels,
                 lt_feat_channels,
                 latent_channels,
                 num_st_feat,
                 num_lt_feat,
                 use_scale=True,
                 pre_activate=True,
                 pre_activate_with_ln=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 dropout_ratio=0.2,
                 zero_init_out_conv=False):
        super().__init__()
        if conv_cfg is None:
            conv_cfg = dict(type='Conv3d')
        self.st_feat_channels = st_feat_channels
        self.lt_feat_channels = lt_feat_channels
        self.latent_channels = latent_channels
        self.num_st_feat = num_st_feat
        self.num_lt_feat = num_lt_feat
        self.use_scale = use_scale
        self.pre_activate = pre_activate
        self.pre_activate_with_ln = pre_activate_with_ln
        self.dropout_ratio = dropout_ratio
        self.zero_init_out_conv = zero_init_out_conv

        self.st_feat_conv = ConvModule(
            self.st_feat_channels,
            self.latent_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.lt_feat_conv = ConvModule(
            self.lt_feat_channels,
            self.latent_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.global_conv = ConvModule(
            self.lt_feat_channels,
            self.latent_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        if pre_activate:
            self.ln = nn.LayerNorm([latent_channels, num_st_feat, 1, 1])
        else:
            self.ln = nn.LayerNorm([st_feat_channels, num_st_feat, 1, 1])

        self.relu = nn.ReLU()

        self.out_conv = ConvModule(
            self.latent_channels,
            self.st_feat_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {pretrained}')
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
            if self.zero_init_out_conv:
                constant_init(self.out_conv, 0, bias=0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, st_feat, lt_feat):
        """Defines the computation performed at every call."""
        n, c = st_feat.size(0), self.latent_channels
        num_st_feat, num_lt_feat = self.num_st_feat, self.num_lt_feat

        theta = self.st_feat_conv(st_feat)
        theta = theta.view(n, c, num_st_feat)

        phi = self.lt_feat_conv(lt_feat)
        phi = phi.view(n, c, num_lt_feat)

        g = self.global_conv(lt_feat)
        g = g.view(n, c, num_lt_feat)

        # (n, num_st_feat, c), (n, c, num_lt_feat)
        # -> (n, num_st_feat, num_lt_feat)
        theta_phi = torch.matmul(theta.permute(0, 2, 1), phi)
        if self.use_scale:
            theta_phi /= c**0.5

        p = theta_phi.softmax(dim=-1)

        # (n, c, num_lt_feat), (n, num_lt_feat, num_st_feat)
        # -> (n, c, num_st_feat, 1, 1)
        out = torch.matmul(g, p.permute(0, 2, 1)).view(n, c, num_st_feat, 1, 1)

        # If need to activate it before out_conv, use relu here, otherwise
        # use relu outside the non local layer.
        if self.pre_activate:
            if self.pre_activate_with_ln:
                out = self.ln(out)
            out = self.relu(out)

        out = self.out_conv(out)

        if not self.pre_activate:
            out = self.ln(out)
        if self.dropout_ratio > 0:
            out = self.dropout(out)

        return out


class FBONonLocal(nn.Module):
    """Non local feature bank operator.

    Args:
        st_feat_channels (int): Channels of short-term features.
        lt_feat_channels (int): Channels of long-term features.
        latent_channels (int): Channels of latent features.
        num_st_feat (int): Number of short-term roi features.
        num_lt_feat (int): Number of long-term roi features.
        num_non_local_layers (int): Number of non-local layers, which is
            at least 1. Default: 2.
        st_feat_dropout_ratio (float): Probability of dropout layer for
            short-term features. Default: 0.2.
        lt_feat_dropout_ratio (float): Probability of dropout layer for
            long-term features. Default: 0.2.
        pre_activate (bool): Whether to use the activation function before
            upsampling in non local layers. Default: True.
        zero_init_out_conv (bool): Whether to use zero initialization for
            out_conv in NonLocalLayer. Default: False.
    """

    def __init__(self,
                 st_feat_channels,
                 lt_feat_channels,
                 latent_channels,
                 num_st_feat,
                 num_lt_feat,
                 num_non_local_layers=2,
                 st_feat_dropout_ratio=0.2,
                 lt_feat_dropout_ratio=0.2,
                 pre_activate=True,
                 zero_init_out_conv=False,
                 **kwargs):
        super().__init__()
        assert num_non_local_layers >= 1, (
            'At least one non_local_layer is needed.')
        self.st_feat_channels = st_feat_channels
        self.lt_feat_channels = lt_feat_channels
        self.latent_channels = latent_channels
        self.num_st_feat = num_st_feat
        self.num_lt_feat = num_lt_feat
        self.num_non_local_layers = num_non_local_layers
        self.st_feat_dropout_ratio = st_feat_dropout_ratio
        self.lt_feat_dropout_ratio = lt_feat_dropout_ratio
        self.pre_activate = pre_activate
        self.zero_init_out_conv = zero_init_out_conv

        self.st_feat_conv = nn.Conv3d(
            st_feat_channels, latent_channels, kernel_size=1)
        self.lt_feat_conv = nn.Conv3d(
            lt_feat_channels, latent_channels, kernel_size=1)

        if self.st_feat_dropout_ratio > 0:
            self.st_feat_dropout = nn.Dropout(self.st_feat_dropout_ratio)

        if self.lt_feat_dropout_ratio > 0:
            self.lt_feat_dropout = nn.Dropout(self.lt_feat_dropout_ratio)

        if not self.pre_activate:
            self.relu = nn.ReLU()

        self.non_local_layers = []
        for idx in range(self.num_non_local_layers):
            layer_name = f'non_local_layer_{idx + 1}'
            self.add_module(
                layer_name,
                NonLocalLayer(
                    latent_channels,
                    latent_channels,
                    latent_channels,
                    num_st_feat,
                    num_lt_feat,
                    pre_activate=self.pre_activate,
                    zero_init_out_conv=self.zero_init_out_conv))
            self.non_local_layers.append(layer_name)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            kaiming_init(self.st_feat_conv)
            kaiming_init(self.lt_feat_conv)
            for layer_name in self.non_local_layers:
                non_local_layer = getattr(self, layer_name)
                non_local_layer.init_weights(pretrained=pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, st_feat, lt_feat):
        """Defines the computation performed at every call."""
        # prepare st_feat
        st_feat = self.st_feat_conv(st_feat)
        if self.st_feat_dropout_ratio > 0:
            st_feat = self.st_feat_dropout(st_feat)

        # prepare lt_feat
        lt_feat = self.lt_feat_conv(lt_feat)
        if self.lt_feat_dropout_ratio > 0:
            lt_feat = self.lt_feat_dropout(lt_feat)

        # fuse short-term and long-term features in NonLocal Layer
        for layer_name in self.non_local_layers:
            identity = st_feat
            non_local_layer = getattr(self, layer_name)
            nl_out = non_local_layer(st_feat, lt_feat)
            nl_out = identity + nl_out
            if not self.pre_activate:
                nl_out = self.relu(nl_out)
            st_feat = nl_out

        return nl_out


class FBOAvg(nn.Module):
    """Avg pool feature bank operator."""

    def __init__(self, **kwargs):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, None, None))

    def init_weights(self, pretrained=None):
        # FBOAvg has no parameters to be initialized.
        pass

    def forward(self, st_feat, lt_feat):
        out = self.avg_pool(lt_feat)
        return out


class FBOMax(nn.Module):
    """Max pool feature bank operator."""

    def __init__(self, **kwargs):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool3d((1, None, None))

    def init_weights(self, pretrained=None):
        """FBOMax has no parameters to be initialized."""
        pass

    def forward(self, st_feat, lt_feat):
        """Defines the computation performed at every call."""
        out = self.max_pool(lt_feat)
        return out


class FBOHead(nn.Module):
    """Feature Bank Operator Head.

    Add feature bank operator for the spatiotemporal detection model to fuse
    short-term features and long-term features.
    Args:
        lfb_cfg (Dict): The config dict for LFB which is used to sample
            long-term features.
        fbo_cfg (Dict): The config dict for feature bank operator (FBO). The
            type of fbo is also in the config dict and supported fbo type is
            `fbo_dict`.
        temporal_pool_type (str): The temporal pool type. Choices are 'avg' or
            'max'. Default: 'avg'.
        spatial_pool_type (str): The spatial pool type. Choices are 'avg' or
            'max'. Default: 'max'.
    """

    fbo_dict = {'non_local': FBONonLocal, 'avg': FBOAvg, 'max': FBOMax}

    def __init__(self,
                 lfb_cfg,
                 fbo_cfg,
                 temporal_pool_type='avg',
                 spatial_pool_type='max'):
        super().__init__()
        fbo_type = fbo_cfg.pop('type', 'non_local')
        assert fbo_type in FBOHead.fbo_dict
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
                Default: None.
        """
        self.fbo.init_weights(pretrained=pretrained)

    def sample_lfb(self, rois, img_metas):
        """Sample long-term features for each ROI feature."""
        inds = rois[:, 0].type(torch.int64)
        lt_feat_list = []
        for ind in inds:
            lt_feat_list.append(self.lfb[img_metas[ind]['img_key']])
        lt_feat = torch.stack(lt_feat_list, dim=0)
        # [N, lfb_channels, window_size * max_num_feat_per_step]
        lt_feat = lt_feat.permute(0, 2, 1).contiguous()
        return lt_feat.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x, rois, img_metas, **kwargs):
        """Defines the computation performed at every call."""
        # [N, C, 1, 1, 1]
        st_feat = self.temporal_pool(x)
        st_feat = self.spatial_pool(st_feat)
        identity = st_feat

        # [N, C, window_size * num_feat_per_step, 1, 1]
        lt_feat = self.sample_lfb(rois, img_metas).to(st_feat.device)

        fbo_feat = self.fbo(st_feat, lt_feat)

        out = torch.cat([identity, fbo_feat], dim=1)
        return out
