import torch.nn as nn

from .conv2plus1d import Conv2plus1d

conv_cfg = {
    'Conv': nn.Conv2d,
    'Conv1d': nn.Conv1d,
    'Conv3d': nn.Conv3d,
    'Conv(2+1)d': Conv2plus1d
    # 'ConvWS': ConvWS2d,
    # TODO: octave conv
}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: Args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): Created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        if not (isinstance(cfg, dict) and 'type' in cfg):
            raise TypeError('cfg must be a dict containing the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
