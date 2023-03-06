# Directly inherit the entire recipe you want to use.
_base_ = 'mmaction::recognition/tsn/' \
         'tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'

# This line is to import your own modules.
custom_imports = dict(imports='models')

# Modify the backbone to use your own backbone.
_base_['model']['backbone'] = dict(type='ExampleNet', depth=50)
# Modify the in_channels of classifier head to fit your backbone.
_base_['model']['cls_head']['in_channels'] = 2048
