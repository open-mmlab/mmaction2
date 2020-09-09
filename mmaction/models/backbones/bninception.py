import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from ...utils import get_root_logger
from ..registry import BACKBONES


@BACKBONES.register_module()
class BNInception(nn.Module):
    """BNInception backbone.

    Args:
        pretrained (str | None): Name of pretrained model. Default: None.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        bn_frozen (bool): Whether to frozen BN layers. Default: False.
        partial_bn (bool): Whether to use partial bn. Default: False.
    """

    def __init__(self,
                 pretrained=None,
                 norm_eval=False,
                 bn_frozen=False,
                 partial_bn=False):
        super(BNInception, self).__init__()

        self.pretrained = pretrained
        self.norm_eval = norm_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn

        inplace = True
        self.conv1_7x7_s2 = nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.9, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d((3, 3),
                                         stride=(2, 2),
                                         dilation=(1, 1),
                                         ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(
            64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.9, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(
            64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(
            192, eps=1e-05, momentum=0.9, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool2d((3, 3),
                                         stride=(2, 2),
                                         dilation=(1, 1),
                                         ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv2d(
            192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_3x3_reduce = nn.Conv2d(
            192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_3x3 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_reduce = nn.Conv2d(
            192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_double_3x3_1 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_2 = nn.Conv2d(
            96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3a_pool_proj = nn.Conv2d(
            192, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_3x3_reduce = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_3x3 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_reduce = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_double_3x3_1 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_2 = nn.Conv2d(
            96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3b_pool_proj = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_3x3_reduce = nn.Conv2d(
            320, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_3x3 = nn.Conv2d(
            128, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_3x3_bn = nn.BatchNorm2d(
            160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_3x3 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_reduce = nn.Conv2d(
            320, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_double_3x3_1 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_2 = nn.Conv2d(
            96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool2d((3, 3),
                                              stride=(2, 2),
                                              dilation=(1, 1),
                                              ceil_mode=True)
        self.inception_4a_1x1 = nn.Conv2d(
            576, 224, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_1x1_bn = nn.BatchNorm2d(
            224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_3x3_reduce = nn.Conv2d(
            576, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_3x3 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_3x3_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_reduce = nn.Conv2d(
            576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_double_3x3_1 = nn.Conv2d(
            96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_2 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4a_pool_proj = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_pool_proj_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv2d(
            576, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_1x1_bn = nn.BatchNorm2d(
            192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_3x3_reduce = nn.Conv2d(
            576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_3x3 = nn.Conv2d(
            96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_3x3_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_reduce = nn.Conv2d(
            576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_double_3x3_1 = nn.Conv2d(
            96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_2 = nn.Conv2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4b_pool_proj = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_pool_proj_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv2d(
            576, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_1x1_bn = nn.BatchNorm2d(
            160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_3x3_reduce = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_3x3 = nn.Conv2d(
            128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_3x3_bn = nn.BatchNorm2d(
            160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_reduce = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_double_3x3_1 = nn.Conv2d(
            128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(
            160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_2 = nn.Conv2d(
            160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(
            160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4c_pool_proj = nn.Conv2d(
            576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_pool_proj_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv2d(
            608, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_1x1_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_3x3_reduce = nn.Conv2d(
            608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_3x3 = nn.Conv2d(
            128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_bn = nn.BatchNorm2d(
            192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_reduce = nn.Conv2d(
            608, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(
            160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_double_3x3_1 = nn.Conv2d(
            160, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(
            192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_2 = nn.Conv2d(
            192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(
            192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4d_pool_proj = nn.Conv2d(
            608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_pool_proj_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_3x3_reduce = nn.Conv2d(
            608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_3x3 = nn.Conv2d(
            128, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_3x3_bn = nn.BatchNorm2d(
            192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_reduce = nn.Conv2d(
            608, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(
            192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_double_3x3_1 = nn.Conv2d(
            192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_2 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool2d((3, 3),
                                              stride=(2, 2),
                                              dilation=(1, 1),
                                              ceil_mode=True)
        self.inception_5a_1x1 = nn.Conv2d(
            1056, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_1x1_bn = nn.BatchNorm2d(
            352, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_3x3_reduce = nn.Conv2d(
            1056, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(
            192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_3x3 = nn.Conv2d(
            192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_3x3_bn = nn.BatchNorm2d(
            320, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_reduce = nn.Conv2d(
            1056, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(
            160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_double_3x3_1 = nn.Conv2d(
            160, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(
            224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_2 = nn.Conv2d(
            224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(
            224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.AvgPool2d(
            3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_5a_pool_proj = nn.Conv2d(
            1056, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_pool_proj_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv2d(
            1024, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_1x1_bn = nn.BatchNorm2d(
            352, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_3x3_reduce = nn.Conv2d(
            1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(
            192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_3x3 = nn.Conv2d(
            192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_bn = nn.BatchNorm2d(
            320, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_reduce = nn.Conv2d(
            1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(
            192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_double_3x3_1 = nn.Conv2d(
            192, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(
            224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_2 = nn.Conv2d(
            224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(
            224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool2d((3, 3),
                                              stride=(1, 1),
                                              padding=(1, 1),
                                              dilation=(1, 1),
                                              ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv2d(
            1024, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_pool_proj_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def conv1(self, x):
        conv1_7x7_s2_out = self.conv1_7x7_s2(x)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        return conv1_7x7_s2_bn_out

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        conv1_7x7_s2_bn_out = self.conv1(x)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_7x7_s2_bn_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(
            conv2_3x3_reduce_out)
        conv2_3x3_out = self.conv2_3x3(conv2_3x3_reduce_bn_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_3x3_bn_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(
            inception_3a_1x1_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(
            pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(
            inception_3a_3x3_reduce_out)
        inception_3a_3x3_out = self.inception_3a_3x3(
            inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(
            inception_3a_3x3_out)
        inception_3a_double_3x3_reduce_out = (
            self.inception_3a_double_3x3_reduce(pool2_3x3_s2_out))
        inception_3a_double_3x3_reduce_bn_out = (
            self.inception_3a_double_3x3_reduce_bn(
                inception_3a_double_3x3_reduce_out))
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(
            inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(
            inception_3a_double_3x3_1_out)
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(
            inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(
            inception_3a_double_3x3_2_out)
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(
            inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(
            inception_3a_pool_proj_out)
        inception_3a_output_out = torch.cat([
            inception_3a_1x1_bn_out, inception_3a_3x3_bn_out,
            inception_3a_double_3x3_2_bn_out, inception_3a_pool_proj_bn_out
        ], 1)
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(
            inception_3b_1x1_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(
            inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(
            inception_3b_3x3_reduce_out)
        inception_3b_3x3_out = self.inception_3b_3x3(
            inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(
            inception_3b_3x3_out)
        inception_3b_double_3x3_reduce_out = (
            self.inception_3b_double_3x3_reduce(inception_3a_output_out))
        inception_3b_double_3x3_reduce_bn_out = (
            self.inception_3b_double_3x3_reduce_bn(
                inception_3b_double_3x3_reduce_out))
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(
            inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(
            inception_3b_double_3x3_1_out)
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(
            inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(
            inception_3b_double_3x3_2_out)
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(
            inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(
            inception_3b_pool_proj_out)
        inception_3b_output_out = torch.cat([
            inception_3b_1x1_bn_out, inception_3b_3x3_bn_out,
            inception_3b_double_3x3_2_bn_out, inception_3b_pool_proj_bn_out
        ], 1)
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(
            inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(
            inception_3c_3x3_reduce_out)
        inception_3c_3x3_out = self.inception_3c_3x3(
            inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(
            inception_3c_3x3_out)
        inception_3c_double_3x3_reduce_out = (
            self.inception_3c_double_3x3_reduce(inception_3b_output_out))
        inception_3c_double_3x3_reduce_bn_out = (
            self.inception_3c_double_3x3_reduce_bn(
                inception_3c_double_3x3_reduce_out))
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(
            inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(
            inception_3c_double_3x3_1_out)
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(
            inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(
            inception_3c_double_3x3_2_out)
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat([
            inception_3c_3x3_bn_out, inception_3c_double_3x3_2_bn_out,
            inception_3c_pool_out
        ], 1)
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(
            inception_4a_1x1_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(
            inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(
            inception_4a_3x3_reduce_out)
        inception_4a_3x3_out = self.inception_4a_3x3(
            inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(
            inception_4a_3x3_out)
        inception_4a_double_3x3_reduce_out = (
            self.inception_4a_double_3x3_reduce(inception_3c_output_out))
        inception_4a_double_3x3_reduce_bn_out = (
            self.inception_4a_double_3x3_reduce_bn(
                inception_4a_double_3x3_reduce_out))
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(
            inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(
            inception_4a_double_3x3_1_out)
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(
            inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(
            inception_4a_double_3x3_2_out)
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(
            inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(
            inception_4a_pool_proj_out)
        inception_4a_output_out = torch.cat([
            inception_4a_1x1_bn_out, inception_4a_3x3_bn_out,
            inception_4a_double_3x3_2_bn_out, inception_4a_pool_proj_bn_out
        ], 1)
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(
            inception_4b_1x1_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(
            inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(
            inception_4b_3x3_reduce_out)
        inception_4b_3x3_out = self.inception_4b_3x3(
            inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(
            inception_4b_3x3_out)
        inception_4b_double_3x3_reduce_out = (
            self.inception_4b_double_3x3_reduce(inception_4a_output_out))
        inception_4b_double_3x3_reduce_bn_out = (
            self.inception_4b_double_3x3_reduce_bn(
                inception_4b_double_3x3_reduce_out))
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(
            inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(
            inception_4b_double_3x3_1_out)
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(
            inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(
            inception_4b_double_3x3_2_out)
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(
            inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(
            inception_4b_pool_proj_out)
        inception_4b_output_out = torch.cat([
            inception_4b_1x1_bn_out, inception_4b_3x3_bn_out,
            inception_4b_double_3x3_2_bn_out, inception_4b_pool_proj_bn_out
        ], 1)
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(
            inception_4c_1x1_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(
            inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(
            inception_4c_3x3_reduce_out)
        inception_4c_3x3_out = self.inception_4c_3x3(
            inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(
            inception_4c_3x3_out)
        inception_4c_double_3x3_reduce_out = (
            self.inception_4c_double_3x3_reduce(inception_4b_output_out))
        inception_4c_double_3x3_reduce_bn_out = (
            self.inception_4c_double_3x3_reduce_bn(
                inception_4c_double_3x3_reduce_out))
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(
            inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(
            inception_4c_double_3x3_1_out)
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(
            inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(
            inception_4c_double_3x3_2_out)
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(
            inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(
            inception_4c_pool_proj_out)
        inception_4c_output_out = torch.cat([
            inception_4c_1x1_bn_out, inception_4c_3x3_bn_out,
            inception_4c_double_3x3_2_bn_out, inception_4c_pool_proj_bn_out
        ], 1)
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(
            inception_4d_1x1_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(
            inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(
            inception_4d_3x3_reduce_out)
        inception_4d_3x3_out = self.inception_4d_3x3(
            inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(
            inception_4d_3x3_out)
        inception_4d_double_3x3_reduce_out = (
            self.inception_4d_double_3x3_reduce(inception_4c_output_out))
        inception_4d_double_3x3_reduce_bn_out = (
            self.inception_4d_double_3x3_reduce_bn(
                inception_4d_double_3x3_reduce_out))
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(
            inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(
            inception_4d_double_3x3_1_out)
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(
            inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(
            inception_4d_double_3x3_2_out)
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(
            inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(
            inception_4d_pool_proj_out)
        inception_4d_output_out = torch.cat([
            inception_4d_1x1_bn_out, inception_4d_3x3_bn_out,
            inception_4d_double_3x3_2_bn_out, inception_4d_pool_proj_bn_out
        ], 1)
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(
            inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(
            inception_4e_3x3_reduce_out)
        inception_4e_3x3_out = self.inception_4e_3x3(
            inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(
            inception_4e_3x3_out)
        inception_4e_double_3x3_reduce_out = (
            self.inception_4e_double_3x3_reduce(inception_4d_output_out))
        inception_4e_double_3x3_reduce_bn_out = (
            self.inception_4e_double_3x3_reduce_bn(
                inception_4e_double_3x3_reduce_out))
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(
            inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(
            inception_4e_double_3x3_1_out)
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(
            inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(
            inception_4e_double_3x3_2_out)
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat([
            inception_4e_3x3_bn_out, inception_4e_double_3x3_2_bn_out,
            inception_4e_pool_out
        ], 1)
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(
            inception_5a_1x1_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(
            inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(
            inception_5a_3x3_reduce_out)
        inception_5a_3x3_out = self.inception_5a_3x3(
            inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(
            inception_5a_3x3_out)
        inception_5a_double_3x3_reduce_out = (
            self.inception_5a_double_3x3_reduce(inception_4e_output_out))
        inception_5a_double_3x3_reduce_bn_out = (
            self.inception_5a_double_3x3_reduce_bn(
                inception_5a_double_3x3_reduce_out))
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(
            inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(
            inception_5a_double_3x3_1_out)
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(
            inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(
            inception_5a_double_3x3_2_out)
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(
            inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(
            inception_5a_pool_proj_out)
        inception_5a_output_out = torch.cat([
            inception_5a_1x1_bn_out, inception_5a_3x3_bn_out,
            inception_5a_double_3x3_2_bn_out, inception_5a_pool_proj_bn_out
        ], 1)
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(
            inception_5b_1x1_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(
            inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(
            inception_5b_3x3_reduce_out)
        inception_5b_3x3_out = self.inception_5b_3x3(
            inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(
            inception_5b_3x3_out)
        inception_5b_double_3x3_reduce_out = (
            self.inception_5b_double_3x3_reduce(inception_5a_output_out))
        inception_5b_double_3x3_reduce_bn_out = (
            self.inception_5b_double_3x3_reduce_bn(
                inception_5b_double_3x3_reduce_out))
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(
            inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(
            inception_5b_double_3x3_1_out)
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(
            inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(
            inception_5b_double_3x3_2_out)
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(
            inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(
            inception_5b_pool_proj_out)
        inception_5b_output_out = torch.cat([
            inception_5b_1x1_bn_out, inception_5b_3x3_bn_out,
            inception_5b_double_3x3_2_bn_out, inception_5b_pool_proj_bn_out
        ], 1)
        return inception_5b_output_out

    def _partial_bn(self):
        logger = get_root_logger()
        logger.info('Freezing BatchNorm2D except the first one.')
        count_bn = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                count_bn += 1
                if count_bn >= 2:
                    m.eval()
                    # shutdown update in frozen mode
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if mode and self.partial_bn:
            self._partial_bn()
