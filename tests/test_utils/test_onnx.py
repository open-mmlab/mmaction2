# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import torch.nn as nn

from tools.deployment.pytorch2onnx import _convert_batchnorm, pytorch2onnx


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, 1)
        self.bn = nn.SyncBatchNorm(2)

    def forward(self, x):
        return self.bn(self.conv(x))

    def forward_dummy(self, x):
        out = self.bn(self.conv(x))
        return (out, )


def test_onnx_exporting():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = osp.join(tmpdir, 'tmp.onnx')
        model = TestModel()
        model = _convert_batchnorm(model)
        # test exporting
        if hasattr(model, 'forward_dummy'):
            model.forward = model.forward_dummy
        pytorch2onnx(model, (2, 1, 1, 1, 1), output_file=out_file, verify=True)
