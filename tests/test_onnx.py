import torch.nn as nn
from tools.pytorch2onnx import pytorch2onnx


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, 1)
        self.bn = nn.BatchNorm3d(2)

    def forward(self, x):
        return self.bn(self.conv(x))

    def forward_dummy(self, x):
        return (self.forward(x), )


def test_onnx_exporting():
    model = TestModel()
    # test exporting
    pytorch2onnx(model, (1, 1, 1, 1, 1), verify=True)
