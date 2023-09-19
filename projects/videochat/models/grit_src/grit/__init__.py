from .data.datasets import grit_coco, object365, vg
from .modeling.backbone import vit
from .modeling.meta_arch import grit
from .modeling.roi_heads import grit_roi_heads


def main():
    # 这里是你的代码逻辑
    print(grit_coco, object365, vg, vit, grit, grit_roi_heads)
