from .modeling.dense_heads.centernet import CenterNet
from .modeling.meta_arch.centernet_detector import CenterNetDetector
from .modeling.roi_heads.custom_roi_heads import (CustomCascadeROIHeads,
                                                  CustomROIHeads)


def main():
    print(CenterNet, CenterNetDetector, CustomCascadeROIHeads, CustomROIHeads)
