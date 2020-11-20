from mmcv.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
RECOGNIZERS = Registry('recognizer')
LOSSES = Registry('loss')
LOCALIZERS = Registry('localizer')
DETECTORS = Registry('detector')
ROI_EXTRACTORS = Registry('roi_extractors')
