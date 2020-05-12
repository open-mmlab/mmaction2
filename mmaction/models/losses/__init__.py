from .base import BaseWeightedLoss
from .cross_entropy_loss import BCELossWithLogits, CrossEntropyLoss
from .nll_loss import NLLLoss

__all__ = [
    'BaseWeightedLoss', 'CrossEntropyLoss', 'NLLLoss', 'BCELossWithLogits'
]
