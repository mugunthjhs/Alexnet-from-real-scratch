"""Losses module - loss functions"""

from .cross_entropy import CrossEntropyLoss, SoftmaxCrossEntropy

__all__ = [
    'CrossEntropyLoss',
    'SoftmaxCrossEntropy',
]
