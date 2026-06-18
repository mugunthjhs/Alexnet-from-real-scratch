import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error: mean((pred - target)^2)"""
    diff = pred - target
    return (diff * diff).mean()
