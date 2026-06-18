import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class Dropout(Module):
    """
    Inverted dropout: randomly zeros activations with probability p during training.
    Scales surviving activations by 1/(1-p) so test-time pass-through is unchanged.
    No-op during eval mode.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype) / (1.0 - self.p)
        result = x.data * mask

        out = Tensor(result, requires_grad=x.requires_grad, is_leaf=False)

        if x.requires_grad:
            _x = x

            def _backward():
                if out.grad is None:
                    return
                grad = out.grad * mask
                _x.grad = grad if _x.grad is None else _x.grad + grad

            out._backward = _backward
            out._prev = {x}
            out._op = "dropout"

        return out

    def __repr__(self):
        return f"Dropout(p={self.p})"
