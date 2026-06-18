import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class Softmax(Module):
    """
    Numerically stable softmax along a given axis.
    Note: for classification losses prefer raw logits + cross_entropy, which
    fuses softmax and log internally for better numerical stability.
    """

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        shifted = x.data - x.data.max(axis=self.axis, keepdims=True)
        exp_x = np.exp(shifted)
        s = exp_x / exp_x.sum(axis=self.axis, keepdims=True)

        out = Tensor(s, requires_grad=x.requires_grad, is_leaf=False)

        if x.requires_grad:
            _x, _s, _axis = x, s, self.axis

            def _backward():
                if out.grad is None:
                    return
                # Jacobian-vector product: s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
                dot = (out.grad * _s).sum(axis=_axis, keepdims=True)
                grad = _s * (out.grad - dot)
                _x.grad = grad if _x.grad is None else _x.grad + grad

            out._backward = _backward
            out._prev = {x}
            out._op = "softmax"

        return out

    def __repr__(self):
        return f"Softmax(axis={self.axis})"
