import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class ReLU(Module):
    """Rectified Linear Unit: f(x) = max(0, x)"""

    def forward(self, x: Tensor) -> Tensor:
        result = np.maximum(0.0, x.data)
        out = Tensor(result, requires_grad=x.requires_grad, is_leaf=False)

        if x.requires_grad:
            _x = x

            def _backward():
                if out.grad is None:
                    return
                grad = out.grad * (result > 0).astype(x.dtype)
                _x.grad = grad if _x.grad is None else _x.grad + grad

            out._backward = _backward
            out._prev = {x}
            out._op = "relu"

        return out

    def __repr__(self):
        return "ReLU()"
