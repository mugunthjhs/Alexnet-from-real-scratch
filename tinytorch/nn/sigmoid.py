import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class Sigmoid(Module):
    """Sigmoid activation: f(x) = 1 / (1 + exp(-x))

    Uses numerically stable computation with clipping to prevent overflow.
    """

    def forward(self, x: Tensor) -> Tensor:
        # Clip input for numerical stability: sigmoid(x) ≈ 0 for x < -500, ≈ 1 for x > 500
        x_clipped = x.clip(min=-500.0, max=500.0)
        # Compute sigmoid = 1 / (1 + e^(-x))
        neg_exp = (-x_clipped).exp()
        sig = 1.0 / (1.0 + neg_exp)

        out = Tensor(sig.data, requires_grad=x.requires_grad, is_leaf=False, dtype=sig.dtype)

        if x.requires_grad:
            _x = x
            _sig = sig.data

            def _backward():
                if out.grad is None:
                    return
                # Sigmoid derivative: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                grad = out.grad * _sig * (1.0 - _sig)
                _x.grad = grad if _x.grad is None else _x.grad + grad

            out._backward = _backward
            out._prev = {x}
            out._op = "sigmoid"

        return out

    def __repr__(self):
        return "Sigmoid()"
