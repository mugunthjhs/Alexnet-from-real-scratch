import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class Tanh(Module):
    """Hyperbolic tangent activation: f(x) = tanh(x) = (e^x - e^-x) / (e^x + e^-x)

    Also equivalent to: tanh(x) = 2*sigmoid(2x) - 1
    """

    def forward(self, x: Tensor) -> Tensor:
        # Compute tanh using: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        # For numerical stability, clip large values
        x_clipped = x.clip(min=-500.0, max=500.0)
        exp_pos = x_clipped.exp()
        exp_neg = (-x_clipped).exp()
        tanh_val = (exp_pos - exp_neg) / (exp_pos + exp_neg)

        out = Tensor(tanh_val.data, requires_grad=x.requires_grad, is_leaf=False, dtype=tanh_val.dtype)

        if x.requires_grad:
            _x = x
            _tanh = tanh_val.data

            def _backward():
                if out.grad is None:
                    return
                # Tanh derivative: d/dx tanh(x) = 1 - tanh(x)^2
                grad = out.grad * (1.0 - _tanh ** 2)
                _x.grad = grad if _x.grad is None else _x.grad + grad

            out._backward = _backward
            out._prev = {x}
            out._op = "tanh"

        return out

    def __repr__(self):
        return "Tanh()"
