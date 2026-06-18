import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class Sigmoid(Module):
    """Sigmoid activation: f(x) = 1 / (1 + exp(-x))"""

    def forward(self, x: Tensor) -> Tensor:
        sig = 1.0 / (1.0 + np.exp(-x.data.clip(-500, 500)))
        out = Tensor(sig, requires_grad=x.requires_grad, is_leaf=False)

        if x.requires_grad:
            _x = x

            def _backward():
                if out.grad is None:
                    return
                grad = out.grad * sig * (1.0 - sig)
                _x.grad = grad if _x.grad is None else _x.grad + grad

            out._backward = _backward
            out._prev = {x}
            out._op = "sigmoid"

        return out

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """Hyperbolic tangent activation: f(x) = tanh(x)"""

    def forward(self, x: Tensor) -> Tensor:
        t = np.tanh(x.data)
        out = Tensor(t, requires_grad=x.requires_grad, is_leaf=False)

        if x.requires_grad:
            _x = x

            def _backward():
                if out.grad is None:
                    return
                grad = out.grad * (1.0 - t ** 2)
                _x.grad = grad if _x.grad is None else _x.grad + grad

            out._backward = _backward
            out._prev = {x}
            out._op = "tanh"

        return out

    def __repr__(self):
        return "Tanh()"
