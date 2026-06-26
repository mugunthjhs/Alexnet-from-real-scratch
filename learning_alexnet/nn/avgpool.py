import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class AvgPool2D(Module):
    """2-D average pooling. Gradient is distributed uniformly over the pooling window."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        k = self.kernel_size
        H_out = (H + 2 * self.padding - k) // self.stride + 1
        W_out = (W + 2 * self.padding - k) // self.stride + 1

        x_data = x.data
        if self.padding > 0:
            x_tensor = Tensor(x_data, requires_grad=False)
            x_data = x_tensor.pad(((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))).data

        result = Tensor.zeros(N, C, H_out, W_out, dtype=x.dtype).data
        for i in range(H_out):
            for j in range(W_out):
                hs, ws = i * self.stride, j * self.stride
                result[:, :, i, j] = x_data[:, :, hs : hs + k, ws : ws + k].mean(axis=(2, 3))

        out = Tensor(result, requires_grad=x.requires_grad, is_leaf=False, dtype=result.dtype)

        if x.requires_grad:
            _x, _x_data, _count = x, x_data, k * k

            def _backward():
                if out.grad is None:
                    return
                dx = Tensor.zeros(*_x_data.shape, dtype=_x_data.dtype).data
                for i in range(H_out):
                    for j in range(W_out):
                        hs, ws = i * self.stride, j * self.stride
                        grad_vals = out.grad[:, :, i, j]
                        grad_expanded = grad_vals.reshape(N, C, 1, 1) / _count
                        dx[:, :, hs : hs + k, ws : ws + k] += grad_expanded
                if self.padding > 0:
                    dx = dx[:, :, self.padding : -self.padding, self.padding : -self.padding]
                _x.grad = dx if _x.grad is None else _x.grad + dx

            out._backward = _backward
            out._prev = {x}
            out._op = "avgpool2d"

        return out

    def __repr__(self):
        pad_str = f", pad={self.padding}" if self.padding > 0 else ""
        return f"AvgPool2D(kernel={self.kernel_size}, stride={self.stride}{pad_str})"
