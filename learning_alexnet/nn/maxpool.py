import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class MaxPool2D(Module):
    """
    2-D max pooling.
    Backward ties each output to the position of its maximum input value.
    When multiple positions tie for the max, gradient is split evenly.
    """

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
            x_data = x_tensor.pad(((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), constant_values=-float('inf')).data

        result = Tensor.zeros(N, C, H_out, W_out, dtype=x.dtype).data
        for i in range(H_out):
            for j in range(W_out):
                hs, ws = i * self.stride, j * self.stride
                patch = x_data[:, :, hs : hs + k, ws : ws + k]
                patch_tensor = Tensor(patch, requires_grad=False)
                max_val = patch_tensor.max(axis=(2, 3)).data
                result[:, :, i, j] = max_val

        out = Tensor(result, requires_grad=x.requires_grad, is_leaf=False, dtype=result.dtype)

        if x.requires_grad:
            _x, _x_data = x, x_data

            def _backward():
                if out.grad is None:
                    return
                dx = Tensor.zeros(*_x_data.shape, dtype=_x_data.dtype).data
                for i in range(H_out):
                    for j in range(W_out):
                        hs, ws = i * self.stride, j * self.stride
                        patch = _x_data[:, :, hs : hs + k, ws : ws + k]
                        max_val = out.data[:, :, i, j].reshape(N, C, 1, 1)
                        mask = (patch == max_val).astype(_x_data.dtype)
                        mask_sum = mask.sum(axis=(2, 3), keepdims=True)
                        mask /= mask_sum.clip(1.0)  # Distribute equally among tied max values
                        grad_vals = out.grad[:, :, i, j].reshape(N, C, 1, 1)
                        dx[:, :, hs : hs + k, ws : ws + k] += mask * grad_vals
                if self.padding > 0:
                    dx = dx[:, :, self.padding : -self.padding, self.padding : -self.padding]
                _x.grad = dx if _x.grad is None else _x.grad + dx

            out._backward = _backward
            out._prev = {x}
            out._op = "maxpool2d"

        return out

    def __repr__(self):
        pad_str = f", pad={self.padding}" if self.padding > 0 else ""
        return f"MaxPool2D(kernel={self.kernel_size}, stride={self.stride}{pad_str})"
