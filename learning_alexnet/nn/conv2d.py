import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


def _im2col(x: np.ndarray, kH: int, kW: int, stride: int, padding: int) -> np.ndarray:
    """
    Convert input feature map into column matrix for matrix-multiply convolution.
    x: (N, C, H, W)
    Returns: (N*H_out*W_out, C*kH*kW)
    """
    N, C, H, W = x.shape
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    if padding > 0:
        x_tensor = Tensor(x, requires_grad=False)
        x_pad = x_tensor.pad(((0, 0), (0, 0), (padding, padding), (padding, padding))).data
    else:
        x_pad = x

    col = Tensor.zeros(N, C, kH, kW, H_out, W_out, dtype=x.dtype).data
    for i in range(kH):
        for j in range(kW):
            col[:, :, i, j, :, :] = x_pad[
                :, :,
                i : i + stride * H_out : stride,
                j : j + stride * W_out : stride,
            ]
    # (N, C, kH, kW, H_out, W_out) -> (N, H_out, W_out, C, kH, kW) -> (N*H_out*W_out, C*kH*kW)
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)


def _col2im(col: np.ndarray, x_shape: tuple, kH: int, kW: int, stride: int, padding: int) -> np.ndarray:
    """Inverse of im2col — scatter column gradients back into the input shape."""
    N, C, H, W = x_shape
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    col_r = col.reshape(N, H_out, W_out, C, kH, kW).transpose(0, 3, 4, 5, 1, 2)
    x_pad = Tensor.zeros(N, C, H + 2 * padding, W + 2 * padding, dtype=col.dtype).data
    for i in range(kH):
        for j in range(kW):
            x_pad[:, :, i : i + stride * H_out : stride, j : j + stride * W_out : stride] += col_r[:, :, i, j, :, :]

    if padding == 0:
        return x_pad
    return x_pad[:, :, padding:-padding, padding:-padding]


class Conv2D(Module):
    """
    2-D convolution implemented via im2col + GEMM.
    Supports arbitrary kernel size, stride, and zero-padding.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel height and width
        stride: Stride for convolution (default: 1)
        padding: Zero-padding (default: 0)
        bias: Whether to use bias (default: True)
        init: Weight initialization method ('kaiming', 'xavier', 'normal') (default: 'kaiming')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        init: str = "kaiming",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init = init

        w_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.W = Tensor(
            Tensor._initialize_weights(w_shape, init=init),
            requires_grad=True,
        )
        if bias:
            self.b = Tensor.zeros(out_channels, requires_grad=True, dtype=self.W.dtype)
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        N, _, H, W = x.shape
        C_out, _, kH, kW = self.W.shape
        H_out = (H + 2 * self.padding - kH) // self.stride + 1
        W_out = (W + 2 * self.padding - kW) // self.stride + 1

        col = _im2col(x.data, kH, kW, self.stride, self.padding)   # (N*H_out*W_out, C*kH*kW)
        W_flat = self.W.data.reshape(C_out, -1)                      # (C_out, C*kH*kW)

        # (N*H_out*W_out, C_out) -> (N, C_out, H_out, W_out)
        result = (col @ W_flat.T).reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)

        needs_grad = x.requires_grad or self.W.requires_grad
        out = Tensor(result, requires_grad=needs_grad, is_leaf=False, dtype=result.dtype)

        if needs_grad:
            _col, _W_flat, _x, _W = col, W_flat, x, self.W
            _stride, _padding, _kH, _kW = self.stride, self.padding, kH, kW

            def _backward():
                if out.grad is None:
                    return
                # dout: (N, C_out, H_out, W_out) -> (N*H_out*W_out, C_out)
                dout = out.grad.transpose(0, 2, 3, 1).reshape(-1, C_out)
                if _W.requires_grad:
                    dW = (dout.T @ _col).reshape(_W.shape)
                    _W.grad = dW if _W.grad is None else _W.grad + dW
                if _x.requires_grad:
                    dcol = dout @ _W_flat
                    dx = _col2im(dcol, _x.shape, _kH, _kW, _stride, _padding)
                    _x.grad = dx if _x.grad is None else _x.grad + dx

            out._backward = _backward
            out._prev = {x, self.W}
            out._op = "conv2d"

        if self.b is not None:
            out = out + self.b.reshape(1, C_out, 1, 1)

        return out

    def __repr__(self):
        bias_str = ", bias=True" if self.b is not None else ", bias=False"
        init_str = f", init={self.init}"
        return (
            f"Conv2D({self.in_channels}, {self.out_channels}, "
            f"kernel={self.kernel_size}, stride={self.stride}, pad={self.padding}{bias_str}{init_str})"
        )
