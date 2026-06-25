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
        out = Tensor(result, requires_grad=needs_grad, is_leaf=False)

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


# def test_conv2d():
#     print("\n" + "=" * 60)
#     print("TESTING CONV2D LAYER")
#     print("=" * 60)

#     print("\n===== 1. Constructor =====")
#     conv = Conv2D(3, 64, kernel_size=3, padding=1)
#     print(conv)
#     print(f"W shape: {conv.W.shape}")
#     print(f"W requires_grad: {conv.W.requires_grad}")
#     if conv.b is not None:
#         print(f"b shape: {conv.b.shape}")
#         print(f"b requires_grad: {conv.b.requires_grad}")
#     assert conv.W.shape == (64, 3, 3, 3)
#     assert conv.b.shape == (64,)
#     print("[PASS] Constructor works")

#     print("\n===== 2. Forward Pass =====")
#     x = Tensor.randn(4, 3, 32, 32)
#     y = conv(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {y.shape}")
#     assert y.shape == (4, 64, 32, 32)
#     print("[PASS] Forward pass works")

#     print("\n===== 3. Forward with Stride =====")
#     conv_stride = Conv2D(3, 64, kernel_size=3, stride=2, padding=1)
#     x = Tensor.randn(2, 3, 64, 64)
#     y = conv_stride(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape with stride=2: {y.shape}")
#     assert y.shape == (2, 64, 32, 32)
#     print("[PASS] Stride works")

#     print("\n===== 4. Forward with Different Padding =====")
#     conv_no_pad = Conv2D(3, 32, kernel_size=5, padding=0)
#     x = Tensor.randn(1, 3, 28, 28)
#     y = conv_no_pad(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape (no padding): {y.shape}")
#     assert y.shape == (1, 32, 24, 24)
#     print("[PASS] Padding=0 works")

#     print("\n===== 5. Weight Initialization (Kaiming) =====")
#     kaiming_conv = Conv2D(64, 128, kernel_size=3, init="kaiming")
#     std = kaiming_conv.W.data.std()
#     print(f"Kaiming W std: {std:.6f}")
#     assert std > 0
#     assert kaiming_conv.init == "kaiming"
#     print("[PASS] Kaiming initialization works")

#     print("\n===== 6. Weight Initialization (Xavier) =====")
#     xavier_conv = Conv2D(64, 128, kernel_size=3, init="xavier")
#     std = xavier_conv.W.data.std()
#     print(f"Xavier W std: {std:.6f}")
#     assert std > 0
#     assert xavier_conv.init == "xavier"
#     print("[PASS] Xavier initialization works")

#     print("\n===== 7. Weight Initialization (Normal) =====")
#     normal_conv = Conv2D(64, 128, kernel_size=3, init="normal")
#     std = normal_conv.W.data.std()
#     print(f"Normal W std: {std:.6f}")
#     assert std > 0
#     assert normal_conv.init == "normal"
#     print("[PASS] Normal initialization works")

#     print("\n===== 8. No Bias Conv2D =====")
#     conv_no_bias = Conv2D(3, 32, kernel_size=3, padding=1, bias=False)
#     assert conv_no_bias.b is None
#     x = Tensor.randn(2, 3, 32, 32)
#     y = conv_no_bias(x)
#     assert y.shape == (2, 32, 32, 32)
#     print("[PASS] No-bias conv works")

#     print("\n===== 9. Backward Pass =====")
#     conv = Conv2D(3, 16, kernel_size=3, padding=1)
#     x = Tensor.randn(2, 3, 16, 16, requires_grad=True)
#     y = conv(x)
#     loss = y.sum()
#     loss.backward()
#     print(f"x.grad shape: {x.grad.shape}")
#     print(f"W.grad shape: {conv.W.grad.shape}")
#     if conv.b is not None:
#         print(f"b.grad shape: {conv.b.grad.shape}")
#     assert x.grad is not None
#     assert conv.W.grad is not None
#     assert conv.b.grad is not None
#     print("[PASS] Backward pass works")

#     print("\n===== 10. Multiple Channels =====")
#     conv = Conv2D(128, 256, kernel_size=3, padding=1)
#     x = Tensor.randn(4, 128, 14, 14)
#     y = conv(x)
#     assert y.shape == (4, 256, 14, 14)
#     print(f"Input channels: 128, Output channels: 256")
#     print(f"Output shape: {y.shape}")
#     print("[PASS] Multi-channel works")

#     print("\n===== 11. Large Kernel =====")
#     conv = Conv2D(3, 64, kernel_size=11, padding=5)
#     x = Tensor.randn(1, 3, 224, 224)
#     y = conv(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape (kernel=11, pad=5): {y.shape}")
#     assert y.shape == (1, 64, 224, 224)
#     print("[PASS] Large kernel works")

#     print("\n===== 12. Repr Output =====")
#     conv_kaiming = Conv2D(3, 64, kernel_size=3, padding=1, init="kaiming")
#     print(f"Repr: {conv_kaiming}")
#     assert "kaiming" in repr(conv_kaiming)
#     assert "bias=True" in repr(conv_kaiming)
#     print("[PASS] Repr shows init and bias")

#     print("\n" + "=" * 60)
#     print("ALL CONV2D TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_conv2d()