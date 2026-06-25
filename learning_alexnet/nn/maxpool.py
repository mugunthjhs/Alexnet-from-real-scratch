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

        out = Tensor(result, requires_grad=x.requires_grad, is_leaf=False)

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







# def test_maxpool2d():
#     print("\n" + "=" * 60)
#     print("TESTING MAXPOOL2D LAYER")
#     print("=" * 60)

#     print("\n===== 1. Constructor =====")
#     pool = MaxPool2D(kernel_size=2, stride=2)
#     print(pool)
#     assert pool.kernel_size == 2
#     assert pool.stride == 2
#     assert pool.padding == 0
#     print("[PASS] Constructor works")

#     print("\n===== 2. Forward Pass (Basic) =====")
#     x = Tensor.randn(4, 3, 32, 32)
#     y = pool(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {y.shape}")
#     assert y.shape == (4, 3, 16, 16)
#     print("[PASS] Forward pass works")

#     print("\n===== 3. Forward with Padding =====")
#     pool_pad = MaxPool2D(kernel_size=3, stride=2, padding=1)
#     x = Tensor.randn(2, 64, 16, 16)
#     y = pool_pad(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape (kernel=3, stride=2, pad=1): {y.shape}")
#     H_out = (16 + 2 * 1 - 3) // 2 + 1
#     W_out = (16 + 2 * 1 - 3) // 2 + 1
#     assert y.shape == (2, 64, H_out, W_out)
#     print("[PASS] Padding works")

#     print("\n===== 4. Forward with Same Stride =====")
#     pool_same = MaxPool2D(kernel_size=2)
#     x = Tensor.randn(1, 32, 28, 28)
#     y = pool_same(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape (stride defaults to kernel_size): {y.shape}")
#     assert y.shape == (1, 32, 14, 14)
#     print("[PASS] Default stride works")

#     print("\n===== 5. Max Values Selected =====")
#     x = Tensor([[[
#         [1.0, 2.0],
#         [3.0, 4.0]
#     ]]])
#     pool = MaxPool2D(kernel_size=2, stride=2)
#     y = pool(x)
#     print(f"Input:\n{x.data}")
#     print(f"Max pool output: {y.data[0, 0, 0, 0]}")
#     assert y.data[0, 0, 0, 0] == 4.0
#     print("[PASS] Max values correctly selected")

#     print("\n===== 6. Backward Pass =====")
#     pool = MaxPool2D(kernel_size=2, stride=2)
#     x = Tensor.randn(2, 16, 8, 8, requires_grad=True)
#     y = pool(x)
#     loss = y.sum()
#     loss.backward()
#     print(f"x.grad shape: {x.grad.shape}")
#     assert x.grad is not None
#     assert x.grad.shape == x.shape
#     print("[PASS] Backward pass works")

#     print("\n===== 7. Gradient Routing =====")
#     x = Tensor([[[
#         [1.0, 2.0],
#         [3.0, 4.0]
#     ]]], requires_grad=True)
#     pool = MaxPool2D(kernel_size=2, stride=2)
#     y = pool(x)
#     loss = y.sum()
#     loss.backward()
#     print(f"Gradient at max position (3,3): {x.grad[0, 0, 1, 1]:.4f}")
#     assert x.grad[0, 0, 1, 1] == 1.0
#     assert x.grad[0, 0, 0, 0] == 0.0
#     print("[PASS] Gradients route to max positions")

#     print("\n===== 8. Multiple Batches =====")
#     pool = MaxPool2D(kernel_size=2, stride=2)
#     x = Tensor.randn(8, 32, 64, 64)
#     y = pool(x)
#     assert y.shape == (8, 32, 32, 32)
#     print(f"Batch size: 8, Output shape: {y.shape}")
#     print("[PASS] Multi-batch works")

#     print("\n===== 9. Many Channels =====")
#     pool = MaxPool2D(kernel_size=2, stride=2)
#     x = Tensor.randn(1, 512, 14, 14)
#     y = pool(x)
#     assert y.shape == (1, 512, 7, 7)
#     print(f"Channels: 512, Output shape: {y.shape}")
#     print("[PASS] Many channels works")

#     print("\n===== 10. Repr Output =====")
#     pool_no_pad = MaxPool2D(kernel_size=2, stride=2)
#     pool_with_pad = MaxPool2D(kernel_size=3, stride=1, padding=1)
#     print(f"No padding: {pool_no_pad}")
#     print(f"With padding: {pool_with_pad}")
#     assert "pad=" not in repr(pool_no_pad)
#     assert "pad=1" in repr(pool_with_pad)
#     print("[PASS] Repr shows padding info")

#     print("\n" + "=" * 60)
#     print("ALL MAXPOOL2D TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_maxpool2d()
