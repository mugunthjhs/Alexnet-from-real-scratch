import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tensor import Tensor
from .module import Module


class LocalResponseNorm(Module):
    """
    Local Response Normalisation (LRN) as introduced in AlexNet.

    For each channel i:
        b_i = a_i / (k + alpha * sum_{j=max(0,i-n//2)}^{min(C-1,i+n//2)} a_j^2) ^ beta

    AlexNet defaults: size=5, alpha=1e-4, beta=0.75, k=2.
    """

    def __init__(self, size: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 2.0):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        half = self.size // 2

        sq = x.data ** 2
        scale = Tensor.zeros(N, C, H, W, dtype=x.dtype).data + self.k
        for c in range(C):
            c0 = max(0, c - half)
            c1 = min(C, c + half + 1)
            scale[:, c, :, :] += self.alpha * sq[:, c0:c1, :, :].sum(axis=1)

        result = x.data / (scale ** self.beta)
        out = Tensor(result, requires_grad=x.requires_grad, is_leaf=False)

        if x.requires_grad:
            _x, _scale, _result = x, scale, result
            _alpha, _beta, _half = self.alpha, self.beta, half
            _C = C

            def _backward():
                if out.grad is None:
                    return
                # Two-term derivative of LRN:
                # dx_i = dout_i / scale_i^beta
                #      - 2*alpha*beta * x_i * sum_{k: i in window(k)} (dout_k * result_k / scale_k)
                dx = out.grad / (_scale ** _beta)
                tmp = out.grad * _result / _scale   # (N, C, H, W)
                window_sum = Tensor.zeros(N, _C, H, W, dtype=_x.dtype).data
                for c in range(_C):
                    c0 = max(0, c - _half)
                    c1 = min(_C, c + _half + 1)
                    for ci in range(c0, c1):
                        window_sum[:, ci, :, :] += tmp[:, c, :, :]
                dx -= 2 * _alpha * _beta * _x.data * window_sum
                _x.grad = dx if _x.grad is None else _x.grad + dx

            out._backward = _backward
            out._prev = {x}
            out._op = "lrn"

        return out

    def __repr__(self):
        return f"LocalResponseNorm(size={self.size}, alpha={self.alpha}, beta={self.beta}, k={self.k})"






# def test_lrn():
#     print("\n" + "=" * 60)
#     print("TESTING LOCAL RESPONSE NORM (LRN) LAYER")
#     print("=" * 60)

#     print("\n===== 1. Constructor =====")
#     lrn = LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
#     print(lrn)
#     assert lrn.size == 5
#     assert lrn.alpha == 1e-4
#     assert lrn.beta == 0.75
#     assert lrn.k == 2.0
#     print("[PASS] Constructor works")

#     print("\n===== 2. Forward Pass (Basic) =====")
#     x = Tensor.randn(2, 64, 32, 32)
#     y = lrn(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {y.shape}")
#     assert y.shape == x.shape
#     print("[PASS] Forward pass works")

#     print("\n===== 3. Forward Normalization Effect =====")
#     x = Tensor.ones(1, 3, 4, 4)
#     lrn = LocalResponseNorm(size=3, alpha=1e-4, beta=0.75, k=2.0)
#     y = lrn(x)
#     print(f"Input value: 1.0")
#     print(f"Output value: {y.data[0, 0, 0, 0]:.6f}")
#     assert y.data[0, 0, 0, 0] < 1.0
#     print("[PASS] Normalization reduces values")

#     print("\n===== 4. Different Sizes =====")
#     for size in [3, 5, 7]:
#         lrn = LocalResponseNorm(size=size)
#         x = Tensor.randn(1, 32, 16, 16)
#         y = lrn(x)
#         assert y.shape == x.shape
#         print(f"Size {size}: shape {y.shape}")
#     print("[PASS] Different sizes work")

#     print("\n===== 5. AlexNet Defaults =====")
#     lrn = LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
#     x = Tensor.randn(1, 96, 55, 55)
#     y = lrn(x)
#     assert y.shape == x.shape
#     print(f"AlexNet LRN: input {x.shape} -> output {y.shape}")
#     print("[PASS] AlexNet defaults work")

#     print("\n===== 6. Backward Pass =====")
#     lrn = LocalResponseNorm(size=5)
#     x = Tensor.randn(2, 16, 8, 8, requires_grad=True)
#     y = lrn(x)
#     loss = y.sum()
#     loss.backward()
#     print(f"x.grad shape: {x.grad.shape}")
#     assert x.grad is not None
#     assert x.grad.shape == x.shape
#     print("[PASS] Backward pass works")

#     print("\n===== 7. Gradient Flow =====")
#     lrn = LocalResponseNorm(size=3)
#     x = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
#     y = lrn(x)
#     loss = y.sum()
#     loss.backward()
#     print(f"Gradient norm: {(x.grad ** 2).sum() ** 0.5:.6f}")
#     assert (x.grad ** 2).sum() > 0
#     print("[PASS] Gradients flow correctly")

#     print("\n===== 8. Multiple Batches =====")
#     lrn = LocalResponseNorm(size=5)
#     x = Tensor.randn(8, 64, 32, 32)
#     y = lrn(x)
#     assert y.shape == (8, 64, 32, 32)
#     print(f"Batch size 8: output shape {y.shape}")
#     print("[PASS] Multi-batch works")

#     print("\n===== 9. Many Channels =====")
#     lrn = LocalResponseNorm(size=5)
#     x = Tensor.randn(1, 256, 13, 13)
#     y = lrn(x)
#     assert y.shape == (1, 256, 13, 13)
#     print(f"Channels 256: output shape {y.shape}")
#     print("[PASS] Many channels works")

#     print("\n===== 10. Parameter Variations =====")
#     params = [
#         {"size": 3, "alpha": 1e-3, "beta": 0.5, "k": 1.0},
#         {"size": 7, "alpha": 1e-5, "beta": 1.0, "k": 3.0},
#         {"size": 5, "alpha": 1e-4, "beta": 0.75, "k": 2.0},
#     ]
#     for p in params:
#         lrn = LocalResponseNorm(**p)
#         x = Tensor.randn(1, 32, 16, 16)
#         y = lrn(x)
#         assert y.shape == x.shape
#         print(f"Params {p['size']},{p['alpha']:.0e},{p['beta']},{p['k']}: [PASS]")
#     print("[PASS] Parameter variations work")

#     print("\n" + "=" * 60)
#     print("ALL LRN TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_lrn()
