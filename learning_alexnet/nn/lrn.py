import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        out = Tensor(result, requires_grad=x.requires_grad, is_leaf=False, dtype=result.dtype)

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
