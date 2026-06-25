import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class Dropout(Module):
    """Inverted dropout: randomly zeros activations with probability p during training.

    Scales surviving activations by 1/(1-p) so test-time pass-through is unchanged.
    No-op during eval mode (training=False).
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        # Generate dropout mask: keep with probability (1-p), scale by 1/(1-p)
        scale = 1.0 / (1.0 - self.p) if self.p < 1.0 else 1.0
        rand_mask = Tensor.rand(*x.shape, dtype=x.dtype)
        keep_mask = Tensor((rand_mask.data > self.p)).astype(x.dtype)
        mask = (keep_mask.data * scale)
        result = x.data * mask

        out = Tensor(result, requires_grad=x.requires_grad, is_leaf=False)

        if x.requires_grad:
            _x = x
            _mask = mask

            def _backward():
                if out.grad is None:
                    return
                # Gradient passes through mask: zeros where dropped, scales where kept
                grad = out.grad * _mask
                _x.grad = grad if _x.grad is None else _x.grad + grad

            out._backward = _backward
            out._prev = {x}
            out._op = "dropout"

        return out

    def __repr__(self):
        return f"Dropout(p={self.p})"


# def test_dropout():
#     print("\n" + "=" * 60)
#     print("TESTING DROPOUT LAYER")
#     print("=" * 60)

#     print("\n===== 1. Constructor =====")
#     dropout = Dropout(p=0.5)
#     print(dropout)
#     assert dropout.p == 0.5
#     print("[PASS] Constructor works")

#     print("\n===== 2. Training Mode (p=0) =====")
#     dropout = Dropout(p=0.0)
#     dropout.train()
#     x = Tensor.randn(4, 3, 32, 32)
#     y = dropout(x)
#     assert (y.data == x.data).all()
#     print("With p=0, no dropout occurs")
#     print("[PASS] p=0 works")

#     print("\n===== 3. Eval Mode (no dropout) =====")
#     dropout = Dropout(p=0.5)
#     dropout.eval()
#     x = Tensor.randn(4, 3, 32, 32)
#     y = dropout(x)
#     assert (y.data == x.data).all()
#     print("In eval mode, no dropout occurs")
#     print("[PASS] Eval mode works")

#     print("\n===== 4. Training Mode (dropout active) =====")
#     dropout = Dropout(p=0.5)
#     dropout.train()
#     x = Tensor.randn(4, 3, 32, 32)
#     y = dropout(x)
#     # With p=0.5, ~50% of values should be zero
#     zero_fraction = (y.data == 0).sum() / y.data.size
#     print(f"Zero fraction: {zero_fraction:.2f} (expected ~0.5)")
#     assert 0.3 < zero_fraction < 0.7  # Allow variance
#     print("[PASS] Training mode applies dropout")

#     print("\n===== 5. Scaling Factor =====")
#     dropout = Dropout(p=0.5)
#     dropout.train()
#     x = Tensor.ones(1, 1000, 1, 1)
#     y = dropout(x)
#     # Non-zero values should be ~2.0 (scaled by 1/(1-0.5))
#     nonzero_vals = y.data[y.data != 0]
#     mean_val = nonzero_vals.mean() if nonzero_vals.size > 0 else 0
#     print(f"Mean of non-zero values: {mean_val:.4f} (expected ~2.0)")
#     assert 1.8 < mean_val < 2.2
#     print("[PASS] Scaling factor correct")

#     print("\n===== 6. Backward Pass =====")
#     dropout = Dropout(p=0.5)
#     dropout.train()
#     x = Tensor.randn(2, 16, 8, 8, requires_grad=True)
#     y = dropout(x)
#     loss = y.sum()
#     loss.backward()
#     print(f"x.grad shape: {x.grad.shape}")
#     assert x.grad is not None
#     assert x.grad.shape == x.shape
#     print("[PASS] Backward pass works")

#     print("\n===== 7. Gradient Masking =====")
#     dropout = Dropout(p=0.5)
#     dropout.train()
#     x = Tensor.ones(1, 100, 1, 1, requires_grad=True)
#     y = dropout(x)
#     loss = y.sum()
#     loss.backward()
#     # Gradients should be 0 or 2.0 (scaled)
#     unique_grads = set(x.grad.flatten().tolist())
#     print(f"Unique gradient values: {unique_grads}")
#     assert all(g in [0.0, 2.0] for g in unique_grads)
#     print("[PASS] Gradient masking correct")

#     print("\n===== 8. Multiple Batches =====")
#     dropout = Dropout(p=0.3)
#     dropout.train()
#     x = Tensor.randn(8, 64, 32, 32)
#     y = dropout(x)
#     assert y.shape == x.shape
#     print(f"Batch size 8: output shape {y.shape}")
#     print("[PASS] Multi-batch works")

#     print("\n===== 9. High Dropout Rate =====")
#     dropout = Dropout(p=0.9)
#     dropout.train()
#     x = Tensor.randn(2, 500, 1, 1)
#     y = dropout(x)
#     zero_fraction = (y.data == 0).sum() / y.data.size
#     print(f"Zero fraction with p=0.9: {zero_fraction:.2f} (expected ~0.9)")
#     assert 0.8 < zero_fraction < 1.0
#     print("[PASS] High dropout works")

#     print("\n===== 10. Chaining with Other Ops =====")
#     dropout = Dropout(p=0.2)
#     dropout.train()
#     x = Tensor.randn(2, 16, requires_grad=True)
#     y = dropout(x)
#     z = y @ Tensor.randn(16, 10)
#     loss = z.sum()
#     loss.backward()
#     assert x.grad is not None
#     print("Chained ops: input -> dropout -> matmul -> loss")
#     print("[PASS] Chaining works")

#     print("\n" + "=" * 60)
#     print("ALL DROPOUT TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_dropout()
