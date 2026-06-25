import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class Flatten(Module):
    """Flatten spatial dims into a single vector per sample."""

    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        flat = 1
        for d in shape[self.start_dim:]:
            flat *= d
        new_shape = shape[: self.start_dim] + (flat,)
        return x.reshape(*new_shape)

    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim})"




# def test_flatten():
#     print("\n" + "=" * 60)
#     print("TESTING FLATTEN LAYER")
#     print("=" * 60)

#     print("\n===== 1. Constructor =====")
#     flatten = Flatten(start_dim=1)
#     print(flatten)
#     assert flatten.start_dim == 1
#     print("[PASS] Constructor works")

#     print("\n===== 2. Forward Pass (Basic) =====")
#     x = Tensor.randn(4, 3, 32, 32)
#     y = flatten(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {y.shape}")
#     assert y.shape == (4, 3 * 32 * 32)
#     print("[PASS] Forward pass works")

#     print("\n===== 3. Flatten from Different Dims =====")
#     test_cases = [
#         ((2, 3, 4, 5), 1, (2, 60)),
#         ((2, 3, 4, 5), 2, (2, 3, 20)),
#         ((2, 3, 4, 5), 3, (2, 3, 4, 5)),
#     ]
#     for input_shape, start_dim, expected_shape in test_cases:
#         flatten = Flatten(start_dim=start_dim)
#         x = Tensor.randn(*input_shape)
#         y = flatten(x)
#         assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
#         print(f"Flatten from dim {start_dim}: {input_shape} -> {y.shape}")
#     print("[PASS] Different start_dim works")

#     print("\n===== 4. Backward Pass =====")
#     flatten = Flatten(start_dim=1)
#     x = Tensor.randn(2, 3, 4, 4, requires_grad=True)
#     y = flatten(x)
#     loss = y.sum()
#     loss.backward()
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {y.shape}")
#     print(f"Gradient shape: {x.grad.shape}")
#     assert x.grad is not None
#     assert x.grad.shape == x.shape
#     print("[PASS] Backward pass works")

#     print("\n===== 5. Gradient Flow =====")
#     flatten = Flatten(start_dim=1)
#     x = Tensor.ones(2, 3, 4, 4, requires_grad=True)
#     y = flatten(x)
#     loss = (y ** 2).sum()
#     loss.backward()
#     expected_grad = 2 * 1.0  # d/dx (x^2) = 2x
#     print(f"Gradient values (sample): {x.grad[0, 0, 0, 0]:.4f}")
#     assert abs(x.grad[0, 0, 0, 0] - expected_grad) < 1e-5
#     print("[PASS] Gradient values correct")

#     print("\n===== 6. Preserves Batch Dimension =====")
#     flatten = Flatten(start_dim=1)
#     x = Tensor.randn(8, 64, 28, 28)
#     y = flatten(x)
#     assert y.shape[0] == 8  # Batch size preserved
#     print(f"Batch size: {y.shape[0]}")
#     print("[PASS] Batch dimension preserved")

#     print("\n===== 7. Start Dim 0 (Flatten All) =====")
#     flatten = Flatten(start_dim=0)
#     x = Tensor.randn(2, 3, 4, 4)
#     y = flatten(x)
#     assert y.shape == (2 * 3 * 4 * 4,)
#     print(f"Flatten all: {x.shape} -> {y.shape}")
#     print("[PASS] Flatten all works")

#     print("\n===== 8. 2D Input (No Op) =====")
#     flatten = Flatten(start_dim=1)
#     x = Tensor.randn(4, 512)
#     y = flatten(x)
#     assert y.shape == x.shape
#     print(f"2D input: {x.shape} (unchanged)")
#     print("[PASS] 2D input works")

#     print("\n===== 9. Chained with Other Ops =====")
#     flatten = Flatten(start_dim=1)
#     x = Tensor.randn(2, 3, 4, 4, requires_grad=True)
#     y = flatten(x)
#     z = y @ Tensor.randn(48, 10)  # (2, 48) @ (48, 10) -> (2, 10)
#     loss = z.sum()
#     loss.backward()
#     assert x.grad is not None
#     print(f"Chained ops: {x.shape} -> flatten -> matmul -> loss")
#     print("[PASS] Chaining works")

#     print("\n===== 10. Repr Output =====")
#     flatten = Flatten(start_dim=1)
#     print(f"Repr: {flatten}")
#     assert "start_dim=1" in repr(flatten)
#     print("[PASS] Repr shows start_dim")

#     print("\n" + "=" * 60)
#     print("ALL FLATTEN TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_flatten()
