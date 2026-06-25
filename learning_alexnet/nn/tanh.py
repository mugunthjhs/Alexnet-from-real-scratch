import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class Tanh(Module):
    """Hyperbolic tangent activation: f(x) = tanh(x) = (e^x - e^-x) / (e^x + e^-x)

    Also equivalent to: tanh(x) = 2*sigmoid(2x) - 1
    """

    def forward(self, x: Tensor) -> Tensor:
        # Compute tanh using: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        # For numerical stability, clip large values
        x_clipped = x.clip(min=-500.0, max=500.0)
        exp_pos = x_clipped.exp()
        exp_neg = (-x_clipped).exp()
        tanh_val = (exp_pos - exp_neg) / (exp_pos + exp_neg)

        out = Tensor(tanh_val.data, requires_grad=x.requires_grad, is_leaf=False)

        if x.requires_grad:
            _x = x
            _tanh = tanh_val.data

            def _backward():
                if out.grad is None:
                    return
                # Tanh derivative: d/dx tanh(x) = 1 - tanh(x)^2
                grad = out.grad * (1.0 - _tanh ** 2)
                _x.grad = grad if _x.grad is None else _x.grad + grad

            out._backward = _backward
            out._prev = {x}
            out._op = "tanh"

        return out

    def __repr__(self):
        return "Tanh()"




# def test_tanh():
#     print("\n" + "=" * 60)
#     print("TESTING TANH ACTIVATION LAYER")
#     print("=" * 60)

#     print("\n===== 1. Constructor =====")
#     tanh = Tanh()
#     print(tanh)
#     print("[PASS] Constructor works")

#     print("\n===== 2. Forward Pass (Basic) =====")
#     x = Tensor.randn(4, 3, 32, 32)
#     y = tanh(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {y.shape}")
#     assert y.shape == x.shape
#     print("[PASS] Forward pass works")

#     print("\n===== 3. Tanh Range =====")
#     x = Tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
#     y = tanh(x)
#     print(f"Input: {x.data}")
#     print(f"Output (tanh): {y.data}")
#     # Check all outputs are in [-1, 1]
#     assert all(-1.0 <= val <= 1.0 for val in y.data.flatten())
#     print("[PASS] Tanh outputs in [-1, 1]")

#     print("\n===== 4. Tanh Properties =====")
#     # tanh(0) = 0
#     x = Tensor([0.0])
#     y = tanh(x)
#     print(f"tanh(0) = {y.data[0]:.6f}")
#     assert abs(y.data[0]) < 1e-5

#     # tanh is odd: tanh(-x) = -tanh(x)
#     x = Tensor([1.0])
#     y_pos = tanh(x)
#     y_neg = tanh(-x)
#     print(f"tanh(1) = {y_pos.data[0]:.6f}")
#     print(f"tanh(-1) = {y_neg.data[0]:.6f}")
#     print(f"Sum = {(y_pos.data[0] + y_neg.data[0]):.6f}")
#     assert abs(y_pos.data[0] + y_neg.data[0]) < 1e-5
#     print("[PASS] Tanh properties verified")

#     print("\n===== 5. Backward Pass =====")
#     tanh_layer = Tanh()
#     x = Tensor.randn(2, 16, 8, 8, requires_grad=True)
#     y = tanh_layer(x)
#     loss = y.sum()
#     loss.backward()
#     print(f"x.grad shape: {x.grad.shape}")
#     assert x.grad is not None
#     assert x.grad.shape == x.shape
#     print("[PASS] Backward pass works")

#     print("\n===== 6. Gradient at Zero =====")
#     tanh_layer = Tanh()
#     x = Tensor([0.0], requires_grad=True)
#     y = tanh_layer(x)
#     loss = y.sum()
#     loss.backward()
#     # d/dx tanh(x) = 1 - tanh(x)^2, at x=0: 1 - 0 = 1
#     expected_grad = 1.0
#     print(f"Gradient at x=0: {x.grad[0]:.6f}")
#     print(f"Expected: {expected_grad:.6f}")
#     assert abs(x.grad[0] - expected_grad) < 1e-4
#     print("[PASS] Gradient values correct")

#     print("\n===== 7. Monotonicity =====")
#     x = Tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
#     y = tanh(x)
#     # Check each successive value is larger
#     for i in range(len(y.data) - 1):
#         assert y.data[i] < y.data[i + 1]
#     print("Monotonically increasing: verified")
#     print("[PASS] Tanh is monotonic")

#     print("\n===== 8. Numerical Stability =====")
#     tanh_layer = Tanh()
#     # Test extreme values (should not overflow)
#     x = Tensor([-1000.0, 1000.0])
#     y = tanh_layer(x)
#     print(f"tanh(-1000) = {y.data[0]:.6f}")
#     print(f"tanh(1000) = {y.data[1]:.6f}")
#     assert y.data[0] < -1 + 1e-5  # Very close to -1
#     assert y.data[1] > 1 - 1e-5  # Very close to 1
#     print("[PASS] Numerically stable for extreme values")

#     print("\n===== 9. Multiple Batches =====")
#     tanh_layer = Tanh()
#     x = Tensor.randn(8, 64, 32, 32)
#     y = tanh_layer(x)
#     assert y.shape == (8, 64, 32, 32)
#     print(f"Batch size 8: output shape {y.shape}")
#     print("[PASS] Multi-batch works")

#     print("\n===== 10. Chaining with Other Ops =====")
#     tanh_layer = Tanh()
#     x = Tensor.randn(2, 16, requires_grad=True)
#     y = tanh_layer(x)
#     z = y @ Tensor.randn(16, 10)
#     loss = z.sum()
#     loss.backward()
#     assert x.grad is not None
#     print("Chained ops: input -> tanh -> matmul -> loss")
#     print("[PASS] Chaining works")

#     print("\n" + "=" * 60)
#     print("ALL TANH TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_tanh()
