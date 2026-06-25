import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class Sigmoid(Module):
    """Sigmoid activation: f(x) = 1 / (1 + exp(-x))

    Uses numerically stable computation with clipping to prevent overflow.
    """

    def forward(self, x: Tensor) -> Tensor:
        # Clip input for numerical stability: sigmoid(x) ≈ 0 for x < -500, ≈ 1 for x > 500
        x_clipped = x.clip(min=-500.0, max=500.0)
        # Compute sigmoid = 1 / (1 + e^(-x))
        neg_exp = (-x_clipped).exp()
        sig = 1.0 / (1.0 + neg_exp)

        out = Tensor(sig.data, requires_grad=x.requires_grad, is_leaf=False)

        if x.requires_grad:
            _x = x
            _sig = sig.data

            def _backward():
                if out.grad is None:
                    return
                # Sigmoid derivative: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                grad = out.grad * _sig * (1.0 - _sig)
                _x.grad = grad if _x.grad is None else _x.grad + grad

            out._backward = _backward
            out._prev = {x}
            out._op = "sigmoid"

        return out

    def __repr__(self):
        return "Sigmoid()"






# def test_sigmoid():
#     print("\n" + "=" * 60)
#     print("TESTING SIGMOID ACTIVATION LAYER")
#     print("=" * 60)

#     print("\n===== 1. Constructor =====")
#     sigmoid = Sigmoid()
#     print(sigmoid)
#     print("[PASS] Constructor works")

#     print("\n===== 2. Forward Pass (Basic) =====")
#     x = Tensor.randn(4, 3, 32, 32)
#     y = sigmoid(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {y.shape}")
#     assert y.shape == x.shape
#     print("[PASS] Forward pass works")

#     print("\n===== 3. Sigmoid Range =====")
#     x = Tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
#     y = sigmoid(x)
#     print(f"Input: {x.data}")
#     print(f"Output (sigmoid): {y.data}")
#     # Check all outputs are in [0, 1]
#     assert all(0.0 <= val <= 1.0 for val in y.data.flatten())
#     print("[PASS] Sigmoid outputs in [0, 1]")

#     print("\n===== 4. Sigmoid Properties =====")
#     # sigmoid(0) = 0.5
#     x = Tensor([0.0])
#     y = sigmoid(x)
#     print(f"sigmoid(0) = {y.data[0]:.6f}")
#     assert abs(y.data[0] - 0.5) < 1e-5

#     # sigmoid(-x) + sigmoid(x) = 1
#     x = Tensor([1.0])
#     y_pos = sigmoid(x)
#     y_neg = sigmoid(-x)
#     print(f"sigmoid(1) = {y_pos.data[0]:.6f}")
#     print(f"sigmoid(-1) = {y_neg.data[0]:.6f}")
#     print(f"Sum = {(y_pos.data[0] + y_neg.data[0]):.6f}")
#     assert abs(y_pos.data[0] + y_neg.data[0] - 1.0) < 1e-5
#     print("[PASS] Sigmoid properties verified")

#     print("\n===== 5. Backward Pass =====")
#     sigmoid_layer = Sigmoid()
#     x = Tensor.randn(2, 16, 8, 8, requires_grad=True)
#     y = sigmoid_layer(x)
#     loss = y.sum()
#     loss.backward()
#     print(f"x.grad shape: {x.grad.shape}")
#     assert x.grad is not None
#     assert x.grad.shape == x.shape
#     print("[PASS] Backward pass works")

#     print("\n===== 6. Gradient at Zero =====")
#     sigmoid_layer = Sigmoid()
#     x = Tensor([0.0], requires_grad=True)
#     y = sigmoid_layer(x)
#     loss = y.sum()
#     loss.backward()
#     # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)), at x=0: 0.5 * 0.5 = 0.25
#     expected_grad = 0.25
#     print(f"Gradient at x=0: {x.grad[0]:.6f}")
#     print(f"Expected: {expected_grad:.6f}")
#     assert abs(x.grad[0] - expected_grad) < 1e-4
#     print("[PASS] Gradient values correct")

#     print("\n===== 7. Monotonicity =====")
#     x = Tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
#     y = sigmoid(x)
#     # Check each successive value is larger
#     for i in range(len(y.data) - 1):
#         assert y.data[i] < y.data[i + 1]
#     print("Monotonically increasing: verified")
#     print("[PASS] Sigmoid is monotonic")

#     print("\n===== 8. Numerical Stability =====")
#     sigmoid_layer = Sigmoid()
#     # Test extreme values (should not overflow)
#     x = Tensor([-1000.0, 1000.0])
#     y = sigmoid_layer(x)
#     print(f"sigmoid(-1000) = {y.data[0]:.6f}")
#     print(f"sigmoid(1000) = {y.data[1]:.6f}")
#     assert y.data[0] < 1e-10  # Very close to 0
#     assert y.data[1] > 1 - 1e-10  # Very close to 1
#     print("[PASS] Numerically stable for extreme values")

#     print("\n===== 9. Multiple Batches =====")
#     sigmoid_layer = Sigmoid()
#     x = Tensor.randn(8, 64, 32, 32)
#     y = sigmoid_layer(x)
#     assert y.shape == (8, 64, 32, 32)
#     print(f"Batch size 8: output shape {y.shape}")
#     print("[PASS] Multi-batch works")

#     print("\n===== 10. Chaining with Other Ops =====")
#     sigmoid_layer = Sigmoid()
#     x = Tensor.randn(2, 16, requires_grad=True)
#     y = sigmoid_layer(x)
#     z = y @ Tensor.randn(16, 10)
#     loss = z.sum()
#     loss.backward()
#     assert x.grad is not None
#     print("Chained ops: input -> sigmoid -> matmul -> loss")
#     print("[PASS] Chaining works")

#     print("\n" + "=" * 60)
#     print("ALL SIGMOID TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_sigmoid()
