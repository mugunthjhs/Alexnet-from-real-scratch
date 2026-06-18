import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class ReLU(Module):
    """Rectified Linear Unit: f(x) = max(0, x)"""

    def forward(self, x: Tensor) -> Tensor:
        result = np.maximum(0.0, x.data)
        out = Tensor(result, requires_grad=x.requires_grad, is_leaf=False)

        if x.requires_grad:
            _x = x

            def _backward():
                if out.grad is None:
                    return
                grad = out.grad * ((result > 0) * 1.0)
                _x.grad = grad if _x.grad is None else _x.grad + grad

            out._backward = _backward
            out._prev = {x}
            out._op = "relu"

        return out

    def __repr__(self):
        return "ReLU()"


# def test_relu():
#     print("=" * 50)
#     print("Testing ReLU")
#     print("=" * 50)

#     print("\n1. Forward Pass")
#     x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
#     relu = ReLU()
#     y = relu(x)
#     print(f"Input:  {x.data}")
#     print(f"Output: {y.data}")
#     assert np.allclose(y.data, [0, 0, 0, 1, 2])
#     print("✓ Forward pass works")

#     print("\n2. Backward Pass (Gradients)")
#     y.backward(np.ones_like(y.data))
#     print(f"Input gradients: {x.grad}")
#     assert np.allclose(x.grad, [0, 0, 0, 1, 1])
#     print("✓ Backward pass works")

#     print("\n3. Dtype Conversion (astype)")
#     x = Tensor([1.5, 2.7, 3.2], requires_grad=True, dtype=np.float32)
#     print(f"Original dtype: {x.dtype}")
#     x_float64 = x.astype(np.float64)
#     print(f"Converted dtype: {x_float64.dtype}")
#     assert x_float64.dtype == np.float64
#     print("✓ astype works")

#     print("\n4. ReLU with Different Dtypes")
#     x_fp16 = Tensor([-1.0, 0.5, 2.0], dtype=np.float16, requires_grad=True)
#     y_fp16 = relu(x_fp16)
#     print(f"Input dtype: {x_fp16.dtype}, Output dtype: {y_fp16.dtype}")
#     print(f"Output: {y_fp16.data}")
#     assert np.allclose(y_fp16.data, [0, 0.5, 2.0])
#     print("✓ ReLU with float16 works")

#     print("\n5. Gradient Flow with astype")
#     x = Tensor([1.0, -1.0, 2.0], requires_grad=True, dtype=np.float32)
#     x_float64 = x.astype(np.float64)
#     relu = ReLU()
#     y = relu(x_float64)
#     y.backward(np.ones_like(y.data))
#     print(f"Original tensor gradient dtype: {x.grad.dtype}")
#     print(f"Gradient: {x.grad}")
#     assert np.allclose(x.grad, [1, 0, 1])
#     print("✓ Gradient flow with astype works")

#     print("\n" + "=" * 50)
#     print("ALL TESTS PASSED ✓")
#     print("=" * 50)


# if __name__ == "__main__":
#     try:
#         test_relu()
#     except ImportError:
#         print("Note: Run this file as a module for full testing")
