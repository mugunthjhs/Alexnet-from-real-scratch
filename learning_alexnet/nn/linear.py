import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype = np.float32, init: str = "kaiming"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.init = init

        w_shape = (out_features, in_features)
        w_data = Tensor._initialize_weights(w_shape, init=init).T
        self.W = Tensor(w_data, requires_grad=True)

        if bias:
            self.b = Tensor.zeros(out_features, dtype=self.dtype, requires_grad=True)
        else:
            self.b = None

    def forward(self,x):
        if x.dtype != self.W.dtype:
            x = x.astype(self.W.dtype)
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def to(self, dtype):
        self.dtype = dtype
        self.W = self.W.astype(dtype)
        if self.b is not None:
            self.b = self.b.astype(dtype)
        return self

    def float(self):
        return self.to(np.float32)

    def double(self):
        return self.to(np.float64)

    def half(self):
        return self.to(np.float16)

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={self.b is not None}, dtype={self.dtype}, init='{self.init}')"






# def test_linear():
#     print("\n" + "=" * 60)
#     print("TESTING LINEAR LAYER")
#     print("=" * 60)

#     print("\n===== 1. Constructor =====")
#     linear = Linear(10, 5)
#     print(linear)
#     print(f"W shape: {linear.W.shape}")
#     print(f"W dtype: {linear.W.dtype}")
#     print(f"W requires_grad: {linear.W.requires_grad}")
#     if linear.b is not None:
#         print(f"b shape: {linear.b.shape}")
#         print(f"b dtype: {linear.b.dtype}")
#         print(f"b requires_grad: {linear.b.requires_grad}")
#     assert linear.W.shape == (10, 5)
#     assert linear.b.shape == (5,)
#     print("[PASS] Constructor works")

#     print("\n===== 2. Forward Pass =====")
#     x = Tensor.randn((4, 10))
#     y = linear(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {y.shape}")
#     assert y.shape == (4, 5)
#     print("[PASS] Forward pass works")

#     print("\n===== 3. Dtype Conversion (half) =====")
#     linear.half()
#     print(f"W dtype after half(): {linear.W.dtype}")
#     if linear.b is not None:
#         print(f"b dtype after half(): {linear.b.dtype}")
#     assert linear.W.dtype == np.float16
#     assert linear.b.dtype == np.float16
#     print("[PASS] half() works")

#     print("\n===== 4. Dtype Conversion (double) =====")
#     linear.double()
#     print(f"W dtype after double(): {linear.W.dtype}")
#     if linear.b is not None:
#         print(f"b dtype after double(): {linear.b.dtype}")
#     assert linear.W.dtype == np.float64
#     assert linear.b.dtype == np.float64
#     print("[PASS] double() works")

#     print("\n===== 5. Dtype Conversion (float) =====")
#     linear.float()
#     print(f"W dtype after float(): {linear.W.dtype}")
#     if linear.b is not None:
#         print(f"b dtype after float(): {linear.b.dtype}")
#     assert linear.W.dtype == np.float32
#     assert linear.b.dtype == np.float32
#     print("[PASS] float() works")

#     print("\n===== 6. Input Dtype Auto-Cast =====")
#     linear.half()
#     x = Tensor.randn((2, 10), dtype=np.float32)
#     print(f"Input dtype: {x.dtype}")
#     print(f"Weight dtype: {linear.W.dtype}")
#     y = linear(x)
#     print(f"Output dtype: {y.dtype}")
#     assert y.shape == (2, 5)
#     print("[PASS] Auto-casting works")

#     print("\n===== 7. Weight Initialization (Kaiming) =====")
#     kaiming = Linear(100, 50, init="kaiming")
#     std = kaiming.W.data.std()
#     print(f"Kaiming std: {std}")
#     assert std > 0
#     print("[PASS] Kaiming initialization works")

#     print("\n===== 8. Weight Initialization (Xavier) =====")
#     xavier = Linear(100, 50, init="xavier")
#     std = xavier.W.data.std()
#     print(f"Xavier std: {std}")
#     assert std > 0
#     print("[PASS] Xavier initialization works")

#     print("\n===== 9. Weight Initialization (Normal) =====")
#     normal = Linear(100, 50, init="normal")
#     std = normal.W.data.std()
#     print(f"Normal std: {std}")
#     assert std > 0
#     print("[PASS] Normal initialization works")

#     print("\n===== 10. Parameter Check (requires_grad) =====")
#     linear = Linear(10, 5)
#     print(f"W requires_grad: {linear.W.requires_grad}")
#     if linear.b is not None:
#         print(f"b requires_grad: {linear.b.requires_grad}")
#     assert linear.W.requires_grad
#     assert linear.b.requires_grad
#     print("[PASS] Parameters registered with requires_grad")

#     print("\n===== 11. Backward Pass =====")
#     linear = Linear(5, 3, dtype=np.float32)
#     x = Tensor.randn((2, 5), requires_grad=True, dtype=np.float32)
#     y = linear(x)
#     loss = y.sum()
#     loss.backward()
#     print(f"W.grad shape: {linear.W.grad.shape}")
#     print(f"b.grad shape: {linear.b.grad.shape}")
#     print(f"x.grad shape: {x.grad.shape}")
#     assert linear.W.grad is not None
#     assert linear.b.grad is not None
#     assert x.grad is not None
#     print("[PASS] Backward pass works")

#     print("\n===== 12. No Bias Linear =====")
#     linear_no_bias = Linear(10, 5, bias=False)
#     assert linear_no_bias.b is None
#     x = Tensor.randn((2, 10))
#     y = linear_no_bias(x)
#     assert y.shape == (2, 5)
#     print("[PASS] No-bias linear works")

#     print("\n===== 13. Mixed Dtype Handling =====")
#     linear_fp16 = Linear(10, 5, dtype=np.float16)
#     x_fp32 = Tensor.randn((2, 10), dtype=np.float32)
#     print(f"Input dtype: {x_fp32.dtype}, Linear dtype: {linear_fp16.W.dtype}")
#     y = linear_fp16(x_fp32)
#     print(f"Output dtype: {y.dtype}")
#     assert y.shape == (2, 5)
#     print("[PASS] Mixed dtype handling works")

#     print("\n===== 14. Sequential Operations =====")
#     linear1 = Linear(10, 8, dtype=np.float32)
#     linear2 = Linear(8, 5, dtype=np.float32)
#     x = Tensor.randn((2, 10), requires_grad=True, dtype=np.float32)
#     h = linear1(x)
#     y = linear2(h)
#     loss = y.sum()
#     loss.backward()
#     assert linear1.W.grad is not None
#     assert linear2.W.grad is not None
#     print(f"linear1.W.grad shape: {linear1.W.grad.shape}")
#     print(f"linear2.W.grad shape: {linear2.W.grad.shape}")
#     print("[PASS] Sequential operations work")

#     print("\n" + "=" * 60)
#     print("ALL TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_linear()



