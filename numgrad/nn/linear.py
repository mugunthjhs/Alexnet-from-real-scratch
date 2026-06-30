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
