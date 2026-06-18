import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


class Linear(Module):
    """Fully-connected layer: out = x @ W + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming He initialisation (good default for layers followed by ReLU)
        std = np.sqrt(2.0 / in_features)
        self.W = Tensor(
            np.random.randn(in_features, out_features).astype(np.float32) * std,
            requires_grad=True,
        )
        if bias:
            self.b = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={self.b is not None})"
