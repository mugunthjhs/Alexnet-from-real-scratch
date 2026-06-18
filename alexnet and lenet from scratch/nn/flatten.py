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
