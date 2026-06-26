import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tensor import Tensor


class DataLoader:
    """
    Wraps a dataset and yields mini-batches of Tensors.

    Parameters
    ----------
    X            : Tensor or array-like  (N, ...)   — inputs
    y            : Tensor or array-like  (N, ...)   — labels / targets
    batch_size   : int                              — samples per batch
    shuffle      : bool                             — shuffle before every epoch
    drop_last    : bool                             — drop the final incomplete batch

    Dtype is taken from the input — pass a Tensor with the desired dtype
    and it will be preserved as-is.
    """

    def __init__(
        self,
        X,
        y,
        batch_size: int  = 32,
        shuffle:    bool = True,
        drop_last:  bool = False,
    ):
        self.X          = X.data if isinstance(X, Tensor) else np.asarray(X)
        self.y          = y.data if isinstance(y, Tensor) else np.asarray(y)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last
        self.N          = len(self.X)

        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got {len(self.X)} and {len(self.y)}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    def __len__(self):
        """Number of batches per epoch."""
        if self.drop_last:
            return self.N // self.batch_size
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = Tensor.rand(self.N).data.argsort().astype(np.int32)
        else:
            indices = Tensor.arange(self.N, dtype=np.int32).data

        for start in range(0, self.N, self.batch_size):
            end = start + self.batch_size
            if self.drop_last and end > self.N:
                break
            batch_idx = indices[start:end]
            yield (
                Tensor(self.X[batch_idx]),
                Tensor(self.y[batch_idx]),
            )

    def __repr__(self):
        return (
            f"DataLoader(N={self.N}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, drop_last={self.drop_last}, "
            f"x_dtype={self.X.dtype}, y_dtype={self.y.dtype})"
        )
