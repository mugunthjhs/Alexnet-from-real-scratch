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





# # ============================================================ #
# #  Tests                                                        #
# # ============================================================ #

# def test_dataloader():
#     print("\n" + "=" * 60)
#     print("TESTING DATALOADER")
#     print("=" * 60)

#     X = Tensor.arange(20, dtype=np.float32).reshape(10, 2)   # 10 samples, 2 features
#     y = Tensor.arange(10, dtype=np.int64)                     # 10 labels

#     # ------------------------------------------------------------------ #
#     print("\n===== 1. repr =====")
#     dl = DataLoader(X, y, batch_size=4, shuffle=False)
#     print(f"  {dl}")
#     assert "N=10" in repr(dl)
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 2. len() -- number of batches =====")
#     dl_nodrop = DataLoader(X, y, batch_size=4, shuffle=False, drop_last=False)
#     dl_drop   = DataLoader(X, y, batch_size=4, shuffle=False, drop_last=True)
#     assert len(dl_nodrop) == 3   # ceil(10/4) = 3
#     assert len(dl_drop)   == 2   # floor(10/4) = 2
#     print(f"  no drop_last: {len(dl_nodrop)} batches  drop_last: {len(dl_drop)} batches")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 3. All samples covered (shuffle=False) =====")
#     dl = DataLoader(X, y, batch_size=3, shuffle=False, drop_last=False)
#     seen = []
#     for xb, yb in dl:
#         assert isinstance(xb, Tensor)
#         assert isinstance(yb, Tensor)
#         seen.extend(yb.data.tolist())
#     assert sorted(seen) == list(range(10))
#     print(f"  labels seen: {seen}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 4. drop_last drops the final incomplete batch =====")
#     dl = DataLoader(X, y, batch_size=4, shuffle=False, drop_last=True)
#     batches = list(dl)
#     assert len(batches) == 2
#     for xb, yb in batches:
#         assert xb.shape[0] == 4
#     print(f"  batch sizes: {[b[0].shape[0] for b in batches]}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 5. Batch shapes are correct =====")
#     dl = DataLoader(X, y, batch_size=3, shuffle=False)
#     xb, yb = next(iter(dl))
#     assert xb.shape == (3, 2)
#     assert yb.shape == (3,)
#     print(f"  xb.shape={xb.shape}  yb.shape={yb.shape}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 6. shuffle changes order across epochs =====")
#     np.random.seed(0)
#     dl = DataLoader(X, y, batch_size=10, shuffle=True)
#     labels_e1 = list(next(iter(dl))[1].data)
#     labels_e2 = list(next(iter(dl))[1].data)
#     assert labels_e1 != labels_e2, "shuffle should produce different order"
#     print(f"  epoch1: {labels_e1}")
#     print(f"  epoch2: {labels_e2}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 7. batch_size >= N returns one batch =====")
#     dl = DataLoader(X, y, batch_size=100, shuffle=False)
#     batches = list(dl)
#     assert len(batches) == 1
#     assert batches[0][0].shape[0] == 10
#     print(f"  single batch of size {batches[0][0].shape[0]}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 8. ValueError -- mismatched X and y =====")
#     try:
#         DataLoader(X, Tensor.arange(5, dtype=np.int64), batch_size=4)
#         assert False
#     except ValueError as e:
#         print(f"  caught: {e}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 9. ValueError -- bad batch_size =====")
#     try:
#         DataLoader(X, y, batch_size=0)
#         assert False
#     except ValueError as e:
#         print(f"  caught: {e}")
#     print("[PASS]")

#     print("\n" + "=" * 60)
#     print("ALL 9 TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_dataloader()
