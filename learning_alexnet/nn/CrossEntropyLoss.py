import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor

try:
    from .module import Module
except ImportError:
    from module import Module


class CrossEntropyLoss(Module):
    """
    Numerically stable softmax cross-entropy loss.

    Forward  : Tensor ops  (.exp / .sum / .log / /)
    Backward : explicit fused gradient  d = softmax - one_hot
    probs is computed once in forward and closed over in _backward.
    Expects raw logits (N, C) and integer class indices (N,).
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(self, logits: Tensor, targets) -> Tensor:
        """
        logits  : Tensor (N, C) — raw unnormalised scores
        targets : Tensor or array-like (N,) — integer class indices in [0, C)
        """
        if logits.ndim != 2:
            raise ValueError(f"logits must be 2D (N, C), got shape {logits.shape}")
        N, C = logits.shape

        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
        t = targets.ravel().astype(int).data          # int numpy array — index only, no grad

        if t.shape[0] != N:
            raise ValueError(f"targets length {t.shape[0]} does not match batch size {N}")
        if t.min() < 0 or t.max() >= C:
            raise ValueError(f"class index out of range [0, {C})")

        row_idx = Tensor.arange(N, dtype=int).data    # [0 … N-1]  for fancy indexing

        # ---- Forward using Tensor ops ----------------------------------------
        max_vals    = Tensor(logits.max(axis=1, keepdims=True).data)  # (N,1) detached
        shifted     = logits - max_vals                                # (N,C)
        exp_shifted = shifted.exp()                                    # (N,C)
        sum_exp     = exp_shifted.sum(axis=1, keepdims=True)           # (N,1)
        probs       = (exp_shifted / sum_exp).data                     # (N,C)  reused in backward
        log_sum_exp = sum_exp.log()                                    # (N,1)
        log_probs   = (shifted - log_sum_exp).data                     # (N,C) numpy — extract data

        correct_log_probs = -log_probs[row_idx, t]                    # (N,)

        if self.reduction == 'mean':
            loss_val = correct_log_probs.mean()
        elif self.reduction == 'sum':
            loss_val = correct_log_probs.sum()
        else:
            loss_val = correct_log_probs                               # (N,)

        out = Tensor(loss_val, requires_grad=logits.requires_grad, is_leaf=False)

        # ---- Backward --------------------------------------------------------
        if logits.requires_grad:
            _logits, _probs, _t, _N, _row = logits, probs, t, N, row_idx
            _reduction = self.reduction

            def _backward():
                if out.grad is None:
                    return
                # d(CE)/d(logit_j) = softmax_j - 1{j == true_class}
                d = _probs.copy()                      # (N, C)
                d[_row, _t] -= 1.0

                if _reduction == 'mean':
                    grad = out.grad * d / _N
                elif _reduction == 'sum':
                    grad = out.grad * d
                else:                                  # 'none': out.grad is (N,)
                    grad = out.grad[:, None] * d

                _logits.grad = grad if _logits.grad is None else _logits.grad + grad

            out._backward = _backward
            out._prev     = {logits}
            out._op       = "cross_entropy"

        return out

    def __repr__(self):
        return f"CrossEntropyLoss(reduction='{self.reduction}')"









# # ============================================================ #
# #  Tests                                                        #
# # ============================================================ #

# def test_cross_entropy_loss():
#     import numpy as np

#     print("\n" + "=" * 60)
#     print("TESTING CROSS ENTROPY LOSS")
#     print("=" * 60)

#     # shared fixtures
#     logits_data = [[1.0, 2.0, 3.0],
#                    [1.0, 2.0, 0.0]]
#     targets     = [2, 0]

#     x       = np.array(logits_data, dtype=np.float32)
#     shifted = x - x.max(axis=1, keepdims=True)
#     exp_x   = np.exp(shifted)
#     probs   = exp_x / exp_x.sum(axis=1, keepdims=True)   # (2, 3)

#     # ------------------------------------------------------------------ #
#     print("\n===== 1. repr =====")
#     assert repr(CrossEntropyLoss())       == "CrossEntropyLoss(reduction='mean')"
#     assert repr(CrossEntropyLoss('sum'))  == "CrossEntropyLoss(reduction='sum')"
#     assert repr(CrossEntropyLoss('none')) == "CrossEntropyLoss(reduction='none')"
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 2. Forward — reduction='mean' =====")
#     logits = Tensor(logits_data, requires_grad=False)
#     expected = -np.log(probs[[0, 1], [2, 0]]).mean()
#     loss = CrossEntropyLoss('mean')(logits, targets)
#     assert np.isclose(loss.data, expected, atol=1e-5), \
#         f"expected {expected:.6f}, got {loss.data:.6f}"
#     print(f"  loss={loss.data:.6f}  expected={expected:.6f}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 3. Forward — reduction='sum' =====")
#     expected_s = -np.log(probs[[0, 1], [2, 0]]).sum()
#     loss_s = CrossEntropyLoss('sum')(logits, targets)
#     assert np.isclose(loss_s.data, expected_s, atol=1e-5)
#     print(f"  loss={loss_s.data:.6f}  expected={expected_s:.6f}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 4. Forward — reduction='none' =====")
#     expected_n = -np.log(probs[[0, 1], [2, 0]])
#     loss_n = CrossEntropyLoss('none')(logits, targets)
#     assert loss_n.shape == (2,)
#     assert np.allclose(loss_n.data, expected_n, atol=1e-5)
#     print(f"  per-sample={loss_n.data}  expected={expected_n}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 5. Backward — reduction='mean' gradient =====")
#     logits2 = Tensor(logits_data, requires_grad=True)
#     loss2   = CrossEntropyLoss('mean')(logits2, targets)
#     loss2.backward()

#     d = probs.copy()
#     d[[0, 1], [2, 0]] -= 1.0
#     expected_g = d / 2
#     assert np.allclose(logits2.grad, expected_g, atol=1e-5), \
#         f"\nexpected:\n{expected_g}\ngot:\n{logits2.grad}"
#     print(f"  grad =\n{logits2.grad}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 6. Backward — reduction='sum' gradient =====")
#     logits3 = Tensor(logits_data, requires_grad=True)
#     loss3   = CrossEntropyLoss('sum')(logits3, targets)
#     loss3.backward()

#     d_s = probs.copy()
#     d_s[[0, 1], [2, 0]] -= 1.0
#     assert np.allclose(logits3.grad, d_s, atol=1e-5)
#     print(f"  grad =\n{logits3.grad}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 7. Backward — reduction='none' gradient =====")
#     logits4  = Tensor(logits_data, requires_grad=True)
#     loss4    = CrossEntropyLoss('none')(logits4, targets)
#     upstream = np.ones(2, dtype=np.float32)
#     loss4.backward(upstream)

#     d_n = probs.copy()
#     d_n[[0, 1], [2, 0]] -= 1.0
#     d_n *= upstream[:, None]
#     assert np.allclose(logits4.grad, d_n, atol=1e-5)
#     print(f"  grad =\n{logits4.grad}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 8. Gradient row-sums must be zero =====")
#     assert np.allclose(logits2.grad.sum(axis=1), 0.0, atol=1e-6)
#     print(f"  row sums = {logits2.grad.sum(axis=1)}  (~0)")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 9. Numerical stability — very large logits =====")
#     big = Tensor([[1000.0, 1001.0, 999.0]], requires_grad=True)
#     loss_big = CrossEntropyLoss()(big, [1])
#     assert np.isfinite(loss_big.data), "loss is not finite"
#     loss_big.backward()
#     assert np.all(np.isfinite(big.grad)), "grad is not finite"
#     print(f"  loss={loss_big.data:.6f}  grad={big.grad}  (finite)")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 10. Gradient accumulation =====")
#     logits5 = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
#     CrossEntropyLoss()(logits5, [2]).backward()
#     grad1 = logits5.grad.copy()
#     CrossEntropyLoss()(logits5, [2]).backward()
#     assert np.allclose(logits5.grad, grad1 * 2, atol=1e-6)
#     print(f"  after 1st: {grad1}")
#     print(f"  after 2nd: {logits5.grad}  (doubled )")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 11. requires_grad=False — no backward registered =====")
#     logits6 = Tensor([[1.0, 2.0, 3.0]], requires_grad=False)
#     loss6   = CrossEntropyLoss()(logits6, [0])
#     assert not loss6.requires_grad
#     assert loss6._op == ""
#     print(f"  loss={loss6.data:.6f}  _op='{loss6._op}'  (no backward)")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 12. Targets as list / ndarray / Tensor =====")
#     base = Tensor([[1.0, 2.0, 3.0]], requires_grad=False)
#     l1 = CrossEntropyLoss()(base, [2])
#     l2 = CrossEntropyLoss()(base, np.array([2]))
#     l3 = CrossEntropyLoss()(base, Tensor([2]))
#     assert np.isclose(l1.data, l2.data) and np.isclose(l2.data, l3.data)
#     print(f"  list={l1.data:.6f}  ndarray={l2.data:.6f}  Tensor={l3.data:.6f}  (all equal)")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 13. ValueError — logits not 2D =====")
#     try:
#         CrossEntropyLoss()(Tensor([1.0, 2.0, 3.0]), [0])
#         assert False, "should have raised"
#     except ValueError as e:
#         print(f"  caught: {e}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 14. ValueError — targets length mismatch =====")
#     try:
#         CrossEntropyLoss()(Tensor([[1.0, 2.0]]), [0, 1])
#         assert False, "should have raised"
#     except ValueError as e:
#         print(f"  caught: {e}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 15. ValueError — class index out of range =====")
#     try:
#         CrossEntropyLoss()(Tensor([[1.0, 2.0]]), [5])
#         assert False, "should have raised"
#     except ValueError as e:
#         print(f"  caught: {e}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 16. ValueError — invalid reduction =====")
#     try:
#         CrossEntropyLoss(reduction='invalid')
#         assert False, "should have raised"
#     except ValueError as e:
#         print(f"  caught: {e}")
#     print("[PASS]")

#     print("\n" + "=" * 60)
#     print("ALL 16 TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_cross_entropy_loss()
