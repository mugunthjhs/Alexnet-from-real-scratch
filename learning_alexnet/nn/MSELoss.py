import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor

try:
    from .module import Module
except ImportError:
    from module import Module


class MSELoss(Module):
    """
    Mean Squared Error loss.

    Forward  : Tensor ops  (sub / mul)
    Backward : explicit gradient  2*(pred - target) / N
    Expects predictions and targets of identical shape.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets) -> Tensor:
        """
        predictions : Tensor           — model outputs (any shape)
        targets     : Tensor / array   — ground truth, same shape as predictions
        """
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        if predictions.shape != targets.shape:
            raise ValueError(
                f"predictions shape {predictions.shape} does not match "
                f"targets shape {targets.shape}"
            )

        # ---- Forward using Tensor ops ----------------------------------------
        diff      = predictions - targets           # Tensor (same shape as input)
        sq        = (diff * diff).data              # numpy  — element-wise square
        diff_data = diff.data                       # numpy  — kept for backward

        N = predictions.data.size                   # total number of elements

        if self.reduction == 'mean':
            loss_val = sq.mean()
        elif self.reduction == 'sum':
            loss_val = sq.sum()
        else:
            loss_val = sq                           # same shape as predictions

        out = Tensor(loss_val, requires_grad=predictions.requires_grad, is_leaf=False)

        # ---- Backward --------------------------------------------------------
        if predictions.requires_grad:
            _predictions, _diff, _N = predictions, diff_data, N
            _reduction = self.reduction

            def _backward():
                if out.grad is None:
                    return
                # d(MSE)/d(pred_i) = 2*(pred_i - target_i)        for sum
                #                  = 2*(pred_i - target_i) / N     for mean
                if _reduction == 'mean':
                    grad = out.grad * 2.0 * _diff / _N
                elif _reduction == 'sum':
                    grad = out.grad * 2.0 * _diff
                else:                               # 'none': out.grad same shape as predictions
                    grad = out.grad * 2.0 * _diff

                _predictions.grad = grad if _predictions.grad is None else _predictions.grad + grad

            out._backward = _backward
            out._prev     = {predictions}
            out._op       = "mse"

        return out

    def __repr__(self):
        return f"MSELoss(reduction='{self.reduction}')"








# # ============================================================ #
# #  Tests                                                        #
# # ============================================================ #

# def test_mse_loss():
#     import numpy as np

#     print("\n" + "=" * 60)
#     print("TESTING MSE LOSS")
#     print("=" * 60)

#     # shared fixtures — clean numbers for easy verification
#     pred_data   = [[1.0, 2.0], [3.0, 4.0]]
#     target_data = [[1.5, 1.5], [2.5, 3.5]]
#     # diff = [[-0.5, 0.5], [0.5, 0.5]]
#     # sq   = [[0.25, 0.25], [0.25, 0.25]]
#     # N    = 4

#     pred_np   = np.array(pred_data,   dtype=np.float32)
#     target_np = np.array(target_data, dtype=np.float32)
#     diff_np   = pred_np - target_np
#     sq_np     = diff_np ** 2
#     N         = pred_np.size

#     # ------------------------------------------------------------------ #
#     print("\n===== 1. repr =====")
#     assert repr(MSELoss())        == "MSELoss(reduction='mean')"
#     assert repr(MSELoss('sum'))   == "MSELoss(reduction='sum')"
#     assert repr(MSELoss('none'))  == "MSELoss(reduction='none')"
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 2. Forward -- reduction='mean' =====")
#     pred   = Tensor(pred_data,   requires_grad=False)
#     target = Tensor(target_data, requires_grad=False)
#     loss   = MSELoss('mean')(pred, target)
#     expected = sq_np.mean()
#     assert np.isclose(loss.data, expected, atol=1e-6), \
#         f"expected {expected:.6f}, got {loss.data:.6f}"
#     print(f"  loss={loss.data:.6f}  expected={expected:.6f}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 3. Forward -- reduction='sum' =====")
#     loss_s   = MSELoss('sum')(pred, target)
#     expected_s = sq_np.sum()
#     assert np.isclose(loss_s.data, expected_s, atol=1e-6)
#     print(f"  loss={loss_s.data:.6f}  expected={expected_s:.6f}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 4. Forward -- reduction='none' =====")
#     loss_n = MSELoss('none')(pred, target)
#     assert loss_n.shape == (2, 2)
#     assert np.allclose(loss_n.data, sq_np, atol=1e-6)
#     print(f"  loss=\n{loss_n.data}")
#     print(f"  expected=\n{sq_np}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 5. Backward -- reduction='mean' gradient =====")
#     pred2   = Tensor(pred_data, requires_grad=True)
#     loss2   = MSELoss('mean')(pred2, target_data)
#     loss2.backward()
#     expected_g = 2.0 * diff_np / N
#     assert np.allclose(pred2.grad, expected_g, atol=1e-6), \
#         f"\nexpected:\n{expected_g}\ngot:\n{pred2.grad}"
#     print(f"  grad=\n{pred2.grad}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 6. Backward -- reduction='sum' gradient =====")
#     pred3   = Tensor(pred_data, requires_grad=True)
#     loss3   = MSELoss('sum')(pred3, target_data)
#     loss3.backward()
#     expected_gs = 2.0 * diff_np
#     assert np.allclose(pred3.grad, expected_gs, atol=1e-6)
#     print(f"  grad=\n{pred3.grad}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 7. Backward -- reduction='none' gradient =====")
#     pred4    = Tensor(pred_data, requires_grad=True)
#     loss4    = MSELoss('none')(pred4, target_data)
#     upstream = np.ones_like(pred_np)
#     loss4.backward(upstream)
#     expected_gn = upstream * 2.0 * diff_np
#     assert np.allclose(pred4.grad, expected_gn, atol=1e-6)
#     print(f"  grad=\n{pred4.grad}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 8. Zero gradient when pred == target =====")
#     pred5  = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
#     loss5  = MSELoss()(pred5, [[1.0, 2.0], [3.0, 4.0]])
#     loss5.backward()
#     assert np.allclose(loss5.data, 0.0, atol=1e-8)
#     assert np.allclose(pred5.grad, 0.0, atol=1e-8)
#     print(f"  loss={loss5.data}  grad=\n{pred5.grad}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 9. Gradient accumulation =====")
#     pred6  = Tensor([[1.0, 2.0]], requires_grad=True)
#     MSELoss()(pred6, [[0.0, 0.0]]).backward()
#     g1 = pred6.grad.copy()
#     MSELoss()(pred6, [[0.0, 0.0]]).backward()
#     assert np.allclose(pred6.grad, g1 * 2, atol=1e-6)
#     print(f"  after 1st: {g1}")
#     print(f"  after 2nd: {pred6.grad}  (doubled)")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 10. requires_grad=False -- no backward registered =====")
#     pred7  = Tensor([[1.0, 2.0]], requires_grad=False)
#     loss7  = MSELoss()(pred7, [[0.0, 0.0]])
#     assert not loss7.requires_grad
#     assert loss7._op == ""
#     print(f"  loss={loss7.data:.6f}  _op='{loss7._op}'  (no backward)")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 11. Targets as list / ndarray / Tensor =====")
#     base   = Tensor([[1.0, 2.0]], requires_grad=False)
#     l1 = MSELoss()(base, [[0.0, 0.0]])
#     l2 = MSELoss()(base, np.zeros((1, 2), dtype=np.float32))
#     l3 = MSELoss()(base, Tensor([[0.0, 0.0]]))
#     assert np.isclose(l1.data, l2.data) and np.isclose(l2.data, l3.data)
#     print(f"  list={l1.data:.6f}  ndarray={l2.data:.6f}  Tensor={l3.data:.6f}  (all equal)")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 12. 1D predictions =====")
#     pred8  = Tensor([1.0, 2.0, 3.0], requires_grad=True)
#     loss8  = MSELoss()(pred8, [0.0, 0.0, 0.0])
#     loss8.backward()
#     expected_1d = (1.0 + 4.0 + 9.0) / 3.0
#     assert np.isclose(loss8.data, expected_1d, atol=1e-6)
#     print(f"  loss={loss8.data:.6f}  expected={expected_1d:.6f}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 13. ValueError -- shape mismatch =====")
#     try:
#         MSELoss()(Tensor([[1.0, 2.0]]), Tensor([1.0]))
#         assert False, "should have raised"
#     except ValueError as e:
#         print(f"  caught: {e}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 14. ValueError -- invalid reduction =====")
#     try:
#         MSELoss(reduction='invalid')
#         assert False, "should have raised"
#     except ValueError as e:
#         print(f"  caught: {e}")
#     print("[PASS]")

#     print("\n" + "=" * 60)
#     print("ALL 14 TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_mse_loss()
