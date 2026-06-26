import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from .module import Module


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
                # d(MSE)/d(pred_i) = 2*(pred_i - target_i)        for sum / none
                #                  = 2*(pred_i - target_i) / N     for mean
                if _reduction == 'mean':
                    grad = out.grad * 2.0 * _diff / _N
                else:                               # 'sum' or 'none'
                    grad = out.grad * 2.0 * _diff

                _predictions.grad = grad if _predictions.grad is None else _predictions.grad + grad

            out._backward = _backward
            out._prev     = {predictions}
            out._op       = "mse"

        return out

    def __repr__(self):
        return f"MSELoss(reduction='{self.reduction}')"
