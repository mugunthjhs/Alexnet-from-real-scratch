import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Numerically stable softmax cross-entropy loss.

    Combines log-softmax and negative log-likelihood in one pass to avoid
    intermediate overflow/underflow from exp().

    logits : (N, C) raw unnormalised scores
    targets: (N,)   integer class indices in [0, C)
    Returns scalar Tensor with requires_grad matching logits.
    """
    N = logits.shape[0]
    t = targets.data.astype(int).ravel()

    # Stable softmax: subtract row-max before exp
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    probs = exp_x / exp_x.sum(axis=1, keepdims=True)   # (N, C)

    # NLL of correct class
    correct_log_probs = -np.log(probs[np.arange(N), t] + 1e-12)
    loss_val = correct_log_probs.mean()

    out = Tensor(loss_val, requires_grad=logits.requires_grad, is_leaf=False)

    if logits.requires_grad:
        _logits, _probs, _t = logits, probs, t

        def _backward():
            if out.grad is None:
                return
            # Gradient of softmax cross-entropy: (probs - one_hot) / N
            d = _probs.copy()
            d[np.arange(N), _t] -= 1.0
            d /= N
            grad = out.grad * d
            _logits.grad = grad if _logits.grad is None else _logits.grad + grad

        out._backward = _backward
        out._prev = {logits}
        out._op = "cross_entropy"

    return out
