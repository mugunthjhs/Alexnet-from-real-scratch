import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor


class SGD:
    """
    Stochastic Gradient Descent optimiser.

    Supports momentum and L2 weight decay.

    Update rule (per parameter):
        if weight_decay > 0:
            grad = grad + weight_decay * param

        if momentum > 0:
            velocity = momentum * velocity + grad
            param   -= lr * velocity
        else:
            param   -= lr * grad
    """

    def __init__(
        self,
        params,
        lr:           float = 1e-2,
        momentum:     float = 0.0,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        self.params       = list(params)          # list of Tensors
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay

        # one velocity buffer per parameter, initialised lazily on first step
        self._velocities  = [None] * len(self.params)

    def zero_grad(self):
        """Set all parameter gradients to None."""
        for p in self.params:
            p.grad = None

    def step(self):
        """Apply one gradient update to every parameter."""
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad.copy()                  # work on a copy — don't mutate .grad

            # L2 weight decay: add weight_decay * param to gradient
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            # Momentum
            if self.momentum != 0.0:
                if self._velocities[i] is None:
                    self._velocities[i] = grad.copy()
                else:
                    self._velocities[i] = self.momentum * self._velocities[i] + grad
                grad = self._velocities[i]

            p.data = p.data - self.lr * grad

    def __repr__(self):
        return (
            f"SGD(lr={self.lr}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay})"
        )
