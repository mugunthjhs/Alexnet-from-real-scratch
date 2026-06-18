import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor


class SGD:
    """
    Stochastic Gradient Descent with optional momentum and L2 weight decay.

    Update rule (with momentum):
        v_t = momentum * v_{t-1} + g_t
        theta_t = theta_{t-1} - lr * v_t
    """

    def __init__(
        self,
        parameters,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        self.params = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            if self.momentum != 0.0:
                self._velocity[i] = self.momentum * self._velocity[i] + g
                g = self._velocity[i]
            p.data -= self.lr * g

    def zero_grad(self):
        for p in self.params:
            p.grad = None
