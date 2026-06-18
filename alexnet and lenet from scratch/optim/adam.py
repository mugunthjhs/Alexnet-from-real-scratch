import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor


class Adam:
    """
    Adam optimiser (Adaptive Moment Estimation).

    Maintains per-parameter first and second moment estimates and applies
    bias correction so early steps are not over-shrunk.

    Update rule:
        m_t = beta1*m_{t-1} + (1-beta1)*g_t
        v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        theta -= lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        parameters,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        b1_t = self.beta1 ** self.t
        b2_t = self.beta2 ** self.t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * g ** 2
            m_hat = self._m[i] / (1.0 - b1_t)
            v_hat = self._v[i] / (1.0 - b2_t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
