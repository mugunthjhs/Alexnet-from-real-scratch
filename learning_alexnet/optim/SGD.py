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




# # ============================================================ #
# #  Tests                                                        #
# # ============================================================ #

# def test_sgd():
#     print("\n" + "=" * 60)
#     print("TESTING SGD OPTIMISER")
#     print("=" * 60)

#     # ------------------------------------------------------------------ #
#     print("\n===== 1. repr =====")
#     opt = SGD([Tensor([1.0], requires_grad=True)], lr=0.01)
#     assert repr(opt) == "SGD(lr=0.01, momentum=0.0, weight_decay=0.0)"
#     print(f"  {opt}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 2. Basic step -- param -= lr * grad =====")
#     p   = Tensor([2.0, 4.0], requires_grad=True)
#     p.grad = np.array([1.0, 2.0])
#     opt = SGD([p], lr=0.1)
#     opt.step()
#     expected = np.array([1.9, 3.8])
#     assert np.allclose(p.data, expected), f"got {p.data}"
#     print(f"  param after step: {p.data}  expected: {expected}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 3. zero_grad clears gradients =====")
#     p2 = Tensor([1.0, 2.0], requires_grad=True)
#     p2.grad = np.array([1.0, 1.0])
#     opt2 = SGD([p2], lr=0.1)
#     opt2.zero_grad()
#     assert p2.grad is None
#     print(f"  p2.grad after zero_grad: {p2.grad}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 4. No update when grad is None =====")
#     p3 = Tensor([5.0, 5.0], requires_grad=True)
#     before = p3.data.copy()
#     SGD([p3], lr=0.1).step()
#     assert np.allclose(p3.data, before)
#     print(f"  param unchanged: {p3.data}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 5. Momentum =====")
#     p4 = Tensor([0.0], requires_grad=True)
#     opt4 = SGD([p4], lr=1.0, momentum=0.9)

#     # step 1: grad=1, v=1, param=0-1=-1
#     p4.grad = np.array([1.0])
#     opt4.step()
#     assert np.isclose(p4.data[0], -1.0), f"step1: {p4.data}"
#     print(f"  step1: {p4.data}  (expected [-1.])")

#     # step 2: grad=1, v=0.9*1+1=1.9, param=-1-1.9=-2.9
#     p4.grad = np.array([1.0])
#     opt4.step()
#     assert np.isclose(p4.data[0], -2.9), f"step2: {p4.data}"
#     print(f"  step2: {p4.data}  (expected [-2.9])")

#     # step 3: grad=1, v=0.9*1.9+1=2.71, param=-2.9-2.71=-5.61
#     p4.grad = np.array([1.0])
#     opt4.step()
#     assert np.isclose(p4.data[0], -5.61), f"step3: {p4.data}"
#     print(f"  step3: {p4.data}  (expected [-5.61])")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 6. Weight decay =====")
#     p5 = Tensor([2.0], requires_grad=True)
#     p5.grad = np.array([0.0])             # grad=0, only weight decay acts
#     opt5 = SGD([p5], lr=0.1, weight_decay=0.5)
#     opt5.step()
#     # effective grad = 0 + 0.5*2 = 1.0 → param = 2 - 0.1*1.0 = 1.9
#     assert np.isclose(p5.data[0], 1.9), f"got {p5.data}"
#     print(f"  param after step: {p5.data}  (expected [1.9])")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 7. Multiple parameters =====")
#     w = Tensor([1.0, 2.0], requires_grad=True)
#     b = Tensor([0.5],       requires_grad=True)
#     w.grad = np.array([0.1, 0.2])
#     b.grad = np.array([0.05])
#     opt6 = SGD([w, b], lr=1.0)
#     opt6.step()
#     assert np.allclose(w.data, [0.9, 1.8])
#     assert np.allclose(b.data, [0.45])
#     print(f"  w={w.data}  b={b.data}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 8. step() does not mutate .grad =====")
#     p7 = Tensor([3.0], requires_grad=True)
#     p7.grad = np.array([1.0])
#     original_grad = p7.grad.copy()
#     SGD([p7], lr=0.1).step()
#     assert np.allclose(p7.grad, original_grad)
#     print(f"  grad preserved: {p7.grad}")
#     print("[PASS]")

#     # ------------------------------------------------------------------ #
#     print("\n===== 9. ValueError -- bad hyperparameters =====")
#     p8 = Tensor([1.0], requires_grad=True)
#     for bad_kwargs, name in [
#         ({"lr": -0.1},           "negative lr"),
#         ({"momentum": 1.0},      "momentum == 1"),
#         ({"weight_decay": -0.1}, "negative weight_decay"),
#     ]:
#         try:
#             SGD([p8], **bad_kwargs)
#             assert False, f"should have raised for {name}"
#         except ValueError as e:
#             print(f"  {name}: caught {e}")
#     print("[PASS]")

#     print("\n" + "=" * 60)
#     print("ALL 9 TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_sgd()
