import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor


class Module:
    """Base class for all neural network layers and models."""

    def __init__(self):
        self.__dict__['_parameters'] = {}
        self.__dict__['_modules'] = {}
        self.__dict__['training'] = True

    def __setattr__(self, name: str, value):
        _params = self.__dict__.get('_parameters', {})
        _mods = self.__dict__.get('_modules', {})
        _params.pop(name, None)
        _mods.pop(name, None)
        if isinstance(value, Tensor) and value.requires_grad:
            _params[name] = value
        elif isinstance(value, Module):
            _mods[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        """Yield all learnable parameters recursively."""
        for p in self._parameters.values():
            yield p
        for mod in self._modules.values():
            yield from mod.parameters()

    def zero_grad(self):
        """Set all parameter gradients to None."""
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True):
        """Switch to training mode (affects Dropout, BatchNorm)."""
        self.__dict__['training'] = mode
        for mod in self._modules.values():
            mod.train(mode)
        return self

    def eval(self):
        """Switch to evaluation mode."""
        return self.train(False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.forward() not implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        lines = [self.__class__.__name__ + "("]
        for name, mod in self._modules.items():
            lines.append(f"  ({name}): {repr(mod)}")
        lines.append(")")
        return "\n".join(lines)


class SimpleLinear(Module):
    def __init__(self):
        super().__init__()
        self.weight = Tensor(10.0, requires_grad=True)
        self.bias = Tensor(1.0, requires_grad=True)

    def forward(self, x):
        return x * self.weight + self.bias





# class SimpleModel(Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = SimpleLinear()
#         self.fc2 = SimpleLinear()


# def test_module():
#     print("\n" + "=" * 60)
#     print("TESTING MODULE CLASS")
#     print("=" * 60)

#     model = SimpleModel()

#     print("\n===== 1. Parameter Registration =====")
#     print(f"fc1 parameters: {list(model.fc1._parameters.keys())}")
#     assert 'weight' in model.fc1._parameters
#     assert 'bias' in model.fc1._parameters
#     print("[PASS] Parameter registration works")

#     print("\n===== 2. Module Registration =====")
#     print(f"Model modules: {list(model._modules.keys())}")
#     assert 'fc1' in model._modules
#     assert 'fc2' in model._modules
#     print("[PASS] Module registration works")

#     print("\n===== 3. parameters() Iterator =====")
#     params = list(model.parameters())
#     print(f"Total parameters: {len(params)}")
#     assert len(params) == 4
#     print("[PASS] parameters() iterator works")

#     print("\n===== 4. zero_grad() =====")
#     for p in model.parameters():
#         p.grad = 100.0

#     print("Before zero_grad():")
#     for i, p in enumerate(model.parameters()):
#         print(f"  param {i}: grad = {p.grad}")

#     model.zero_grad()

#     print("After zero_grad():")
#     for i, p in enumerate(model.parameters()):
#         print(f"  param {i}: grad = {p.grad}")
#         assert p.grad is None
#     print("[PASS] zero_grad() works")

#     print("\n===== 5. train() mode =====")
#     model.train()
#     print(f"Model training: {model.training}")
#     print(f"fc1 training: {model.fc1.training}")
#     print(f"fc2 training: {model.fc2.training}")
#     assert model.training == True
#     assert model.fc1.training == True
#     assert model.fc2.training == True
#     print("[PASS] train() mode works")

#     print("\n===== 6. eval() mode =====")
#     model.eval()
#     print(f"Model training: {model.training}")
#     print(f"fc1 training: {model.fc1.training}")
#     print(f"fc2 training: {model.fc2.training}")
#     assert model.training == False
#     assert model.fc1.training == False
#     assert model.fc2.training == False
#     print("[PASS] eval() mode works")

#     print("\n===== 7. forward() and __call__() =====")
#     model.train()
#     x = Tensor(5.0)
#     result1 = model.fc1.forward(x)
#     result2 = model.fc1(x)
#     print(f"forward(5) = {result1.data}")
#     print(f"__call__(5) = {result2.data}")
#     assert result1.data == result2.data
#     print("[PASS] forward() and __call__() work")

#     print("\n===== 8. Sequential Module Calls =====")
#     x = Tensor(2.0)
#     h = model.fc1(x)
#     y = model.fc2(h)
#     print(f"Input: {x.data}")
#     print(f"After fc1: {h.data}")
#     print(f"After fc2: {y.data}")
#     print("[PASS] Sequential module calls work")

#     print("\n===== 9. __repr__() =====")
#     print("Model representation:")
#     print(model)
#     print("[PASS] __repr__() works")

#     print("\n===== 10. Parameter Gradient Flow =====")
#     model.train()
#     x = Tensor(1.0, requires_grad=True)
#     out = model.fc1(x)
#     out.backward()

#     print(f"fc1.weight.grad = {model.fc1.weight.grad}")
#     print(f"fc1.bias.grad = {model.fc1.bias.grad}")
#     assert model.fc1.weight.grad is not None
#     assert model.fc1.bias.grad is not None
#     print("[PASS] Parameter gradient flow works")

#     print("\n" + "=" * 60)
#     print("ALL MODULE TESTS PASSED")
#     print("=" * 60)


# if __name__ == "__main__":
#     test_module()
