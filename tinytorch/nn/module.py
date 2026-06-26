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
