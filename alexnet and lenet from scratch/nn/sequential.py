from .module import Module
from tensor import Tensor


class Sequential(Module):
    """Chain layers so output of one feeds into the next."""

    def __init__(self, *layers):
        super().__init__()
        self.__dict__['_layers'] = list(layers)
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer
            object.__setattr__(self, str(i), layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._layers:
            x = layer(x)
        return x

    def __getitem__(self, idx):
        return self._layers[idx]

    def __len__(self):
        return len(self._layers)

    def __repr__(self):
        lines = ["Sequential("]
        for i, layer in enumerate(self._layers):
            lines.append(f"  ({i}): {repr(layer)}")
        lines.append(")")
        return "\n".join(lines)
