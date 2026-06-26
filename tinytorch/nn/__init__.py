"""Neural-network layers, containers and losses (NumPy autograd).

Public API — import directly from the package, e.g.::

    from nn import Conv2D, ReLU, MaxPool2D, Linear, Sequential, CrossEntropyLoss
"""

from .module import Module
from .sequential import Sequential

# Layers
from .linear import Linear
from .conv2d import Conv2D
from .maxpool import MaxPool2D
from .avgpool import AvgPool2D
from .flatten import Flatten
from .dropout import Dropout
from .lrn import LocalResponseNorm

# Activations
from .relu import ReLU
from .sigmoid import Sigmoid
from .tanh import Tanh

# Losses
from .loss import CrossEntropyLoss, MSELoss

__all__ = [
    "Module",
    "Sequential",
    "Linear",
    "Conv2D",
    "MaxPool2D",
    "AvgPool2D",
    "Flatten",
    "Dropout",
    "LocalResponseNorm",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "CrossEntropyLoss",
    "MSELoss",
]
