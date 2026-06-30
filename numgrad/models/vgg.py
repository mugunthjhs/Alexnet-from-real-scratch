import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from nn import (
    Module,
    Sequential,
    Conv2D,
    ReLU,
    MaxPool2D,
    Dropout,
    Flatten,
    Linear,
)

# Zero-mean normal weight init with variance 1e-2 (std = 0.1).
_PAPER_WEIGHT_STD = 0.1


class VGG19(Module):
    """The 19-weight-layer VGG (configuration E), reproduced from the paper.

    Args:
        num_classes: Size of the final softmax layer (default 1000, the
            ILSVRC class count).
        dropout: Drop probability for the two FC dropout layers (paper: 0.5).
        init: Weight initialisation. "normal" reproduces the paper exactly
            (N(0, 0.1^2), biases 0); "xavier" uses the Glorot scheme the paper
            mentions as a pre-training-free alternative.
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, init: str = "normal"):
        super().__init__()
        self.num_classes = num_classes
        self.init = init

        # Sixteen convolutional layers (3x3, stride 1, pad 1) in five blocks,
        # each block closed by a 2x2/stride-2 max-pool. Per-block conv counts
        # are 2, 2, 4, 4, 4.
        self.features = Sequential(
            # block 1: 3 -> 64
            Conv2D(3, 64, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            # block 2: 64 -> 128
            Conv2D(64, 128, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            # block 3: 128 -> 256  (four conv layers)
            Conv2D(128, 256, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            # block 4: 256 -> 512  (four conv layers)
            Conv2D(256, 512, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            # block 5: 512 -> 512  (four conv layers)
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, init=init),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
        )

        self.flatten = Flatten(start_dim=1)

        # Three fully-connected layers; dropout on the first two.
        self.classifier = Sequential(
            Dropout(p=dropout),
            Linear(512 * 7 * 7, 4096, init=init),
            ReLU(),
            Dropout(p=dropout),
            Linear(4096, 4096, init=init),
            ReLU(),
            Linear(4096, num_classes, init=init),
        )

        # The framework's init="normal" uses std=0.01; reset the weights to
        # std=0.1 here. Biases are already zero from construction.
        if init == "normal":
            self._init_paper_weights()

    def _init_paper_weights(self):
        """Reset every conv/FC weight to N(0, 0.1^2); biases stay zero."""
        for layer in list(self.features) + list(self.classifier):
            if isinstance(layer, (Conv2D, Linear)):
                w = Tensor.randn(*layer.W.shape).data * _PAPER_WEIGHT_STD
                layer.W.data[:] = w

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def __repr__(self):
        return (
            f"VGG19(num_classes={self.num_classes}, init='{self.init}')\n"
            f"  (features): {repr(self.features)}\n"
            f"  (classifier): {repr(self.classifier)}"
        )
