

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import (
    Module,
    Sequential,
    Conv2D,
    ReLU,
    LocalResponseNorm,
    MaxPool2D,
    Dropout,
    Flatten,
    Linear,
)


class AlexNet(Module):
    """The 2012 ImageNet-winning CNN, reproduced from the paper.

    Args:
        num_classes: Size of the final softmax layer (default 1000, the
            ILSVRC class count).
        dropout: Drop probability for the two FC dropout layers (paper: 0.5).
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes

        # Five convolutional layers (the two LRN-then-pool blocks, then the
        # conv3-4-5 stack that runs without intervening pool/norm).
        self.features = Sequential(
            # conv1: 3 -> 96, 11x11, stride 4, pad 2
            Conv2D(3, 96, kernel_size=11, stride=4, padding=2, init="normal"),
            ReLU(),
            LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            MaxPool2D(kernel_size=3, stride=2),
            # conv2: 96 -> 256, 5x5, pad 2
            Conv2D(96, 256, kernel_size=5, stride=1, padding=2, init="normal"),
            ReLU(),
            LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            MaxPool2D(kernel_size=3, stride=2),
            # conv3: 256 -> 384, 3x3, pad 1
            Conv2D(256, 384, kernel_size=3, stride=1, padding=1, init="normal"),
            ReLU(),
            # conv4: 384 -> 384, 3x3, pad 1
            Conv2D(384, 384, kernel_size=3, stride=1, padding=1, init="normal"),
            ReLU(),
            # conv5: 384 -> 256, 3x3, pad 1
            Conv2D(384, 256, kernel_size=3, stride=1, padding=1, init="normal"),
            ReLU(),
            MaxPool2D(kernel_size=3, stride=2),
        )

        self.flatten = Flatten(start_dim=1)

        # Three fully-connected layers; dropout on the first two.
        self.classifier = Sequential(
            Dropout(p=dropout),
            Linear(256 * 6 * 6, 4096, init="normal"),
            ReLU(),
            Dropout(p=dropout),
            Linear(4096, 4096, init="normal"),
            ReLU(),
            Linear(4096, num_classes, init="normal"),
        )

        self._init_biases()

    def _init_biases(self):
        """Set biases per Section 5: 1 for conv2/conv4/conv5 and the two
        hidden FC layers, 0 everywhere else. Weights are already N(0, 0.01)
        via init="normal"."""
        # features indices: conv1=0, conv2=4, conv3=8, conv4=10, conv5=12
        for idx in (4, 10, 12):
            self.features[idx].b.data[:] = 1.0
        # classifier indices: fc6=1, fc7=4, fc8=6 — bias=1 on the two hidden FCs
        for idx in (1, 4):
            self.classifier[idx].b.data[:] = 1.0

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def __repr__(self):
        return (
            f"AlexNet(num_classes={self.num_classes})\n"
            f"  (features): {repr(self.features)}\n"
            f"  (classifier): {repr(self.classifier)}"
        )
