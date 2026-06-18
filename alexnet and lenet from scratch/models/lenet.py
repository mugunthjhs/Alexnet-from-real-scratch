import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
import nn


class LeNet5(nn.Module):
    """
    LeNet-5 adapted for 28x28 greyscale images (MNIST).

    Architecture:
        Conv1(1->6, 5x5, pad=2)  -> ReLU -> MaxPool(2x2)   [28x28 -> 14x14]
        Conv2(6->16, 5x5)        -> ReLU -> MaxPool(2x2)   [14x14 -> 5x5]
        Flatten -> FC(400->120)  -> ReLU
        FC(120->84)              -> ReLU
        FC(84->num_classes)                                 [logits]

    Original LeNet used tanh + AvgPool; this modern variant uses ReLU + MaxPool.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        self.conv1   = nn.Conv2D(1, 6, kernel_size=5, padding=2)
        self.pool1   = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2   = nn.Conv2D(6, 16, kernel_size=5, padding=0)
        self.pool2   = nn.MaxPool2D(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(16 * 5 * 5, 120)
        self.fc2     = nn.Linear(120, 84)
        self.fc3     = nn.Linear(84, num_classes)
        self.relu    = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv1(x))   # (N, 6, 28, 28)
        x = self.pool1(x)              # (N, 6, 14, 14)
        x = self.relu(self.conv2(x))   # (N, 16, 10, 10)
        x = self.pool2(x)              # (N, 16, 5, 5)
        x = self.flatten(x)            # (N, 400)
        x = self.relu(self.fc1(x))     # (N, 120)
        x = self.relu(self.fc2(x))     # (N, 84)
        x = self.fc3(x)                # (N, num_classes) — raw logits
        return x

    def __repr__(self):
        return f"LeNet5(num_classes={self.num_classes})"
