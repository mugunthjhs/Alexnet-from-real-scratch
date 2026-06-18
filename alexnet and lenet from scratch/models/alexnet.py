import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
import nn


class AlexNet(nn.Module):
    """
    AlexNet for RGB images.

    Default input size is 224x224 (matches ImageNette dataset).
    Pass input_size=227 to match the original paper.

    Architecture (for 224x224):
        Conv1(3->96, 11x11, stride=4)  -> ReLU -> LRN -> MaxPool(3,s=2)  [54 -> 26]
        Conv2(96->256, 5x5, pad=2)     -> ReLU -> LRN -> MaxPool(3,s=2)  [26 -> 12]
        Conv3(256->384, 3x3, pad=1)    -> ReLU                           [12]
        Conv4(384->384, 3x3, pad=1)    -> ReLU                           [12]
        Conv5(384->256, 3x3, pad=1)    -> ReLU -> MaxPool(3,s=2)         [12 -> 5]
        Flatten -> Dropout -> FC(fc_in->4096) -> ReLU
        Dropout  -> FC(4096->4096)            -> ReLU
        FC(4096->num_classes)                                             [logits]
    """

    def __init__(self, num_classes: int = 10, input_size: int = 224):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Compute flattened size after feature layers
        fc_in = self._feature_output_size(input_size)

        # Feature extractor
        self.conv1 = nn.Conv2D(3, 96, kernel_size=11, stride=4, padding=0)
        self.lrn1  = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        self.pool1 = nn.MaxPool2D(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2D(96, 256, kernel_size=5, stride=1, padding=2)
        self.lrn2  = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        self.pool2 = nn.MaxPool2D(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2D(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2D(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2D(384, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2D(kernel_size=3, stride=2)

        # Classifier
        self.flatten = nn.Flatten()
        self.drop1   = nn.Dropout(p=0.5)
        self.fc1     = nn.Linear(fc_in, 4096)
        self.drop2   = nn.Dropout(p=0.5)
        self.fc2     = nn.Linear(4096, 4096)
        self.fc3     = nn.Linear(4096, num_classes)
        self.relu    = nn.ReLU()

    @staticmethod
    def _feature_output_size(h: int) -> int:
        """Trace spatial dimension through feature layers to find fc input size."""
        h = (h - 11) // 4 + 1          # conv1
        h = (h - 3) // 2 + 1           # pool1
        h = h                           # conv2 (pad=2, k=5, stride=1 preserves size)
        h = (h - 3) // 2 + 1           # pool2
        # conv3/4/5 with pad=1 preserve size
        h = (h - 3) // 2 + 1           # pool3
        return 256 * h * h

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv1(x))
        x = self.lrn1(x)
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.lrn2(x)
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.drop1(x)
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)     # raw logits
        return x

    def __repr__(self):
        return f"AlexNet(num_classes={self.num_classes}, input_size={self.input_size})"
