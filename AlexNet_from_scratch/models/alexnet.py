"""
AlexNet: ImageNet Classification with Deep Convolutional Neural Networks

Krizhevsky, Sutskever & Hinton (2012)

Architecture (for ImageNet with 1000 classes, adapted for ImageNette with 10 classes):
Input: (224 x 224 x 3)
    Conv (96 filters, 11x11, stride 4) + ReLU + LRN
    MaxPool (3x3, stride 2)
    Dropout (p=0.5)
    → (54 x 54 x 96)
    
    Conv (256 filters, 5x5, padding 2, stride 1) + ReLU + LRN
    MaxPool (3x3, stride 2)
    Dropout (p=0.5)
    → (26 x 26 x 256)
    
    Conv (384 filters, 3x3, padding 1) + ReLU
    → (26 x 26 x 384)
    
    Conv (384 filters, 3x3, padding 1) + ReLU
    → (26 x 26 x 384)
    
    Conv (256 filters, 3x3, padding 1) + ReLU
    MaxPool (3x3, stride 2)
    Dropout (p=0.5)
    → (12 x 12 x 256) -> (5 x 5 x 256) = 6400 features
    
    Flatten → (6400)
    FC (4096) + ReLU + Dropout (p=0.5)
    → (4096)
    
    FC (4096) + ReLU + Dropout (p=0.5)
    → (4096)
    
    FC (num_classes) + Softmax
Output: (num_classes)

Key innovations:
1. ReLU activation (enables training of deeper networks)
2. Dropout (prevents co-adaptation, reduces overfitting)
3. Local Response Normalization (lateral inhibition)
4. Overlapping pooling (non-overlapping doesn't work well)
5. Multi-GPU training (split conv1-2 and conv4-5 across GPUs)

Total parameters: ~60 million (for 1000 classes)

This is Stage 10 and the final goal of the project.
"""

import numpy as np
from tensor import Tensor
import sys
sys.path.append('..')
from layers.conv2d import Conv2D
from layers.maxpool import MaxPool2D
from layers.linear import Linear
from layers.relu import ReLU
from layers.dropout import Dropout
from layers.lrn import LocalResponseNormalization


class AlexNet:
    """
    AlexNet: Deep Convolutional Neural Network for image classification.
    
    Input: (batch, 3, 224, 224) - RGB 224x224 images
    Output: (batch, num_classes) - logits
    """
    
    def __init__(self, num_classes: int = 1000):
        """
        Initialize AlexNet.
        
        Args:
            num_classes: Number of output classes (1000 for ImageNet, 10 for ImageNette)
        """
        # ===== Conv Block 1 =====
        # Input: (B, 3, 224, 224)
        # Conv (96 filters, 11x11, stride 4, padding 0)
        # Output: (B, 96, 54, 54)
        # Formula: (224 - 11) / 4 + 1 = 54
        self.conv1 = Conv2D(3, 96, kernel_size=11, stride=4, padding=0, bias=True)
        self.relu1 = ReLU()
        self.lrn1 = LocalResponseNormalization(size=5, alpha=1e-4, beta=0.75)
        # MaxPool (3x3, stride 2, no padding)
        # Output: (B, 96, 26, 26)
        # Formula: (54 - 3) / 2 + 1 = 26
        self.pool1 = MaxPool2D(pool_size=3, stride=2, padding=0)
        self.dropout1 = Dropout(p=0.5)
        
        # ===== Conv Block 2 =====
        # Input: (B, 96, 26, 26)
        # Conv (256 filters, 5x5, stride 1, padding 2)
        # Output: (B, 256, 26, 26)
        # Formula: (26 + 2*2 - 5) / 1 + 1 = 26
        self.conv2 = Conv2D(96, 256, kernel_size=5, stride=1, padding=2, bias=True)
        self.relu2 = ReLU()
        self.lrn2 = LocalResponseNormalization(size=5, alpha=1e-4, beta=0.75)
        # MaxPool (3x3, stride 2)
        # Output: (B, 256, 12, 12)
        # Formula: (26 - 3) / 2 + 1 = 12
        self.pool2 = MaxPool2D(pool_size=3, stride=2, padding=0)
        self.dropout2 = Dropout(p=0.5)
        
        # ===== Conv Block 3 =====
        # Input: (B, 256, 12, 12)
        # Conv (384 filters, 3x3, stride 1, padding 1)
        # Output: (B, 384, 12, 12)
        self.conv3 = Conv2D(256, 384, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3 = ReLU()
        
        # ===== Conv Block 4 =====
        # Input: (B, 384, 12, 12)
        # Conv (384 filters, 3x3, stride 1, padding 1)
        # Output: (B, 384, 12, 12)
        self.conv4 = Conv2D(384, 384, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu4 = ReLU()
        
        # ===== Conv Block 5 =====
        # Input: (B, 384, 12, 12)
        # Conv (256 filters, 3x3, stride 1, padding 1)
        # Output: (B, 256, 12, 12)
        self.conv5 = Conv2D(384, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu5 = ReLU()
        # MaxPool (3x3, stride 2)
        # Output: (B, 256, 5, 5) = 6400
        # Formula: (12 - 3) / 2 + 1 = 5
        self.pool5 = MaxPool2D(pool_size=3, stride=2, padding=0)
        self.dropout3 = Dropout(p=0.5)
        
        # ===== Fully Connected Layers =====
        # Flatten: (B, 256, 5, 5) -> (B, 6400)
        # FC (4096)
        self.fc1 = Linear(256 * 5 * 5, 4096)
        self.relu_fc1 = ReLU()
        self.dropout_fc1 = Dropout(p=0.5)
        
        # FC (4096)
        self.fc2 = Linear(4096, 4096)
        self.relu_fc2 = ReLU()
        self.dropout_fc2 = Dropout(p=0.5)
        
        # FC (num_classes)
        self.fc3 = Linear(4096, num_classes)
        
        self.num_classes = num_classes
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through AlexNet.
        
        Args:
            x: Input tensor (batch, 3, 224, 224)
            
        Returns:
            Output logits (batch, num_classes)
        """
        # ===== Conv Block 1 =====
        y = self.conv1(x)                      # (B, 96, 54, 54)
        y = self.relu1(y)
        y = self.lrn1(y)
        y = self.pool1(y)                      # (B, 96, 26, 26)
        y = self._apply_dropout(y, self.dropout1)
        
        # ===== Conv Block 2 =====
        y = self.conv2(y)                      # (B, 256, 26, 26)
        y = self.relu2(y)
        y = self.lrn2(y)
        y = self.pool2(y)                      # (B, 256, 12, 12)
        y = self._apply_dropout(y, self.dropout2)
        
        # ===== Conv Block 3 =====
        y = self.conv3(y)                      # (B, 384, 12, 12)
        y = self.relu3(y)
        
        # ===== Conv Block 4 =====
        y = self.conv4(y)                      # (B, 384, 12, 12)
        y = self.relu4(y)
        
        # ===== Conv Block 5 =====
        y = self.conv5(y)                      # (B, 256, 12, 12)
        y = self.relu5(y)
        y = self.pool5(y)                      # (B, 256, 5, 5)
        y = self._apply_dropout(y, self.dropout3)
        
        # ===== Flatten =====
        batch_size = y.shape[0]
        y = y.reshape(batch_size, -1)          # (B, 6400)
        
        # ===== Fully Connected Layers =====
        y = self.fc1(y)                        # (B, 4096)
        y = self.relu_fc1(y)
        y = self._apply_dropout(y, self.dropout_fc1)
        
        y = self.fc2(y)                        # (B, 4096)
        y = self.relu_fc2(y)
        y = self._apply_dropout(y, self.dropout_fc2)
        
        y = self.fc3(y)                        # (B, num_classes)
        
        return y
    
    def _apply_dropout(self, x: Tensor, dropout: Dropout) -> Tensor:
        """Apply dropout based on training mode."""
        if self.training:
            return dropout(x, training=True)
        else:
            return dropout(x, training=False)
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass when called as function."""
        return self.forward(x)
    
    def train(self):
        """Set to training mode."""
        self.training = True
        self.dropout1.train()
        self.dropout2.train()
        self.dropout3.train()
        self.dropout_fc1.train()
        self.dropout_fc2.train()
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
        self.dropout1.eval()
        self.dropout2.eval()
        self.dropout3.eval()
        self.dropout_fc1.eval()
        self.dropout_fc2.eval()
    
    def parameters(self):
        """Get all learnable parameters."""
        params = []
        # Conv layers
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            params.extend(layer.parameters())
        # FC layers
        for layer in [self.fc1, self.fc2, self.fc3]:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        """Reset all gradients."""
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5,
                     self.fc1, self.fc2, self.fc3]:
            layer.zero_grad()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"AlexNet(num_classes={self.num_classes})"


# ============================================================================
# Model Statistics
# ============================================================================

def count_parameters(model: AlexNet) -> int:
    """Count total number of parameters."""
    total = 0
    for param in model.parameters():
        total += param.size
    return total


def print_model_info(model: AlexNet):
    """Print detailed model information."""
    print(f"\n{model}")
    print("=" * 70)
    print("Layer                                   Params        Output Shape")
    print("-" * 70)
    
    # Conv1: (3, 96, 11, 11)
    conv1_params = (11*11*3+1) * 96
    print(f"Conv2D(3 → 96, 11×11, stride 4)    {conv1_params:>12,}  (B, 96, 54, 54)")
    
    # Conv2: (96, 256, 5, 5)
    conv2_params = (5*5*96+1) * 256
    print(f"Conv2D(96 → 256, 5×5, pad 2)       {conv2_params:>12,}  (B, 256, 26, 26)")
    
    # Conv3: (256, 384, 3, 3)
    conv3_params = (3*3*256+1) * 384
    print(f"Conv2D(256 → 384, 3×3, pad 1)      {conv3_params:>12,}  (B, 384, 12, 12)")
    
    # Conv4: (384, 384, 3, 3)
    conv4_params = (3*3*384+1) * 384
    print(f"Conv2D(384 → 384, 3×3, pad 1)      {conv4_params:>12,}  (B, 384, 12, 12)")
    
    # Conv5: (384, 256, 3, 3)
    conv5_params = (3*3*384+1) * 256
    print(f"Conv2D(384 → 256, 3×3, pad 1)      {conv5_params:>12,}  (B, 256, 5, 5)")
    
    # FC1: (6400, 4096)
    fc1_params = (6400+1) * 4096
    print(f"Linear(6400 → 4096)                 {fc1_params:>12,}  (B, 4096)")
    
    # FC2: (4096, 4096)
    fc2_params = (4096+1) * 4096
    print(f"Linear(4096 → 4096)                 {fc2_params:>12,}  (B, 4096)")
    
    # FC3: (4096, num_classes)
    fc3_params = (4096+1) * model.num_classes
    print(f"Linear(4096 → {model.num_classes})                   {fc3_params:>12,}  (B, {model.num_classes})")
    
    total = (conv1_params + conv2_params + conv3_params + conv4_params + conv5_params +
             fc1_params + fc2_params + fc3_params)
    
    print("-" * 70)
    print(f"Total Parameters                        {total:>12,}")
    print("=" * 70)


# ============================================================================
# Tests
# ============================================================================

def test_alexnet_forward():
    """Test forward pass with correct shapes."""
    print("Testing AlexNet Forward Pass...")
    
    model = AlexNet(num_classes=10)  # ImageNette
    
    # ImageNet input (or ImageNette for testing)
    x = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
    
    output = model(x)
    
    assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
    print(f"✓ Input shape (2, 3, 224, 224) → Output shape {output.shape}")


def test_alexnet_backward():
    """Test backward pass."""
    print("\nTesting AlexNet Backward Pass...")
    
    model = AlexNet(num_classes=10)
    
    x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32), requires_grad=True)
    output = model(x)
    
    # Compute loss (sum for simplicity)
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist
    param_count = len(model.parameters())
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    
    assert grad_count == param_count, \
        f"Only {grad_count}/{param_count} parameters have gradients"
    
    print(f"✓ Backward pass successful, {grad_count} parameters have gradients")


def test_alexnet_shapes():
    """Test intermediate shapes through network."""
    print("\nTesting AlexNet Intermediate Shapes...")
    
    model = AlexNet(num_classes=10)
    x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
    
    # Test critical shape transformations
    y = model.conv1(x)
    assert y.shape == (1, 96, 54, 54), f"After Conv1: {y.shape}"
    
    y = model.pool1(y)
    assert y.shape == (1, 96, 26, 26), f"After Pool1: {y.shape}"
    
    y = model.conv2(y)
    assert y.shape == (1, 256, 26, 26), f"After Conv2: {y.shape}"
    
    y = model.pool2(y)
    assert y.shape == (1, 256, 12, 12), f"After Pool2: {y.shape}"
    
    y = model.conv5(model.conv4(model.conv3(y)))
    assert y.shape == (1, 256, 12, 12), f"After Conv3-5: {y.shape}"
    
    y = model.pool5(y)
    assert y.shape == (1, 256, 5, 5), f"After Pool5: {y.shape}"
    
    print(f"✓ All intermediate shapes correct")


def test_alexnet_parameters():
    """Test parameter counting."""
    print("\nTesting AlexNet Parameters...")
    
    model = AlexNet(num_classes=1000)  # Original ImageNet
    total = count_parameters(model)
    
    # Approximate: ~60M parameters
    assert 50_000_000 < total < 70_000_000, f"Total {total} not in expected range"
    print(f"✓ Total parameters: {total:,} (~60M expected)")
    
    model10 = AlexNet(num_classes=10)
    total10 = count_parameters(model10)
    print(f"✓ ImageNette version: {total10:,}")


def test_alexnet_train_eval():
    """Test train/eval mode."""
    print("\nTesting AlexNet Train/Eval Modes...")
    
    model = AlexNet(num_classes=10)
    
    # Training mode
    model.train()
    x_train = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
    out_train1 = model(x_train)
    out_train2 = model(x_train)
    # Dropout is random, outputs should be different
    
    # Eval mode
    model.eval()
    x_eval = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
    out_eval1 = model(x_eval)
    out_eval2 = model(x_eval)
    # No dropout, outputs should be identical
    assert np.allclose(out_eval1.data, out_eval2.data), \
        "Eval mode should be deterministic"
    
    print(f"✓ Train/eval mode works correctly")


if __name__ == "__main__":
    test_alexnet_forward()
    test_alexnet_backward()
    test_alexnet_shapes()
    test_alexnet_parameters()
    test_alexnet_train_eval()
    
    print("\n" + "=" * 70)
    print_model_info(AlexNet(num_classes=10))
    
    print("\n✓ All AlexNet tests passed!")
