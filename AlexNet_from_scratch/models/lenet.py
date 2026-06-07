"""
LeNet-5: The First CNN Architecture

Architecture:
Input (28x28x1)
    → Conv (6 filters, 5x5) → ReLU → (24x24x6)
    → MaxPool (2x2) → (12x12x6)
    → Conv (16 filters, 5x5) → ReLU → (8x8x16)
    → MaxPool (2x2) → (4x4x16)
    → Flatten → (256)
    → Dense (120) → ReLU
    → Dense (84) → ReLU
    → Dense (10) → Softmax
Output (10 classes)

Total parameters: ~44K
Achieves ~98-99% accuracy on MNIST

This is Stage 9 of the project.
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


class LeNet5:
    """
    LeNet-5 Convolutional Neural Network.
    
    Input: (batch, 1, 28, 28) - grayscale 28x28 images
    Output: (batch, 10) - logits for 10 classes
    """
    
    def __init__(self, num_classes: int = 10):
        """
        Initialize LeNet-5.
        
        Args:
            num_classes: Number of output classes (default 10 for MNIST)
        """
        # Conv layer 1: 1 -> 6 channels, 5x5 kernel
        self.conv1 = Conv2D(1, 6, kernel_size=5, stride=1, padding=0, bias=True)
        self.relu1 = ReLU()
        
        # Max pool layer 1: 2x2, stride 2
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        
        # Conv layer 2: 6 -> 16 channels, 5x5 kernel
        self.conv2 = Conv2D(6, 16, kernel_size=5, stride=1, padding=0, bias=True)
        self.relu2 = ReLU()
        
        # Max pool layer 2: 2x2, stride 2
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        
        # Flatten: (batch, 16, 4, 4) -> (batch, 256)
        # This happens in forward pass
        
        # Dense layers
        self.fc1 = Linear(16 * 4 * 4, 120)
        self.relu3 = ReLU()
        
        self.fc2 = Linear(120, 84)
        self.relu4 = ReLU()
        
        self.fc3 = Linear(84, num_classes)
        
        self.num_classes = num_classes
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through LeNet-5.
        
        Args:
            x: Input tensor (batch, 1, 28, 28)
            
        Returns:
            Output logits (batch, num_classes)
        """
        # Conv block 1
        y = self.conv1(x)           # (B, 6, 24, 24)
        y = self.relu1(y)           # (B, 6, 24, 24)
        y = self.pool1(y)           # (B, 6, 12, 12)
        
        # Conv block 2
        y = self.conv2(y)           # (B, 16, 8, 8)
        y = self.relu2(y)           # (B, 16, 8, 8)
        y = self.pool2(y)           # (B, 16, 4, 4)
        
        # Flatten
        batch_size = y.shape[0]
        y = y.reshape(batch_size, -1)  # (B, 256)
        
        # Dense layers
        y = self.fc1(y)             # (B, 120)
        y = self.relu3(y)           # (B, 120)
        
        y = self.fc2(y)             # (B, 84)
        y = self.relu4(y)           # (B, 84)
        
        y = self.fc3(y)             # (B, num_classes)
        
        return y
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass when called as function."""
        return self.forward(x)
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
    
    def parameters(self):
        """Get all learnable parameters."""
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params
    
    def zero_grad(self):
        """Reset all gradients."""
        self.conv1.zero_grad()
        self.conv2.zero_grad()
        self.fc1.zero_grad()
        self.fc2.zero_grad()
        self.fc3.zero_grad()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LeNet5(num_classes={self.num_classes})"


# ============================================================================
# Model Statistics
# ============================================================================

def count_parameters(model: LeNet5) -> int:
    """Count total number of parameters."""
    total = 0
    for param in model.parameters():
        total += param.size
    return total


def print_model_info(model: LeNet5):
    """Print detailed model information."""
    print(f"\n{model}")
    print("=" * 60)
    print("Layer                         Parameters")
    print("-" * 60)
    
    # Conv1
    conv1_params = (5 * 5 * 1 + 1) * 6
    print(f"Conv2D (1 → 6, 5×5)           {conv1_params:>10,}")
    
    # Conv2
    conv2_params = (5 * 5 * 6 + 1) * 16
    print(f"Conv2D (6 → 16, 5×5)          {conv2_params:>10,}")
    
    # FC1
    fc1_params = (16 * 4 * 4 + 1) * 120
    print(f"Linear (256 → 120)            {fc1_params:>10,}")
    
    # FC2
    fc2_params = (120 + 1) * 84
    print(f"Linear (120 → 84)             {fc2_params:>10,}")
    
    # FC3
    fc3_params = (84 + 1) * 10
    print(f"Linear (84 → 10)              {fc3_params:>10,}")
    
    total = conv1_params + conv2_params + fc1_params + fc2_params + fc3_params
    print("-" * 60)
    print(f"Total Parameters              {total:>10,}")
    print("=" * 60)


# ============================================================================
# Tests
# ============================================================================

def test_lenet_forward():
    """Test forward pass with correct shapes."""
    print("Testing LeNet-5 Forward Pass...")
    
    model = LeNet5(num_classes=10)
    
    # MNIST input: batch=4, 1 channel, 28x28
    x = Tensor(np.random.randn(4, 1, 28, 28).astype(np.float32))
    
    output = model(x)
    
    assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"
    print(f"✓ Input shape (4, 1, 28, 28) → Output shape {output.shape}")


def test_lenet_backward():
    """Test backward pass."""
    print("\nTesting LeNet-5 Backward Pass...")
    
    model = LeNet5(num_classes=10)
    
    x = Tensor(np.random.randn(2, 1, 28, 28).astype(np.float32), requires_grad=True)
    output = model(x)
    
    # Compute loss (sum for simplicity)
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist and have correct shapes
    for i, param in enumerate(model.parameters()):
        assert param.grad is not None, f"Parameter {i} has no gradient"
        assert param.grad.shape == param.shape, \
            f"Parameter {i}: gradient shape {param.grad.shape} != {param.shape}"
    
    print(f"✓ Backward pass successful, all parameters have gradients")


def test_lenet_shapes():
    """Test intermediate shapes through network."""
    print("\nTesting LeNet-5 Intermediate Shapes...")
    
    model = LeNet5()
    x = Tensor(np.random.randn(2, 1, 28, 28).astype(np.float32))
    
    # Forward through each stage
    y = model.conv1(x)
    print(f"  After Conv1: {y.shape} (expected (2, 6, 24, 24))")
    assert y.shape == (2, 6, 24, 24)
    
    y = model.relu1(y)
    print(f"  After ReLU1: {y.shape}")
    assert y.shape == (2, 6, 24, 24)
    
    y = model.pool1(y)
    print(f"  After Pool1: {y.shape} (expected (2, 6, 12, 12))")
    assert y.shape == (2, 6, 12, 12)
    
    y = model.conv2(y)
    print(f"  After Conv2: {y.shape} (expected (2, 16, 8, 8))")
    assert y.shape == (2, 16, 8, 8)
    
    y = model.relu2(y)
    print(f"  After ReLU2: {y.shape}")
    
    y = model.pool2(y)
    print(f"  After Pool2: {y.shape} (expected (2, 16, 4, 4))")
    assert y.shape == (2, 16, 4, 4)
    
    y = y.reshape(2, -1)
    print(f"  After Flatten: {y.shape} (expected (2, 256))")
    assert y.shape == (2, 256)
    
    print(f"✓ All intermediate shapes correct")


def test_lenet_parameters():
    """Test parameter counting."""
    print("\nTesting LeNet-5 Parameters...")
    
    model = LeNet5()
    total = count_parameters(model)
    
    # Expected: (5*5*1+1)*6 + (5*5*6+1)*16 + (256+1)*120 + (120+1)*84 + (84+1)*10
    # = 156 + 2416 + 30840 + 10164 + 850 = 44426
    expected = 156 + 2416 + 30840 + 10164 + 850
    
    assert total == expected, f"Expected {expected} parameters, got {total}"
    print(f"✓ Total parameters: {total:,} (expected {expected:,})")


def test_lenet_train_eval_modes():
    """Test train/eval mode switching."""
    print("\nTesting LeNet-5 Train/Eval Modes...")
    
    model = LeNet5()
    
    assert model.training == True
    model.eval()
    assert model.training == False
    model.train()
    assert model.training == True
    
    print(f"✓ Train/eval mode switching works")


if __name__ == "__main__":
    test_lenet_forward()
    test_lenet_backward()
    test_lenet_shapes()
    test_lenet_parameters()
    test_lenet_train_eval_modes()
    
    print("\n" + "=" * 60)
    print_model_info(LeNet5())
    
    print("\n✓ All LeNet-5 tests passed!")
