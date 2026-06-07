"""Unit tests for models"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensor import Tensor
from models.lenet import LeNet5
from models.alexnet import AlexNet


def test_lenet_forward():
    """Test LeNet-5 forward pass."""
    print("Testing LeNet-5 forward...")
    
    model = LeNet5(num_classes=10)
    x = Tensor(np.random.randn(2, 1, 28, 28))
    y = model(x)
    
    assert y.shape == (2, 10), f"Expected (2, 10), got {y.shape}"
    
    print("✓ LeNet-5 forward works")


def test_lenet_backward():
    """Test LeNet-5 backward pass."""
    print("Testing LeNet-5 backward...")
    
    model = LeNet5(num_classes=10)
    x = Tensor(np.random.randn(2, 1, 28, 28), requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Check all parameters have gradients
    for param in model.parameters():
        assert param.grad is not None, f"Parameter {param} missing gradient"
    
    print("✓ LeNet-5 backward works")


def test_lenet_parameter_count():
    """Test LeNet-5 parameter count."""
    print("Testing LeNet-5 parameter count...")
    
    model = LeNet5(num_classes=10)
    total_params = sum(np.prod(p.shape) for p in model.parameters())
    
    # LeNet-5 should have ~44k parameters
    assert 40000 < total_params < 50000, f"Expected ~44k, got {total_params}"
    
    print(f"✓ LeNet-5 has {total_params:,} parameters")


def test_alexnet_forward():
    """Test AlexNet forward pass."""
    print("Testing AlexNet forward...")
    
    model = AlexNet(num_classes=10)
    x = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
    y = model(x)
    
    assert y.shape == (2, 10), f"Expected (2, 10), got {y.shape}"
    
    print("✓ AlexNet forward works")


def test_alexnet_backward():
    """Test AlexNet backward pass."""
    print("Testing AlexNet backward...")
    
    model = AlexNet(num_classes=10)
    x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32), requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Check all parameters have gradients
    for param in model.parameters():
        assert param.grad is not None, f"Parameter missing gradient"
    
    print("✓ AlexNet backward works")


def test_alexnet_parameter_count():
    """Test AlexNet parameter count."""
    print("Testing AlexNet parameter count...")
    
    model = AlexNet(num_classes=10)
    total_params = sum(np.prod(p.shape) for p in model.parameters())
    
    # AlexNet with 10 classes should have ~58-60M parameters
    assert 55000000 < total_params < 62000000, f"Expected ~58M, got {total_params}"
    
    print(f"✓ AlexNet has {total_params:,} parameters")


def test_model_train_eval():
    """Test train/eval modes."""
    print("Testing train/eval modes...")
    
    model = LeNet5(num_classes=10)
    
    # Switch to eval mode
    model.eval()
    
    # Forward pass in eval mode
    x = Tensor(np.random.randn(2, 1, 28, 28))
    y1 = model(x)
    y2 = model(x)
    
    # Should be identical in eval mode
    assert np.allclose(y1.data, y2.data)
    
    # Switch to train mode
    model.train()
    
    print("✓ Train/eval modes work")


if __name__ == "__main__":
    print("\n=== Model Tests ===\n")
    test_lenet_forward()
    test_lenet_backward()
    test_lenet_parameter_count()
    test_alexnet_forward()
    test_alexnet_backward()
    test_alexnet_parameter_count()
    test_model_train_eval()
    print("\n✓ All model tests passed!")
