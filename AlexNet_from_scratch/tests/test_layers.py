"""Unit tests for layers"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensor import Tensor
from layers.linear import Linear
from layers.relu import ReLU, LeakyReLU
from layers.conv2d import Conv2D
from layers.maxpool import MaxPool2D


def test_linear_forward():
    """Test linear layer forward pass."""
    print("Testing Linear forward...")
    
    layer = Linear(10, 5)
    x = Tensor(np.random.randn(2, 10))
    y = layer(x)
    
    assert y.shape == (2, 5), f"Expected (2, 5), got {y.shape}"
    
    # Verify computation: y = x @ W + b
    expected = x.data @ layer.weight.data + layer.bias.data
    assert np.allclose(y.data, expected)
    
    print("✓ Linear forward works")


def test_linear_backward():
    """Test linear layer backward pass."""
    print("Testing Linear backward...")
    
    layer = Linear(10, 5)
    x = Tensor(np.random.randn(2, 10), requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    
    print("✓ Linear backward works")


def test_relu_forward():
    """Test ReLU forward pass."""
    print("Testing ReLU forward...")
    
    relu = ReLU()
    x = Tensor(np.array([[-2, -1, 0, 1, 2]], dtype=np.float32))
    y = relu(x)
    
    expected = np.array([[0, 0, 0, 1, 2]], dtype=np.float32)
    assert np.allclose(y.data, expected)
    
    print("✓ ReLU forward works")


def test_relu_backward():
    """Test ReLU backward pass."""
    print("Testing ReLU backward...")
    
    relu = ReLU()
    x = Tensor(np.array([[-2, -1, 0, 1, 2]], dtype=np.float32), requires_grad=True)
    y = relu(x)
    loss = y.sum()
    loss.backward()
    
    expected_grad = np.array([[0, 0, 0, 1, 1]], dtype=np.float32)
    assert np.allclose(x.grad, expected_grad)
    
    print("✓ ReLU backward works")


def test_leaky_relu():
    """Test LeakyReLU."""
    print("Testing LeakyReLU...")
    
    leaky_relu = LeakyReLU(alpha=0.01)
    x = Tensor(np.array([[-2, -1, 0, 1, 2]], dtype=np.float32))
    y = leaky_relu(x)
    
    expected = np.array([[-0.02, -0.01, 0, 1, 2]], dtype=np.float32)
    assert np.allclose(y.data, expected)
    
    print("✓ LeakyReLU works")


def test_conv2d_output_shape():
    """Test Conv2D output shape calculation."""
    print("Testing Conv2D shapes...")
    
    # AlexNet layer 1: (B, 3, 224, 224) -> (B, 96, 54, 54)
    conv = Conv2D(3, 96, kernel_size=11, stride=4, padding=0)
    x = Tensor(np.random.randn(2, 3, 224, 224))
    y = conv(x)
    
    assert y.shape == (2, 96, 54, 54), f"Expected (2, 96, 54, 54), got {y.shape}"
    
    print("✓ Conv2D shapes work")


def test_conv2d_backward():
    """Test Conv2D backward pass."""
    print("Testing Conv2D backward...")
    
    conv = Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
    x = Tensor(np.random.randn(2, 3, 28, 28), requires_grad=True)
    y = conv(x)
    loss = y.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert conv.weight.grad is not None
    assert conv.weight.grad.shape == conv.weight.shape
    
    print("✓ Conv2D backward works")


def test_maxpool_forward():
    """Test MaxPool2D forward pass."""
    print("Testing MaxPool2D forward...")
    
    pool = MaxPool2D(pool_size=2, stride=2)
    x = Tensor(np.random.randn(2, 16, 28, 28))
    y = pool(x)
    
    assert y.shape == (2, 16, 14, 14)
    
    print("✓ MaxPool2D forward works")


def test_maxpool_backward():
    """Test MaxPool2D backward pass."""
    print("Testing MaxPool2D backward...")
    
    pool = MaxPool2D(pool_size=2, stride=2)
    x = Tensor(np.random.randn(2, 16, 28, 28), requires_grad=True)
    y = pool(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    print("✓ MaxPool2D backward works")


if __name__ == "__main__":
    print("\n=== Layer Tests ===\n")
    test_linear_forward()
    test_linear_backward()
    test_relu_forward()
    test_relu_backward()
    test_leaky_relu()
    test_conv2d_output_shape()
    test_conv2d_backward()
    test_maxpool_forward()
    test_maxpool_backward()
    print("\n✓ All layer tests passed!")
