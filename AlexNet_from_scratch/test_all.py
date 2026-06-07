"""
Test suite for the entire AlexNet framework

Run with: python -m pytest tests/ -v
Or: python test_all.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensor import Tensor
from layers.linear import Linear
from layers.relu import ReLU
from layers.conv2d import Conv2D
from layers.maxpool import MaxPool2D
from layers.lrn import LocalResponseNormalization
from layers.dropout import Dropout
from losses.cross_entropy import CrossEntropyLoss
from optim.sgd import SGD_Momentum
from models.lenet import LeNet5
from models.alexnet import AlexNet


def test_tensor_creation():
    """Test tensor creation."""
    print("\n=== Testing Tensor Creation ===")
    
    # From list
    t1 = Tensor([1, 2, 3])
    assert t1.shape == (3,)
    print("✓ Creation from list")
    
    # From numpy
    t2 = Tensor(np.array([[1, 2], [3, 4]]))
    assert t2.shape == (2, 2)
    print("✓ Creation from numpy array")
    
    # Zeros and ones
    t3 = Tensor.zeros(3, 4)
    assert t3.shape == (3, 4) and np.all(t3.data == 0)
    print("✓ Creation with zeros")
    
    t4 = Tensor.ones(2, 3)
    assert t4.shape == (2, 3) and np.all(t4.data == 1)
    print("✓ Creation with ones")


def test_tensor_operations():
    """Test tensor arithmetic and shape operations."""
    print("\n=== Testing Tensor Operations ===")
    
    # Addition
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([1.0, 1.0, 1.0])
    c = a + b
    assert np.allclose(c.data, [2, 3, 4])
    print("✓ Addition")
    
    # Multiplication
    d = a * b
    assert np.allclose(d.data, [1, 2, 3])
    print("✓ Multiplication")
    
    # Reshape
    e = Tensor(np.arange(6))
    e_reshaped = e.reshape(2, 3)
    assert e_reshaped.shape == (2, 3)
    print("✓ Reshape")
    
    # Matrix multiplication
    m1 = Tensor(np.random.randn(3, 4))
    m2 = Tensor(np.random.randn(4, 5))
    m3 = m1 @ m2
    assert m3.shape == (3, 5)
    print("✓ Matrix multiplication")


def test_linear_layer():
    """Test fully connected layer."""
    print("\n=== Testing Linear Layer ===")
    
    layer = Linear(10, 5)
    x = Tensor(np.random.randn(2, 10))
    y = layer(x)
    
    assert y.shape == (2, 5)
    print("✓ Linear forward pass")
    
    # Backward
    loss = y.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    print("✓ Linear backward pass")


def test_relu_activation():
    """Test ReLU activation."""
    print("\n=== Testing ReLU ===")
    
    relu = ReLU()
    
    # Forward
    x = Tensor(np.array([[-1.0, 0.0, 1.0, 2.0]]))
    y = relu(x)
    expected = np.array([[0.0, 0.0, 1.0, 2.0]])
    assert np.allclose(y.data, expected)
    print("✓ ReLU forward")
    
    # Backward
    x_grad = Tensor(np.array([[-1.0, 0.0, 1.0, 2.0]]), requires_grad=True)
    y_grad = relu(x_grad)
    loss = y_grad.sum()
    loss.backward()
    
    expected_grad = np.array([[0.0, 0.0, 1.0, 1.0]])
    assert np.allclose(x_grad.grad, expected_grad)
    print("✓ ReLU backward")


def test_conv2d_layer():
    """Test 2D convolution layer."""
    print("\n=== Testing Conv2D ===")
    
    conv = Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
    
    # Forward
    x = Tensor(np.random.randn(2, 3, 28, 28))
    y = conv(x)
    
    assert y.shape == (2, 16, 28, 28), f"Got {y.shape}"
    print("✓ Conv2D forward pass")
    
    # Backward
    loss = y.sum()
    loss.backward()
    
    assert conv.weight.grad is not None
    assert conv.weight.grad.shape == conv.weight.shape
    print("✓ Conv2D backward pass")


def test_maxpool_layer():
    """Test max pooling layer."""
    print("\n=== Testing MaxPool2D ===")
    
    pool = MaxPool2D(pool_size=2, stride=2)
    
    # Forward
    x = Tensor(np.random.randn(2, 16, 28, 28))
    y = pool(x)
    
    assert y.shape == (2, 16, 14, 14)
    print("✓ MaxPool2D forward pass")
    
    # Backward
    x_grad = Tensor(np.random.randn(2, 16, 28, 28), requires_grad=True)
    y_grad = pool(x_grad)
    loss = y_grad.sum()
    loss.backward()
    
    assert x_grad.grad is not None
    print("✓ MaxPool2D backward pass")


def test_dropout_layer():
    """Test dropout layer."""
    print("\n=== Testing Dropout ===")
    
    dropout = Dropout(p=0.5)
    
    # Training mode
    dropout.train()
    x = Tensor(np.ones((100, 100)))
    y = dropout(x)
    
    zero_ratio = np.sum(y.data == 0) / y.size
    assert 0.4 < zero_ratio < 0.6, f"Zero ratio: {zero_ratio}"
    print("✓ Dropout training mode")
    
    # Eval mode
    dropout.eval()
    y_eval = dropout(x)
    assert np.allclose(y_eval.data, x.data)
    print("✓ Dropout eval mode")


def test_cross_entropy_loss():
    """Test cross-entropy loss."""
    print("\n=== Testing Cross-Entropy Loss ===")
    
    loss_fn = CrossEntropyLoss()
    
    # Simple case
    logits = Tensor(np.array([[2.0, 1.0]], dtype=np.float32))
    targets = np.array([0], dtype=np.int64)
    
    loss = loss_fn(logits, targets)
    assert loss.data.shape == ()
    assert loss.data > 0
    print("✓ Cross-entropy forward")
    
    # Backward
    loss.backward()
    assert logits.grad is not None
    print("✓ Cross-entropy backward")


def test_optimizer():
    """Test SGD optimizer."""
    print("\n=== Testing Optimizer ===")
    
    param = Tensor(np.array([1.0, 2.0, 3.0]))
    param.grad = np.array([0.1, 0.2, 0.3])
    
    optimizer = SGD_Momentum([param], learning_rate=0.01, momentum=0.9)
    
    old_value = param.data.copy()
    optimizer.step()
    
    assert not np.allclose(param.data, old_value)
    print("✓ Optimizer step")


def test_lenet_model():
    """Test LeNet-5 architecture."""
    print("\n=== Testing LeNet-5 ===")
    
    model = LeNet5(num_classes=10)
    
    # Forward
    x = Tensor(np.random.randn(2, 1, 28, 28))
    y = model(x)
    
    assert y.shape == (2, 10)
    print("✓ LeNet-5 forward pass")
    
    # Backward
    loss = y.sum()
    loss.backward()
    
    for param in model.parameters():
        assert param.grad is not None, "Parameter missing gradient"
    print("✓ LeNet-5 backward pass")


def test_alexnet_model():
    """Test AlexNet architecture."""
    print("\n=== Testing AlexNet ===")
    
    model = AlexNet(num_classes=10)
    
    # Forward
    x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
    y = model(x)
    
    assert y.shape == (1, 10)
    print("✓ AlexNet forward pass")
    
    # Backward
    loss = y.sum()
    loss.backward()
    
    for param in model.parameters():
        assert param.grad is not None, "Parameter missing gradient"
    print("✓ AlexNet backward pass")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AlexNet Framework - Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        test_tensor_creation()
        test_tensor_operations()
        test_linear_layer()
        test_relu_activation()
        test_conv2d_layer()
        test_maxpool_layer()
        test_dropout_layer()
        test_cross_entropy_loss()
        test_optimizer()
        test_lenet_model()
        test_alexnet_model()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
