"""Unit tests for tensor.py"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensor import Tensor


def test_tensor_shapes():
    """Test shape tracking."""
    print("Testing tensor shapes...")
    
    # 1D
    t1 = Tensor([1, 2, 3])
    assert t1.shape == (3,), f"Expected (3,), got {t1.shape}"
    
    # 2D
    t2 = Tensor([[1, 2], [3, 4]])
    assert t2.shape == (2, 2), f"Expected (2, 2), got {t2.shape}"
    
    # 4D (batch of images)
    t4 = Tensor(np.random.randn(2, 3, 28, 28))
    assert t4.shape == (2, 3, 28, 28)
    
    print("✓ Shape tracking works")


def test_tensor_broadcasting():
    """Test broadcasting."""
    print("Testing broadcasting...")
    
    a = Tensor([[1, 2, 3]])  # (1, 3)
    b = Tensor([[1], [2], [3]])  # (3, 1)
    
    # Should broadcast correctly
    c = a + b
    assert c.shape == (3, 3)
    
    print("✓ Broadcasting works")


def test_tensor_indexing():
    """Test indexing and slicing."""
    print("Testing indexing...")
    
    t = Tensor(np.arange(12).reshape(3, 4))
    
    # Indexing
    val = t[0, 0]
    assert val.data == 0
    
    # Slicing
    row = t[0, :]
    assert row.shape == (4,)
    
    print("✓ Indexing works")


def test_matmul():
    """Test matrix multiplication."""
    print("Testing matrix multiplication...")
    
    a = Tensor(np.random.randn(2, 3))
    b = Tensor(np.random.randn(3, 4))
    
    c = a @ b
    assert c.shape == (2, 4)
    
    # Verify against numpy
    expected = a.data @ b.data
    assert np.allclose(c.data, expected)
    
    print("✓ Matrix multiplication works")


def test_reshape():
    """Test reshape operation."""
    print("Testing reshape...")
    
    t = Tensor(np.arange(12))
    t_reshaped = t.reshape(3, 4)
    
    assert t_reshaped.shape == (3, 4)
    assert np.array_equal(t_reshaped.data, np.arange(12).reshape(3, 4))
    
    print("✓ Reshape works")


def test_flatten():
    """Test flatten operation."""
    print("Testing flatten...")
    
    t = Tensor(np.random.randn(2, 3, 4))
    t_flat = t.flatten()
    
    assert t_flat.shape == (24,)
    
    print("✓ Flatten works")


def test_transpose():
    """Test transpose operation."""
    print("Testing transpose...")
    
    t = Tensor(np.random.randn(3, 4))
    t_T = t.transpose()
    
    assert t_T.shape == (4, 3)
    assert np.array_equal(t_T.data, t.data.T)
    
    print("✓ Transpose works")


def test_sum():
    """Test sum operation."""
    print("Testing sum...")
    
    t = Tensor(np.arange(12).reshape(3, 4))
    
    # Sum all
    s = t.sum()
    assert s.data == np.arange(12).sum()
    
    # Sum along axis
    s_axis0 = t.sum(axis=0)
    assert s_axis0.shape == (4,)
    
    print("✓ Sum works")


if __name__ == "__main__":
    print("\n=== Tensor Tests ===\n")
    test_tensor_shapes()
    test_tensor_broadcasting()
    test_tensor_indexing()
    test_matmul()
    test_reshape()
    test_flatten()
    test_transpose()
    test_sum()
    print("\n✓ All tensor tests passed!")
