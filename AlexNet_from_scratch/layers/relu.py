"""
ReLU Activation Function

y = max(0, x)

Gradient:
- dy/dx = 1 if x > 0
- dy/dx = 0 if x <= 0

ReLU enables training of deep networks by:
1. Non-linearity (enables learning nonlinear functions)
2. Sparsity (kills negative values)
3. Computational efficiency (simple thresholding)
4. Better gradient flow (no saturation region like sigmoid/tanh)
"""

import numpy as np
from tensor import Tensor


class ReLU:
    """
    Rectified Linear Unit activation.
    
    y = max(0, x)
    
    Properties:
    - Unbounded above (no saturation)
    - Kills negative values (sparsity)
    - Linear in positive region (easier gradients)
    """
    
    def __init__(self):
        """Initialize ReLU (no parameters)."""
        pass
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = max(0, x)
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Activated tensor (same shape as input)
        """
        # Compute output
        y_data = np.maximum(0, x.data)
        out = Tensor(y_data, requires_grad=x.requires_grad, is_leaf=False)
        
        if x.requires_grad:
            # Store mask for backward pass
            mask = (x.data > 0).astype(x.data.dtype)
            
            def _backward(grad):
                # Gradient only flows where input was positive
                grad_x = grad * mask
                x.backward(grad_x)
            
            out._backward = _backward
            out._prev.add(x)
        
        return out
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass when called as function."""
        return self.forward(x)
    
    def parameters(self):
        """ReLU has no learnable parameters."""
        return []
    
    def zero_grad(self):
        """No-op for ReLU (no parameters)."""
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return "ReLU()"


class LeakyReLU:
    """
    Leaky ReLU activation.
    
    y = x if x > 0 else alpha * x
    
    Properties:
    - Allows small negative gradient (avoids dead neurons)
    - alpha typically 0.01 or 0.1
    - Can improve training stability
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Initialize Leaky ReLU.
        
        Args:
            alpha: Slope for negative values
        """
        self.alpha = alpha
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = max(alpha*x, x)
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        y_data = np.where(x.data > 0, x.data, self.alpha * x.data)
        out = Tensor(y_data, requires_grad=x.requires_grad, is_leaf=False)
        
        if x.requires_grad:
            def _backward(grad):
                # dy/dx = 1 if x > 0, else alpha
                grad_x = np.where(x.data > 0, grad, self.alpha * grad)
                x.backward(grad_x)
            
            out._backward = _backward
            out._prev.add(x)
        
        return out
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass when called as function."""
        return self.forward(x)
    
    def parameters(self):
        """Leaky ReLU has no learnable parameters."""
        return []
    
    def zero_grad(self):
        """No-op for LeakyReLU."""
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LeakyReLU(alpha={self.alpha})"


# ============================================================================
# Tests
# ============================================================================

def test_relu_forward():
    """Test ReLU forward pass."""
    print("Testing ReLU Forward Pass...")
    
    # Test case 1: Simple values
    x = Tensor(np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]))
    relu = ReLU()
    y = relu(x)
    
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
    assert np.allclose(y.data, expected), f"Expected {expected}, got {y.data}"
    print(f"✓ ReLU forward: {y.data}")
    
    # Test case 2: 2D input (batch)
    x2d = Tensor(np.array([[-1.0, 0.5], [0.0, -0.5], [1.0, 2.0]]))
    y2d = relu(x2d)
    expected2d = np.array([[0.0, 0.5], [0.0, 0.0], [1.0, 2.0]])
    assert np.allclose(y2d.data, expected2d)
    print(f"✓ ReLU forward (2D): shape {y2d.shape}")


def test_relu_backward():
    """Test ReLU backward pass with gradient checking."""
    print("\nTesting ReLU Backward Pass...")
    
    x = Tensor(np.array([[-1.0, 0.0, 1.0, 2.0]]), requires_grad=True)
    relu = ReLU()
    y = relu(x)
    
    # Backward
    loss = y.sum()
    loss.backward()
    
    # Gradient should be [0, 0, 1, 1] (only positive inputs get gradient)
    expected_grad = np.array([[0.0, 0.0, 1.0, 1.0]])
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print(f"✓ ReLU backward gradient: {x.grad}")


def test_relu_dead_neurons():
    """Test that ReLU kills negative values."""
    print("\nTesting ReLU Sparsity...")
    
    # Create input with mix of positive and negative values
    x = Tensor(np.random.randn(100))
    relu = ReLU()
    y = relu(x)
    
    # Count dead neurons (zero outputs)
    dead_ratio = np.sum(y.data == 0) / y.size
    print(f"✓ Dead neurons: {dead_ratio:.2%} (expected ~50%)")


def test_leaky_relu_forward():
    """Test Leaky ReLU forward pass."""
    print("\nTesting Leaky ReLU Forward Pass...")
    
    x = Tensor(np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]))
    leaky_relu = LeakyReLU(alpha=0.1)
    y = leaky_relu(x)
    
    expected = np.array([[-0.2, -0.1, 0.0, 1.0, 2.0]])
    assert np.allclose(y.data, expected), f"Expected {expected}, got {y.data}"
    print(f"✓ Leaky ReLU forward: {y.data}")


def test_leaky_relu_backward():
    """Test Leaky ReLU backward pass."""
    print("\nTesting Leaky ReLU Backward Pass...")
    
    x = Tensor(np.array([[-1.0, 0.0, 1.0]]), requires_grad=True)
    leaky_relu = LeakyReLU(alpha=0.1)
    y = leaky_relu(x)
    
    loss = y.sum()
    loss.backward()
    
    # Gradient should be [0.1, 0.1, 1.0] (alpha for negative, 1 for positive)
    expected_grad = np.array([[0.1, 0.1, 1.0]])
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print(f"✓ Leaky ReLU backward gradient: {x.grad}")


if __name__ == "__main__":
    test_relu_forward()
    test_relu_backward()
    test_relu_dead_neurons()
    test_leaky_relu_forward()
    test_leaky_relu_backward()
    print("\n✓ All ReLU tests passed!")
