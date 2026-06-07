"""
Stage 3: Linear Layer (Fully Connected)

Implements y = xW + b with proper shape handling and gradient computation.

Mathematical derivations:
- Forward: y = xW + b
- Backward dL/dX = dL/dY @ W^T
- Backward dL/dW = X^T @ dL/dY
- Backward dL/db = sum(dL/dY) over batch dimension
"""

import numpy as np
from tensor import Tensor


class Linear:
    """
    Fully connected (dense) layer.
    
    Maps input of shape (batch, in_features) to (batch, out_features).
    
    Parameters:
        weight: Shape (in_features, out_features)
        bias: Shape (out_features,)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: np.dtype = np.float32
    ):
        """
        Initialize linear layer with Xavier initialization.
        
        Xavier initialization: 
        W ~ Uniform(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
        
        This ensures similar variance across layers, enabling stable training.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias term
            dtype: Data type for parameters
        """
        # Xavier initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        
        # Weight: (in_features, out_features)
        w_data = np.random.uniform(-limit, limit, (in_features, out_features))
        self.weight = Tensor(w_data, requires_grad=True, dtype=dtype)
        
        # Bias: (out_features,)
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=dtype), requires_grad=True)
        else:
            self.bias = None
        
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = xW + b
        
        Args:
            x: Input tensor with shape (..., in_features)
            
        Returns:
            Output tensor with shape (..., out_features)
        """
        # Matrix multiplication
        # x: (..., in_features) @ W: (in_features, out_features) = (..., out_features)
        y = x @ self.weight
        
        # Add bias (broadcasts across batch dimension)
        if self.bias is not None:
            y = y + self.bias
        
        return y
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass when called as function."""
        return self.forward(x)
    
    def parameters(self):
        """Return list of learnable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
    
    def zero_grad(self):
        """Reset all gradients to None."""
        self.weight.zero_grad()
        if self.bias is not None:
            self.bias.zero_grad()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Linear({self.in_features}, {self.out_features}, bias={self.has_bias})"


# ============================================================================
# Comprehensive Test Suite
# ============================================================================

def test_linear_forward():
    """Test forward pass with known values."""
    print("Testing Linear Forward Pass...")
    
    # Create layer
    layer = Linear(in_features=3, out_features=2)
    
    # Set known weights and bias
    layer.weight.data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    layer.bias.data = np.array([0.5, -0.5], dtype=np.float32)
    
    # Input
    x = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    
    # Forward: [1, 2, 3] @ [[1, 0], [0, 1], [1, 1]] + [0.5, -0.5]
    #        = [1*1 + 2*0 + 3*1, 1*0 + 2*1 + 3*1] + [0.5, -0.5]
    #        = [4, 5] + [0.5, -0.5]
    #        = [4.5, 4.5]
    y = layer(x)
    
    expected = np.array([[4.5, 4.5]], dtype=np.float32)
    assert np.allclose(y.data, expected), f"Expected {expected}, got {y.data}"
    print(f"✓ Forward pass correct: {y.data}")


def test_linear_backward():
    """Test backward pass with gradient checking."""
    print("\nTesting Linear Backward Pass...")
    
    # Create simple layer
    layer = Linear(in_features=3, out_features=2)
    layer.weight.data = np.ones((3, 2), dtype=np.float32)
    layer.bias.data = np.zeros(2, dtype=np.float32)
    
    # Input
    x = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), requires_grad=True)
    
    # Forward
    y = layer(x)
    
    # Loss (sum for simplicity)
    loss = y.sum()
    
    # Backward
    loss.backward()
    
    # Check gradient shapes
    assert x.grad.shape == x.shape, f"Input gradient shape mismatch"
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad.shape == layer.bias.shape
    
    print(f"✓ Input gradient shape: {x.grad.shape}")
    print(f"✓ Weight gradient shape: {layer.weight.grad.shape}")
    print(f"✓ Bias gradient shape: {layer.bias.grad.shape}")
    
    # Numerical gradient check for weights
    from autograd import check_gradient
    
    def loss_fn(w):
        layer.weight.data = w.reshape(layer.weight.shape)
        x_test = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        y_test = layer(x_test)
        return float(y_test.sum().data)
    
    w_initial = layer.weight.data.copy()
    analytical_grad = layer.weight.grad
    
    check_gradient(loss_fn, w_initial.copy(), analytical_grad, verbose=True)


def test_linear_shapes():
    """Test forward and backward with various batch sizes."""
    print("\nTesting Linear Layer Shapes...")
    
    test_cases = [
        (1, 10, 5),      # (batch, in_features, out_features)
        (32, 784, 128),
        (64, 256, 64),
        (1, 1, 1),       # Edge case: single input/output
    ]
    
    for batch, in_feat, out_feat in test_cases:
        layer = Linear(in_feat, out_feat)
        x = Tensor(np.random.randn(batch, in_feat), requires_grad=True)
        y = layer(x)
        
        assert y.shape == (batch, out_feat), f"Shape mismatch for {(batch, in_feat, out_feat)}"
        
        # Backward
        loss = y.sum()
        loss.backward()
        
        assert x.grad.shape == x.shape
        assert layer.weight.grad.shape == layer.weight.shape
        
        print(f"✓ {(batch, in_feat, out_feat)}: x {x.shape} -> y {y.shape}")


def test_linear_without_bias():
    """Test linear layer without bias."""
    print("\nTesting Linear Layer Without Bias...")
    
    layer = Linear(3, 2, bias=False)
    x = Tensor(np.ones((1, 3)))
    y = layer(x)
    
    assert y.shape == (1, 2)
    assert layer.bias is None
    
    print(f"✓ Linear layer without bias works")


if __name__ == "__main__":
    test_linear_forward()
    test_linear_backward()
    test_linear_shapes()
    test_linear_without_bias()
    print("\n✓ All linear layer tests passed!")
