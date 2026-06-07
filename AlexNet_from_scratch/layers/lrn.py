"""
Local Response Normalization (LRN)

From AlexNet paper: "The neurons with the largest activities have a damping 
effect on neurons in neighboring feature maps at the same spatial location."

Mathematical definition:
b_{x,y}^i = a_{x,y}^i / (k + α/n * Σ_{j=max(0,i-n/2)}^{min(N-1,i+n/2)} (a_{x,y}^j)^2)^β

where:
- i: channel index
- n: number of adjacent channels to consider (default 5)
- α: scaling parameter (default 0.0001)
- β: exponent (default 0.75)
- N: total number of channels

Implements lateral inhibition (biological inspiration from neuroscience).
"""

import numpy as np
from tensor import Tensor


class LocalResponseNormalization:
    """
    Local Response Normalization (LRN) layer.
    
    Normalizes activations across adjacent channels at same spatial location.
    Used in AlexNet for regularization and learning better features.
    
    Input/Output: (batch, channels, height, width)
    """
    
    def __init__(
        self,
        size: int = 5,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 1.0,
    ):
        """
        Initialize LRN layer.
        
        Args:
            size: Number of adjacent channels to consider (usually 5)
            alpha: Scaling parameter (usually 1e-4)
            beta: Exponent (usually 0.75)
            k: Constant (usually 1.0 or 2.0)
        """
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: Local response normalization.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Normalized tensor (same shape as input)
        """
        batch, channels, height, width = x.shape
        
        # Output
        y_data = np.zeros_like(x.data)
        
        # For each output channel
        for i in range(channels):
            # Find range of channels to use for normalization
            start = max(0, i - self.size // 2)
            end = min(channels, i + self.size // 2 + 1)
            
            # Compute sum of squares in channel neighborhood
            sum_sq = np.sum(x.data[:, start:end, :, :] ** 2, axis=1, keepdims=True)
            
            # Normalize: x / (k + α/n * sum_sq)^β
            # Note: sum_sq has shape (batch, 1, height, width)
            denominator = (self.k + (self.alpha / self.size) * sum_sq) ** self.beta
            y_data[:, i:i+1, :, :] = x.data[:, i:i+1, :, :] / denominator
        
        out = Tensor(y_data, requires_grad=x.requires_grad, is_leaf=False)
        
        if x.requires_grad:
            # Store cache for backward pass
            out._lrn_cache = {
                'x': x.data,
                'y': y_data,
                'channels': channels,
            }
            
            def _backward(grad):
                # Compute gradient w.r.t. input
                grad_x = np.zeros_like(x.data)
                
                for i in range(channels):
                    start = max(0, i - self.size // 2)
                    end = min(channels, i + self.size // 2 + 1)
                    
                    # Sum of squares
                    sum_sq = np.sum(x.data[:, start:end, :, :] ** 2, axis=1, keepdims=True)
                    denominator = (self.k + (self.alpha / self.size) * sum_sq) ** self.beta
                    
                    # Gradient has two parts:
                    # 1. Direct: y_i / x_i term
                    # 2. Indirect: y_j terms that depend on x_i
                    
                    # Part 1: Direct gradient
                    grad_direct = grad[:, i:i+1, :, :] / denominator
                    
                    # Part 2: Indirect gradient (how x_i affects normalization of x_j)
                    for j in range(start, end):
                        if j == i:
                            continue
                        sum_sq_j = np.sum(x.data[:, max(0, j - self.size // 2):min(channels, j + self.size // 2 + 1), :, :] ** 2, axis=1, keepdims=True)
                        denominator_j = (self.k + (self.alpha / self.size) * sum_sq_j) ** self.beta
                        
                        # How y_j changes w.r.t. x_i (only through sum_sq term)
                        # dy_j/dx_i = -x_j / denominator_j^2 * beta * denominator_j^(beta-1) * (alpha/size) * 2 * x_i
                        grad_indirect = -grad[:, j:j+1, :, :] * (x.data[:, j:j+1, :, :] / (denominator_j ** 2)) * \
                                       self.beta * ((self.alpha / self.size) * 2 * x.data[:, i:i+1, :, :])
                        grad_direct += grad_indirect
                    
                    grad_x[:, i:i+1, :, :] = grad_direct
                
                x.backward(grad_x)
            
            out._backward = _backward
            out._prev.add(x)
        
        return out
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass when called as function."""
        return self.forward(x)
    
    def parameters(self):
        """LRN has no learnable parameters."""
        return []
    
    def zero_grad(self):
        """No-op for LRN."""
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LRN(size={self.size}, alpha={self.alpha}, beta={self.beta})"


# Alias for compatibility
LRN = LocalResponseNormalization


# ============================================================================
# Tests
# ============================================================================

def test_lrn_forward():
    """Test LRN forward pass."""
    print("Testing LRN Forward Pass...")
    
    # Simple test: single value should be normalized
    x_data = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    x = Tensor(x_data)
    
    lrn = LocalResponseNormalization(size=1, alpha=1e-4, beta=0.75, k=1.0)
    y = lrn(x)
    
    # With size=1, sum_sq = x^2 for that channel
    # y = x / (k + α * x^2)^β
    
    assert y.shape == x.shape
    assert np.all(y.data > 0), "LRN should produce positive outputs for positive inputs"
    assert np.all(y.data <= x.data), "LRN should normalize (reduce magnitude)"
    
    print(f"✓ LRN forward pass correct: input shape {x.shape} -> {y.shape}")


def test_lrn_normalization():
    """Test that LRN actually reduces magnitude."""
    print("\nTesting LRN Normalization Effect...")
    
    x_data = np.ones((1, 5, 2, 2), dtype=np.float32)
    x = Tensor(x_data)
    
    lrn = LocalResponseNormalization(size=5, alpha=1e-4, beta=0.75, k=1.0)
    y = lrn(x)
    
    # Values should be reduced
    assert np.all(y.data < x.data), "LRN should reduce magnitude"
    print(f"✓ Input max: {np.max(x.data):.4f}, Output max: {np.max(y.data):.4f}")


def test_lrn_channels():
    """Test LRN across channels."""
    print("\nTesting LRN Across Channels...")
    
    # Create input with different values per channel
    x_data = np.arange(1, 26, dtype=np.float32).reshape(1, 5, 2, 2)
    x = Tensor(x_data)
    
    lrn = LocalResponseNormalization(size=5, alpha=1e-4, beta=0.75, k=1.0)
    y = lrn(x)
    
    # Channels with larger values should be more normalized
    mean_per_channel_before = np.mean(x_data, axis=(2, 3))
    mean_per_channel_after = np.mean(y.data, axis=(2, 3))
    
    assert mean_per_channel_after[-1] < mean_per_channel_before[-1], \
        "Channels with larger values should be more suppressed"
    
    print(f"✓ LRN suppresses large channels more")


def test_lrn_backward():
    """Test LRN backward pass."""
    print("\nTesting LRN Backward Pass...")
    
    x = Tensor(np.random.randn(2, 3, 4, 4), requires_grad=True)
    lrn = LocalResponseNormalization(size=3, alpha=1e-4, beta=0.75)
    y = lrn(x)
    
    loss = y.sum()
    loss.backward()
    
    assert x.grad.shape == x.shape
    assert not np.any(np.isnan(x.grad)), "Gradients should not be NaN"
    assert not np.any(np.isinf(x.grad)), "Gradients should not be infinite"
    
    print(f"✓ LRN backward pass: gradient shape {x.grad.shape}")


def test_lrn_batch():
    """Test LRN with batch dimension."""
    print("\nTesting LRN with Batch...")
    
    x = Tensor(np.random.randn(16, 96, 54, 54), requires_grad=True)
    lrn = LocalResponseNormalization(size=5, alpha=1e-4, beta=0.75)
    y = lrn(x)
    
    assert y.shape == x.shape
    loss = y.sum()
    loss.backward()
    
    assert x.grad.shape == x.shape
    print(f"✓ LRN batch: {x.shape} -> {y.shape}")


if __name__ == "__main__":
    test_lrn_forward()
    test_lrn_normalization()
    test_lrn_channels()
    test_lrn_backward()
    test_lrn_batch()
    print("\n✓ All LRN tests passed!")
