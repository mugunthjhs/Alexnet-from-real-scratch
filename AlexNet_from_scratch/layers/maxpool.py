"""
MaxPool Layer

Applies max pooling for dimensionality reduction.

Forward:
Y[i,j] = max(X[i*S:i*S+P, j*S:j*S+P])

Backward:
Gradient flows only to position with maximum value (store argmax indices).

Output shape:
H_out = floor((H - P) / S) + 1
W_out = floor((W - P) / S) + 1
"""

import numpy as np
from tensor import Tensor


class MaxPool2D:
    """
    2D Max Pooling layer.
    
    Reduces spatial dimensions by taking maximum over pooling windows.
    
    Input:  (batch, channels, height, width)
    Output: (batch, channels, out_height, out_width)
    """
    
    def __init__(
        self,
        pool_size: int = 2,
        stride: int = None,
        padding: int = 0,
    ):
        """
        Initialize max pooling layer.
        
        Args:
            pool_size: Side length of square pooling window
            stride: Stride between windows (default: same as pool_size for non-overlapping)
            padding: Padding to add to input
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.padding = padding
    
    def _compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute output shape.
        
        Args:
            input_shape: (batch, channels, height, width)
            
        Returns:
            (batch, channels, out_height, out_width)
        """
        batch, channels, h_in, w_in = input_shape
        
        h_out = (h_in + 2 * self.padding - self.pool_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.pool_size) // self.stride + 1
        
        return (batch, channels, h_out, w_out)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: max pooling.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Max pooled tensor
        """
        # Pad input if needed
        if self.padding > 0:
            x_data = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                           mode='constant', constant_values=-np.inf)
        else:
            x_data = x.data
        
        batch, channels, h_padded, w_padded = x_data.shape
        out_h, out_w = self._compute_output_shape(x.shape)[2:]
        
        # Output and argmax indices
        y_data = np.zeros((batch, channels, out_h, out_w), dtype=x_data.dtype)
        argmax_indices = np.zeros((batch, channels, out_h, out_w, 2), dtype=np.int32)
        
        # Max pooling
        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        i_start = i * self.stride
                        j_start = j * self.stride
                        patch = x_data[b, c, i_start:i_start+self.pool_size, j_start:j_start+self.pool_size]
                        
                        # Find max and its location
                        max_val = np.max(patch)
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        
                        y_data[b, c, i, j] = max_val
                        argmax_indices[b, c, i, j] = [i_start + max_idx[0], j_start + max_idx[1]]
        
        out = Tensor(y_data, requires_grad=x.requires_grad, is_leaf=False)
        
        if x.requires_grad:
            # Store cache for backward
            out._pool_cache = {
                'argmax_indices': argmax_indices,
                'input_shape': x_data.shape if self.padding > 0 else x.shape,
            }
            
            def _backward(grad):
                batch, channels, h_padded, w_padded = out._pool_cache['input_shape']
                grad_x = np.zeros((batch, channels, h_padded, w_padded), dtype=grad.dtype)
                
                # Route gradient to max position
                for b in range(batch):
                    for c in range(channels):
                        for i in range(out_h):
                            for j in range(out_w):
                                i_max, j_max = argmax_indices[b, c, i, j]
                                grad_x[b, c, int(i_max), int(j_max)] = grad[b, c, i, j]
                
                # Remove padding from gradient
                if self.padding > 0:
                    grad_x = grad_x[:, :, self.padding:-self.padding, self.padding:-self.padding]
                
                x.backward(grad_x)
            
            out._backward = _backward
            out._prev.add(x)
        
        return out
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass when called as function."""
        return self.forward(x)
    
    def parameters(self):
        """MaxPool has no learnable parameters."""
        return []
    
    def zero_grad(self):
        """No-op for MaxPool."""
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MaxPool2D(pool_size={self.pool_size}, stride={self.stride}, padding={self.padding})"


# ============================================================================
# Tests
# ============================================================================

def test_maxpool_output_shape():
    """Test output shape calculation."""
    print("Testing MaxPool Output Shape...")
    
    test_cases = [
        # (pool_size, stride, padding, input_h, expected_h)
        (2, 2, 0, 4, 2),
        (2, 2, 0, 28, 14),
        (3, 2, 0, 26, 12),  # (26 - 3) / 2 + 1 = 12.5 -> 12
        (3, 1, 1, 28, 28),  # Same padding
    ]
    
    for pool, stride, pad, h_in, h_expected in test_cases:
        maxpool = MaxPool2D(pool_size=pool, stride=stride, padding=pad)
        x = Tensor(np.random.randn(1, 3, h_in, h_in))
        y = maxpool(x)
        
        assert y.shape[2] == h_expected, f"Expected height {h_expected}, got {y.shape[2]}"
        print(f"✓ Pool {pool}x{pool}, stride {stride}: {h_in}x{h_in} -> {y.shape[2]}x{y.shape[3]}")


def test_maxpool_forward():
    """Test forward pass with known values."""
    print("\nTesting MaxPool Forward Pass...")
    
    # Simple 2x2 pooling on 4x4 input
    x_data = np.array([[[[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0],
                         [9.0, 10.0, 11.0, 12.0],
                         [13.0, 14.0, 15.0, 16.0]]]], dtype=np.float32)
    
    x = Tensor(x_data)
    maxpool = MaxPool2D(pool_size=2, stride=2)
    y = maxpool(x)
    
    expected = np.array([[[[6.0, 8.0],
                          [14.0, 16.0]]]], dtype=np.float32)
    
    assert np.allclose(y.data, expected), f"Expected {expected}, got {y.data}"
    print(f"✓ MaxPool forward pass correct")


def test_maxpool_backward():
    """Test backward pass with gradient routing."""
    print("\nTesting MaxPool Backward Pass...")
    
    x = Tensor(np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32), requires_grad=True)
    maxpool = MaxPool2D(pool_size=2, stride=2)
    y = maxpool(x)
    
    loss = y.sum()
    loss.backward()
    
    # Gradient should be [0, 0, 0, 1] (only max element gets gradient)
    expected_grad = np.array([[[[0.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    assert np.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"
    print(f"✓ MaxPool backward gradient correct: {x.grad}")


def test_maxpool_batch():
    """Test pooling with batch dimension."""
    print("\nTesting MaxPool with Batch...")
    
    x = Tensor(np.random.randn(4, 16, 28, 28), requires_grad=True)
    maxpool = MaxPool2D(pool_size=2, stride=2)
    y = maxpool(x)
    
    assert y.shape == (4, 16, 14, 14)
    
    loss = y.sum()
    loss.backward()
    
    assert x.grad.shape == x.shape
    print(f"✓ MaxPool batch processing: {x.shape} -> {y.shape}")


if __name__ == "__main__":
    test_maxpool_output_shape()
    test_maxpool_forward()
    test_maxpool_backward()
    test_maxpool_batch()
    print("\n✓ All MaxPool tests passed!")
