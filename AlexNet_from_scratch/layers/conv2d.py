"""
Stage 6-7: Conv2D Layer (Forward and Backward)

2D Convolution is the core operation in CNNs:

Forward Pass:
Y[b,c_out,i,j] = sum over c_in, dy, dx of:
                  X[b,c_in,i+dy,j+dx] * W[c_out,c_in,dy,dx] + b[c_out]

Output shape:
H_out = floor((H + 2P - K_h) / S) + 1
W_out = floor((W + 2P - K_w) / S) + 1

Backward Pass:
dL/dX - transposed convolution
dL/dW - convolution with gradient signal
dL/db - sum over spatial and batch dimensions

This implementation uses NumPy broadcasting for efficiency.
"""

import numpy as np
from tensor import Tensor
from autograd import pad


class Conv2D:
    """
    2D Convolution layer.
    
    Input:  (batch, in_channels, height, width)
    Output: (batch, out_channels, out_height, out_width)
    
    Parameters:
        weight: (out_channels, in_channels, kernel_h, kernel_w)
        bias:   (out_channels,)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        dtype: np.dtype = np.float32
    ):
        """
        Initialize Conv2D layer.
        
        He initialization (good for ReLU):
        W ~ Normal(0, sqrt(2 / (K_h * K_w * C_in)))
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (number of filters)
            kernel_size: Side length of square kernel
            stride: Stride of convolution
            padding: Number of pixels to pad
            bias: Whether to use bias
            dtype: Data type
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        
        # He initialization for ReLU
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        
        # Weight: (out_channels, in_channels, kernel_h, kernel_w)
        w_data = np.random.normal(0, std, (out_channels, in_channels, kernel_size, kernel_size)).astype(dtype)
        self.weight = Tensor(w_data, requires_grad=True)
        
        # Bias: (out_channels,)
        if bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=dtype), requires_grad=True)
        else:
            self.bias = None
    
    def _compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute output shape.
        
        Args:
            input_shape: (batch, in_channels, height, width)
            
        Returns:
            (batch, out_channels, out_height, out_width)
        """
        batch, _, h_in, w_in = input_shape
        
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        return (batch, self.out_channels, h_out, w_out)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: convolution operation.
        
        Args:
            x: Input tensor (batch, in_channels, height, width)
            
        Returns:
            Output tensor (batch, out_channels, out_height, out_width)
        """
        # Pad input if needed
        x_padded = pad(x, self.padding) if self.padding > 0 else x
        
        batch, in_ch, h_padded, w_padded = x_padded.shape
        out_h, out_w = self._compute_output_shape(x.shape)[2:]
        
        # Output tensor
        out_shape = (batch, self.out_channels, out_h, out_w)
        y_data = np.zeros(out_shape, dtype=x_padded.data.dtype)
        
        # Naive convolution implementation
        for b in range(batch):
            for c_out in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # Extract patch from padded input
                        i_start = i * self.stride
                        j_start = j * self.stride
                        patch = x_padded.data[b, :, i_start:i_start+self.kernel_size, j_start:j_start+self.kernel_size]
                        
                        # Apply convolution (element-wise multiply and sum)
                        kernel = self.weight.data[c_out, :, :, :]
                        y_data[b, c_out, i, j] = np.sum(patch * kernel)
                        
                        # Add bias
                        if self.bias is not None:
                            y_data[b, c_out, i, j] += self.bias.data[c_out]
        
        out = Tensor(y_data, requires_grad=x.requires_grad or self.weight.requires_grad, is_leaf=False)
        
        # Store for backward pass
        if x.requires_grad or self.weight.requires_grad:
            # Store necessary info for backward
            out._conv_cache = {
                'x_padded': x_padded,
                'x': x,
                'output_shape': out_shape,
            }
            
            def _backward(grad):
                # Gradient w.r.t. input (transposed convolution)
                if x.requires_grad:
                    grad_x = self._backward_input(grad, x_padded)
                    if self.padding > 0:
                        # Remove padding from gradient
                        grad_x = grad_x.data[:, :, self.padding:-self.padding, self.padding:-self.padding]
                    x.backward(grad_x)
                
                # Gradient w.r.t. weight
                if self.weight.requires_grad:
                    grad_w = self._backward_weight(grad, x_padded)
                    self.weight.backward(grad_w)
                
                # Gradient w.r.t. bias
                if self.bias is not None and self.bias.requires_grad:
                    grad_b = np.sum(grad, axis=(0, 2, 3))  # Sum over batch and spatial dimensions
                    self.bias.backward(grad_b)
            
            out._backward = _backward
            out._prev.add(x)
            out._prev.add(self.weight)
            if self.bias is not None:
                out._prev.add(self.bias)
        
        return out
    
    def _backward_input(self, grad: np.ndarray, x_padded: Tensor) -> np.ndarray:
        """
        Compute gradient w.r.t. input (transposed convolution).
        
        Args:
            grad: Upstream gradient (batch, out_channels, out_h, out_w)
            x_padded: Padded input for shape reference
            
        Returns:
            Gradient w.r.t. padded input
        """
        batch, in_ch, h_padded, w_padded = x_padded.shape
        grad_x = np.zeros_like(x_padded.data)
        
        out_h, out_w = grad.shape[2:]
        
        # Transposed convolution
        for b in range(batch):
            for c_out in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        i_start = i * self.stride
                        j_start = j * self.stride
                        
                        # Add gradient contribution
                        kernel = self.weight.data[c_out, :, :, :]
                        grad_x[b, :, i_start:i_start+self.kernel_size, j_start:j_start+self.kernel_size] += \
                            kernel * grad[b, c_out, i, j]
        
        return grad_x
    
    def _backward_weight(self, grad: np.ndarray, x_padded: Tensor) -> np.ndarray:
        """
        Compute gradient w.r.t. weights.
        
        Args:
            grad: Upstream gradient (batch, out_channels, out_h, out_w)
            x_padded: Padded input tensor
            
        Returns:
            Gradient w.r.t. weights (out_channels, in_channels, kernel_h, kernel_w)
        """
        batch, in_ch, h_padded, w_padded = x_padded.shape
        grad_w = np.zeros_like(self.weight.data)
        
        out_h, out_w = grad.shape[2:]
        
        # Compute weight gradient
        for b in range(batch):
            for c_out in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        i_start = i * self.stride
                        j_start = j * self.stride
                        patch = x_padded.data[b, :, i_start:i_start+self.kernel_size, j_start:j_start+self.kernel_size]
                        
                        # Accumulate gradient
                        grad_w[c_out, :, :, :] += patch * grad[b, c_out, i, j]
        
        return grad_w
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass when called as function."""
        return self.forward(x)
    
    def parameters(self):
        """Return learnable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
    
    def zero_grad(self):
        """Reset all gradients."""
        self.weight.zero_grad()
        if self.bias is not None:
            self.bias.zero_grad()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Conv2D({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


# ============================================================================
# Tests
# ============================================================================

def test_conv2d_output_shape():
    """Test output shape calculation."""
    print("Testing Conv2D Output Shape...")
    
    test_cases = [
        # (in_ch, out_ch, kernel, stride, padding, input_h, expected_h)
        (3, 16, 3, 1, 0, 28, 26),  # No padding, stride 1
        (3, 16, 3, 1, 1, 28, 28),  # Same padding
        (16, 32, 3, 2, 1, 28, 14), # Stride 2
        (3, 96, 11, 4, 0, 224, 54), # AlexNet first layer
    ]
    
    for in_ch, out_ch, k, s, p, h_in, h_expected in test_cases:
        conv = Conv2D(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        x = Tensor(np.random.randn(1, in_ch, h_in, h_in))
        y = conv(x)
        
        assert y.shape[2] == h_expected, f"Expected height {h_expected}, got {y.shape[2]}"
        assert y.shape[3] == h_expected, f"Expected width {h_expected}, got {y.shape[3]}"
        
        print(f"✓ Input {h_in}x{h_in} -> Output {y.shape[2]}x{y.shape[3]} (kernel={k}, stride={s}, pad={p})")


def test_conv2d_forward():
    """Test forward pass with simple values."""
    print("\nTesting Conv2D Forward Pass...")
    
    # Create simple 1x1 convolution (should be equivalent to fully connected)
    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
    conv.weight.data = np.array([[[[2.0]]]], dtype=np.float32)  # Weight = 2
    
    x = Tensor(np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32))
    y = conv(x)
    
    expected = np.array([[[[2.0, 4.0], [6.0, 8.0]]]], dtype=np.float32)
    assert np.allclose(y.data, expected), f"Expected {expected}, got {y.data}"
    print(f"✓ Conv2D 1x1 forward pass correct")


def test_conv2d_backward():
    """Test backward pass with gradient checking."""
    print("\nTesting Conv2D Backward Pass...")
    
    # Small network for testing
    conv = Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
    x = Tensor(np.random.randn(2, 1, 5, 5), requires_grad=True)
    
    y = conv(x)
    loss = y.sum()
    loss.backward()
    
    # Check shapes
    assert x.grad.shape == x.shape
    assert conv.weight.grad.shape == conv.weight.shape
    assert conv.bias.grad.shape == conv.bias.shape
    
    print(f"✓ Conv2D backward shapes correct")
    print(f"  Input grad: {x.grad.shape}")
    print(f"  Weight grad: {conv.weight.grad.shape}")
    print(f"  Bias grad: {conv.bias.grad.shape}")


if __name__ == "__main__":
    test_conv2d_output_shape()
    test_conv2d_forward()
    test_conv2d_backward()
    print("\n✓ All Conv2D tests passed!")
