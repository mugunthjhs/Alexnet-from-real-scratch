"""
Dropout Layer

Regularization technique: randomly "drop" (set to zero) fraction of neurons
during training to prevent co-adaptation.

During training:
y = x * mask / (1 - p)

During evaluation:
y = x (no dropout)

The scaling factor (1 - p) ensures same expected value during train and test.
This is "inverted dropout" - scale during training, not during test.
"""

import numpy as np
from tensor import Tensor


class Dropout:
    """
    Dropout regularization layer.
    
    During training: randomly zero out fraction p of inputs
    During testing: no dropout (pass through)
    
    Helps prevent overfitting by forcing network to learn redundant representations.
    """
    
    def __init__(self, p: float = 0.5):
        """
        Initialize dropout layer.
        
        Args:
            p: Dropout probability (fraction of neurons to drop)
        """
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        
        self.p = p
        self.training = True
    
    def forward(self, x: Tensor, training: bool = None) -> Tensor:
        """
        Forward pass: apply dropout during training, pass through during eval.
        
        Args:
            x: Input tensor
            training: Override training mode (default: use self.training)
            
        Returns:
            Tensor with dropout applied (or unchanged if not training)
        """
        if training is None:
            training = self.training
        
        if not training or self.p == 0:
            # No dropout during evaluation
            return x
        
        # Generate dropout mask: 1 with probability (1-p), 0 with probability p
        mask = np.random.binomial(1, 1 - self.p, x.shape).astype(x.data.dtype)
        
        # Apply mask and scale by 1/(1-p) to maintain expected value
        scale = 1.0 / (1.0 - self.p)
        y_data = x.data * mask * scale
        
        out = Tensor(y_data, requires_grad=x.requires_grad, is_leaf=False)
        
        if x.requires_grad:
            def _backward(grad):
                grad_x = grad * mask * scale
                x.backward(grad_x)
            
            out._backward = _backward
            out._prev.add(x)
        
        return out
    
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
        """Dropout has no learnable parameters."""
        return []
    
    def zero_grad(self):
        """No-op for dropout."""
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        mode = "train" if self.training else "eval"
        return f"Dropout(p={self.p}, mode={mode})"


# ============================================================================
# Tests
# ============================================================================

def test_dropout_training():
    """Test dropout during training."""
    print("Testing Dropout During Training...")
    
    x = Tensor(np.ones((1, 100)), requires_grad=True)
    dropout = Dropout(p=0.5)
    dropout.train()
    
    y = dropout(x)
    
    # Should have approximately 50% zeros
    zero_ratio = np.sum(y.data == 0) / y.size
    print(f"✓ Zero ratio: {zero_ratio:.2%} (expected ~50%)")
    
    # Non-zero values should be scaled
    non_zero = y.data[y.data != 0]
    assert np.all(non_zero == 2.0), f"Non-zero values should be scaled by 1/(1-p) = 2.0"
    print(f"✓ Non-zero values scaled correctly to {non_zero[0]}")


def test_dropout_eval():
    """Test no dropout during evaluation."""
    print("\nTesting Dropout During Evaluation...")
    
    x = Tensor(np.ones((1, 100)))
    dropout = Dropout(p=0.5)
    dropout.eval()
    
    y = dropout(x)
    
    # Should be identical to input
    assert np.allclose(y.data, x.data), "No dropout during evaluation"
    print(f"✓ Output identical to input during evaluation")


def test_dropout_backward():
    """Test backward pass with dropout."""
    print("\nTesting Dropout Backward Pass...")
    
    x = Tensor(np.ones((2, 50)), requires_grad=True)
    dropout = Dropout(p=0.5)
    dropout.train()
    
    y = dropout(x)
    loss = y.sum()
    loss.backward()
    
    # Gradient should be scaled version of dropout mask
    assert x.grad.shape == x.shape
    
    # Gradient should have zeros where output was zero
    grad_zero_ratio = np.sum(x.grad == 0) / x.grad.size
    print(f"✓ Gradient zero ratio: {grad_zero_ratio:.2%} (expected ~50%)")


def test_dropout_no_dropout():
    """Test with dropout probability 0 (no dropout)."""
    print("\nTesting Dropout with p=0...")
    
    x = Tensor(np.random.randn(10, 20))
    dropout = Dropout(p=0.0)
    dropout.train()
    
    y = dropout(x)
    
    # Should be identical (no dropout)
    assert np.allclose(y.data, x.data)
    print(f"✓ p=0: No dropout applied")


def test_dropout_high_p():
    """Test with high dropout probability."""
    print("\nTesting Dropout with p=0.9...")
    
    x = Tensor(np.ones((1, 1000)))
    dropout = Dropout(p=0.9)
    dropout.train()
    
    y = dropout(x)
    
    # Should have ~90% zeros
    zero_ratio = np.sum(y.data == 0) / y.size
    assert 0.85 < zero_ratio < 0.95, f"Expected ~90% zeros, got {zero_ratio:.2%}"
    
    # Non-zero values should be scaled by 1/0.1 = 10
    non_zero = y.data[y.data != 0]
    assert np.all(non_zero == 10.0)
    
    print(f"✓ p=0.9: {zero_ratio:.2%} zeros, scaled values = {non_zero[0]}")


def test_dropout_different_shapes():
    """Test dropout with various tensor shapes."""
    print("\nTesting Dropout with Different Shapes...")
    
    shapes = [(32, 128), (16, 3, 32, 32), (1, 256), (64, 10)]
    
    for shape in shapes:
        x = Tensor(np.ones(shape))
        dropout = Dropout(p=0.5)
        dropout.train()
        
        y = dropout(x)
        assert y.shape == x.shape
        print(f"✓ Shape {shape} -> {y.shape}")


if __name__ == "__main__":
    test_dropout_training()
    test_dropout_eval()
    test_dropout_backward()
    test_dropout_no_dropout()
    test_dropout_high_p()
    test_dropout_different_shapes()
    print("\n✓ All Dropout tests passed!")
