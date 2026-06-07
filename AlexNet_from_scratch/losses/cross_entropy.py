"""
Cross-Entropy Loss

For classification with softmax outputs.

Mathematical definition:
L = -mean(log(softmax(logits)[true_class]))

Key insight: Combined softmax + cross-entropy backward pass gives:
dL/dz = (softmax(z) - one_hot(y)) / batch_size

This is remarkably simple and numerically stable!
"""

import numpy as np
from tensor import Tensor


class CrossEntropyLoss:
    """
    Cross-entropy loss for multi-class classification.
    
    Combines softmax and cross-entropy for numerical stability and efficiency.
    
    Input:
        logits: Predicted logits with shape (batch, num_classes)
        targets: True class indices with shape (batch,) or one-hot (batch, num_classes)
    
    Output:
        Scalar loss
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize loss.
        
        Args:
            reduction: 'mean' or 'sum'
        """
        self.reduction = reduction
    
    def forward(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        """
        Forward pass: compute cross-entropy loss.
        
        Args:
            logits: Predicted logits (batch, num_classes)
            targets: True labels - either class indices (batch,) or one-hot (batch, num_classes)
            
        Returns:
            Scalar loss tensor
        """
        # Ensure targets are one-hot encoded
        if targets.ndim == 1:
            batch_size = targets.shape[0]
            num_classes = logits.shape[1]
            targets_one_hot = np.zeros((batch_size, num_classes), dtype=logits.data.dtype)
            targets_one_hot[np.arange(batch_size), targets] = 1.0
        else:
            targets_one_hot = targets
            batch_size = targets.shape[0]
        
        # Softmax: subtract max for numerical stability
        logits_data = logits.data
        logits_max = np.max(logits_data, axis=1, keepdims=True)
        logits_shift = logits_data - logits_max
        exp_logits = np.exp(logits_shift)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Cross-entropy: -sum(targets * log(probs))
        eps = 1e-10  # Numerical stability
        cross_entropy_vals = -np.sum(targets_one_hot * np.log(probs + eps), axis=1)
        
        if self.reduction == 'mean':
            loss_val = np.mean(cross_entropy_vals)
        elif self.reduction == 'sum':
            loss_val = np.sum(cross_entropy_vals)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        
        out = Tensor(loss_val, requires_grad=logits.requires_grad)
        
        if logits.requires_grad:
            # Store for backward
            out._ce_cache = {
                'probs': probs,
                'targets_one_hot': targets_one_hot,
                'batch_size': batch_size,
            }
            
            def _backward(grad):
                # Gradient: (probs - targets) / batch_size (for mean reduction)
                # or (probs - targets) (for sum reduction)
                probs = out._ce_cache['probs']
                targets_one_hot = out._ce_cache['targets_one_hot']
                batch_size = out._ce_cache['batch_size']
                
                grad_logits = (probs - targets_one_hot)
                
                if self.reduction == 'mean':
                    grad_logits = grad_logits / batch_size
                
                # Multiply by upstream gradient
                grad_logits = grad_logits * grad
                
                logits.backward(grad_logits)
            
            out._backward = _backward
            out._prev.add(logits)
        
        return out
    
    def __call__(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        """Forward pass when called as function."""
        return self.forward(logits, targets)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


class SoftmaxCrossEntropy:
    """
    Alias for CrossEntropyLoss for compatibility.
    """
    def __init__(self, reduction: str = 'mean'):
        self.loss = CrossEntropyLoss(reduction=reduction)
    
    def forward(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        return self.loss.forward(logits, targets)
    
    def __call__(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        return self.loss(logits, targets)


# ============================================================================
# Tests
# ============================================================================

def test_cross_entropy_forward():
    """Test forward pass with known values."""
    print("Testing Cross-Entropy Forward Pass...")
    
    # Simple case: 2 classes, 1 sample
    logits = Tensor(np.array([[2.0, 1.0]], dtype=np.float32))
    targets = np.array([0], dtype=np.int64)  # First class
    
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    
    # Softmax: e^2 / (e^2 + e^1), e^1 / (e^2 + e^1)
    exp2 = np.exp(2.0)
    exp1 = np.exp(1.0)
    prob0 = exp2 / (exp2 + exp1)
    
    # Cross-entropy: -log(prob0)
    expected_loss = -np.log(prob0)
    
    assert np.isclose(loss.data, expected_loss), \
        f"Expected {expected_loss}, got {loss.data}"
    print(f"✓ Forward pass correct: loss = {loss.data}")


def test_cross_entropy_backward():
    """Test backward pass - gradient of softmax + CE."""
    print("\nTesting Cross-Entropy Backward Pass...")
    
    logits = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), requires_grad=True)
    targets = np.array([1], dtype=np.int64)  # Second class
    
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    loss.backward()
    
    # Gradient should be: softmax(logits) - one_hot(targets)
    logits_data = logits.data
    logits_max = np.max(logits_data)
    exp_logits = np.exp(logits_data - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    
    one_hot = np.zeros_like(logits_data)
    one_hot[0, 1] = 1.0
    
    expected_grad = (probs - one_hot) / 1  # batch_size = 1
    
    assert np.allclose(logits.grad, expected_grad), \
        f"Expected gradient {expected_grad}, got {logits.grad}"
    print(f"✓ Backward pass correct: dL/dz = softmax - one_hot")


def test_cross_entropy_batch():
    """Test with batch of samples."""
    print("\nTesting Cross-Entropy with Batch...")
    
    batch_size = 4
    num_classes = 3
    logits = Tensor(np.random.randn(batch_size, num_classes), requires_grad=True)
    targets = np.array([0, 1, 2, 0], dtype=np.int64)
    
    loss_fn = CrossEntropyLoss(reduction='mean')
    loss = loss_fn(logits, targets)
    loss.backward()
    
    assert loss.data.shape == (), "Loss should be scalar"
    assert logits.grad.shape == logits.shape
    print(f"✓ Batch processing: loss shape {loss.data.shape}, grad shape {logits.grad.shape}")


def test_cross_entropy_one_hot():
    """Test with one-hot encoded targets."""
    print("\nTesting Cross-Entropy with One-Hot Targets...")
    
    logits = Tensor(np.random.randn(2, 5), requires_grad=True)
    targets_one_hot = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, targets_one_hot)
    loss.backward()
    
    assert logits.grad.shape == logits.shape
    print(f"✓ One-hot targets work correctly")


def test_cross_entropy_perfect_prediction():
    """Test loss when prediction is perfect."""
    print("\nTesting Cross-Entropy with Perfect Prediction...")
    
    # Create logits that heavily favor the true class
    logits = Tensor(np.array([[100.0, 0.0, 0.0]], dtype=np.float32))
    targets = np.array([0], dtype=np.int64)
    
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    
    # Loss should be very close to 0
    assert loss.data < 0.01, f"Loss should be near 0, got {loss.data}"
    print(f"✓ Perfect prediction: loss ≈ 0 ({loss.data:.6f})")


def test_cross_entropy_wrong_prediction():
    """Test loss when prediction is wrong."""
    print("\nTesting Cross-Entropy with Wrong Prediction...")
    
    # Create logits that favor wrong class
    logits = Tensor(np.array([[0.0, 100.0, 0.0]], dtype=np.float32))
    targets = np.array([0], dtype=np.int64)  # First class is true
    
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    
    # Loss should be very large
    assert loss.data > 10, f"Loss should be large, got {loss.data}"
    print(f"✓ Wrong prediction: loss is large ({loss.data:.2f})")


def test_cross_entropy_sum_reduction():
    """Test sum reduction instead of mean."""
    print("\nTesting Cross-Entropy with Sum Reduction...")
    
    logits = Tensor(np.random.randn(4, 3))
    targets = np.array([0, 1, 2, 0], dtype=np.int64)
    
    loss_mean = CrossEntropyLoss(reduction='mean')(logits, targets)
    loss_sum = CrossEntropyLoss(reduction='sum')(logits, targets)
    
    # Sum should be approximately mean * batch_size
    assert np.isclose(loss_sum.data / 4, loss_mean.data, rtol=1e-5), \
        f"Sum/batch should equal mean"
    print(f"✓ Sum reduction: {loss_sum.data:.4f} ≈ {loss_mean.data:.4f} * 4")


if __name__ == "__main__":
    test_cross_entropy_forward()
    test_cross_entropy_backward()
    test_cross_entropy_batch()
    test_cross_entropy_one_hot()
    test_cross_entropy_perfect_prediction()
    test_cross_entropy_wrong_prediction()
    test_cross_entropy_sum_reduction()
    print("\n✓ All Cross-Entropy tests passed!")
