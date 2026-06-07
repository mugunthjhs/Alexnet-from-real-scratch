"""
Stage 2: Automatic Differentiation

Provides the computational graph and backpropagation machinery.
This module builds on Tensor to support complex operations with proper
gradient tracking and chain rule application.

Key concepts:
- Computational graph: DAG of operations
- Topological sort: Process operations in reverse order for backprop
- Chain rule: Multiply upstream and local gradients
"""

import numpy as np
from tensor import Tensor
from typing import List, Set


class ComputationalGraph:
    """
    Manages computational graph and backpropagation.
    
    Builds an implicit DAG as operations are performed, then performs
    reverse-mode automatic differentiation (backpropagation).
    """
    
    def __init__(self):
        """Initialize computational graph."""
        self.nodes = set()
        self.edges = dict()  # node -> list of parents
    
    def reset(self):
        """Clear graph for new computation."""
        self.nodes.clear()
        self.edges.clear()
    
    def add_node(self, tensor: Tensor):
        """Add tensor to graph."""
        self.nodes.add(id(tensor))
    
    def backward(self, loss: Tensor, retain_graph: bool = False):
        """
        Backpropagate through entire graph.
        
        Args:
            loss: Scalar tensor to backpropagate from
            retain_graph: Keep graph after backprop (for multiple backward passes)
        """
        if loss.data.size != 1:
            raise ValueError("Can only backpropagate from scalar tensors")
        
        # Initialize gradient for loss
        loss.backward(np.ones_like(loss.data))
        
        if not retain_graph:
            self.reset()


def _topological_sort(tensor: Tensor) -> List[Tensor]:
    """
    Topological sort of computational graph from a tensor.
    
    Returns nodes in reverse topological order (for backpropagation).
    
    Args:
        tensor: Starting tensor (usually loss)
        
    Returns:
        List of tensors in topological order for backward pass
    """
    visited = set()
    topo_order = []
    
    def dfs(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        for prev_node in node._prev:
            dfs(prev_node)
        topo_order.append(node)
    
    dfs(tensor)
    return topo_order


def backpropagate(loss: Tensor):
    """
    Backpropagate through computational graph from loss.
    
    This is the core automatic differentiation algorithm:
    1. Perform topological sort from loss
    2. Apply chain rule in reverse order
    
    Args:
        loss: Scalar loss tensor to backpropagate from
    """
    if loss.data.size != 1:
        raise ValueError("Can only backpropagate from scalar tensors")
    
    # Start with gradient of 1 for the loss
    loss.grad = np.ones_like(loss.data)
    
    # Get topological order for backprop
    topo_order = _topological_sort(loss)
    
    # Backpropagate through each operation
    for tensor in reversed(topo_order):
        if tensor._backward is not None:
            tensor._backward()


# ============================================================================
# Convenience Functions for Common Operations
# ============================================================================

def pad(tensor: Tensor, padding: int) -> Tensor:
    """
    Pad tensor with zeros.
    
    Args:
        tensor: Input tensor with shape (B, C, H, W) or (H, W)
        padding: Number of pixels to pad on each side
        
    Returns:
        Padded tensor
    """
    if padding == 0:
        return tensor
    
    ndim = tensor.ndim
    if ndim == 4:
        # (B, C, H, W)
        pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    elif ndim == 3:
        # (C, H, W)
        pad_width = ((0, 0), (padding, padding), (padding, padding))
    elif ndim == 2:
        # (H, W)
        pad_width = ((padding, padding), (padding, padding))
    else:
        raise ValueError(f"Unsupported tensor dimensions: {ndim}")
    
    out = Tensor(np.pad(tensor.data, pad_width, mode='constant', constant_values=0),
                 requires_grad=tensor.requires_grad, is_leaf=False)
    
    if tensor.requires_grad:
        # Gradient flows back through padding
        if ndim == 4:
            def _backward(grad):
                grad_unpadded = grad[:, :, padding:-padding, padding:-padding]
                tensor.backward(grad_unpadded)
        elif ndim == 3:
            def _backward(grad):
                grad_unpadded = grad[:, padding:-padding, padding:-padding]
                tensor.backward(grad_unpadded)
        else:  # ndim == 2
            def _backward(grad):
                grad_unpadded = grad[padding:-padding, padding:-padding]
                tensor.backward(grad_unpadded)
        
        out._backward = _backward
        out._prev.add(tensor)
    
    return out


def concatenate(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """
    Concatenate tensors along axis.
    
    Args:
        tensors: List of tensors to concatenate
        axis: Axis along which to concatenate
        
    Returns:
        Concatenated tensor
    """
    data = np.concatenate([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=requires_grad, is_leaf=False)
    
    if requires_grad:
        # Track split points for backward
        split_sizes = [t.shape[axis] for t in tensors]
        split_indices = np.cumsum(split_sizes)[:-1]
        
        def _backward(grad):
            grads = np.split(grad, split_indices, axis=axis)
            for tensor, g in zip(tensors, grads):
                if tensor.requires_grad:
                    tensor.backward(g)
        
        out._backward = _backward
        out._prev.update(tensors)
    
    return out


def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """
    Stack tensors along new axis.
    
    Args:
        tensors: List of tensors to stack
        axis: Axis along which to stack
        
    Returns:
        Stacked tensor
    """
    data = np.stack([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=requires_grad, is_leaf=False)
    
    if requires_grad:
        def _backward(grad):
            grads = np.split(grad, len(tensors), axis=axis)
            for tensor, g in zip(tensors, grads):
                if tensor.requires_grad:
                    # Remove the stacked axis
                    tensor.backward(np.squeeze(g, axis))
        
        out._backward = _backward
        out._prev.update(tensors)
    
    return out


def where(condition: np.ndarray, x: Tensor, y: Tensor) -> Tensor:
    """
    Element-wise selection from x or y based on condition.
    
    Args:
        condition: Boolean array
        x: Tensor to select from when condition is True
        y: Tensor to select from when condition is False
        
    Returns:
        Selected tensor
    """
    requires_grad = x.requires_grad or y.requires_grad
    out = Tensor(np.where(condition, x.data, y.data), requires_grad=requires_grad, is_leaf=False)
    
    if requires_grad:
        def _backward(grad):
            if x.requires_grad:
                grad_x = np.where(condition, grad, 0)
                x.backward(grad_x)
            if y.requires_grad:
                grad_y = np.where(condition, 0, grad)
                y.backward(grad_y)
        
        out._backward = _backward
        out._prev.add(x)
        out._prev.add(y)
    
    return out


def clip(tensor: Tensor, min_val: float, max_val: float) -> Tensor:
    """
    Clip tensor values to range [min_val, max_val].
    
    Args:
        tensor: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped tensor
    """
    out = Tensor(np.clip(tensor.data, min_val, max_val), requires_grad=tensor.requires_grad, is_leaf=False)
    
    if tensor.requires_grad:
        def _backward(grad):
            # Gradient is zero outside [min_val, max_val]
            mask = (tensor.data >= min_val) & (tensor.data <= max_val)
            grad_clipped = np.where(mask, grad, 0)
            tensor.backward(grad_clipped)
        
        out._backward = _backward
        out._prev.add(tensor)
    
    return out


# ============================================================================
# Loss Function Helpers (will be moved to losses/ module)
# ============================================================================

def softmax(logits: Tensor, axis: int = -1) -> Tensor:
    """
    Softmax function for converting logits to probabilities.
    
    Mathematical definition:
    σ(z)_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
    
    Numerical stability: Subtract max to prevent overflow
    
    Args:
        logits: Input tensor (typically last axis is class dimension)
        axis: Axis along which to apply softmax
        
    Returns:
        Probabilities summing to 1 along axis
    """
    # Subtract max for numerical stability
    z_max = np.max(logits.data, axis=axis, keepdims=True)
    exp_z = np.exp(logits.data - z_max)
    sum_exp = np.sum(exp_z, axis=axis, keepdims=True)
    probs = exp_z / sum_exp
    
    out = Tensor(probs, requires_grad=logits.requires_grad, is_leaf=False)
    
    if logits.requires_grad:
        def _backward(grad):
            # Jacobian of softmax is: diag(p) - p @ p^T
            # But for backprop through loss, it simplifies
            # Here we compute the full Jacobian for generality
            p = probs
            grad_logits = grad * p  # Element-wise first term
            
            # Subtract second term: -p * (grad.T @ p)
            grad_sum = np.sum(grad * p, axis=axis, keepdims=True)
            grad_logits -= p * grad_sum
            
            logits.backward(grad_logits)
        
        out._backward = _backward
        out._prev.add(logits)
    
    return out


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """
    Cross-entropy loss for classification.
    
    Mathematical definition:
    L = -mean(log(softmax(logits)[true_class]))
    
    Args:
        logits: Predicted logits with shape (B, C) where C is num_classes
        targets: True class indices with shape (B,) or one-hot matrix (B, C)
        
    Returns:
        Scalar loss tensor
    """
    # Compute softmax
    z_max = np.max(logits.data, axis=1, keepdims=True)
    exp_z = np.exp(logits.data - z_max)
    sum_exp = np.sum(exp_z, axis=1, keepdims=True)
    probs = exp_z / sum_exp
    
    # Compute loss
    if targets.ndim == 1:
        # targets are class indices
        loss_vals = -np.log(probs[np.arange(len(targets)), targets] + 1e-10)
    else:
        # targets are one-hot encoded
        loss_vals = -np.sum(targets * np.log(probs + 1e-10), axis=1)
    
    loss = np.mean(loss_vals)
    out = Tensor(loss, requires_grad=logits.requires_grad, is_leaf=False)
    
    if logits.requires_grad:
        def _backward(grad):
            # Gradient: (probs - targets) / batch_size
            batch_size = logits.data.shape[0]
            if targets.ndim == 1:
                targets_one_hot = np.zeros_like(logits.data)
                targets_one_hot[np.arange(batch_size), targets] = 1
            else:
                targets_one_hot = targets
            
            grad_logits = (probs - targets_one_hot) / batch_size * grad
            logits.backward(grad_logits)
        
        out._backward = _backward
        out._prev.add(logits)
    
    return out


# ============================================================================
# Numerical Gradient Checking
# ============================================================================

def numerical_gradient(
    function,
    x: np.ndarray,
    eps: float = 1e-5,
    use_central: bool = True
) -> np.ndarray:
    """
    Compute numerical gradient using finite differences.
    
    Central difference (more accurate):
    ∇f(x) ≈ (f(x+ε) - f(x-ε)) / (2ε)
    
    Forward difference (simpler):
    ∇f(x) ≈ (f(x+ε) - f(x)) / ε
    
    Args:
        function: Function to differentiate (takes array, returns scalar)
        x: Point at which to compute gradient
        eps: Finite difference step size
        use_central: Use central difference (more accurate, more expensive)
        
    Returns:
        Numerical gradient with same shape as x
    """
    grad = np.zeros_like(x)
    
    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += eps
        
        if use_central:
            x_minus = x.copy()
            x_minus.flat[i] -= eps
            grad.flat[i] = (function(x_plus) - function(x_minus)) / (2 * eps)
        else:
            grad.flat[i] = (function(x_plus) - function(x)) / eps
    
    return grad


def check_gradient(
    function,
    x: np.ndarray,
    analytical_grad: np.ndarray,
    eps: float = 1e-5,
    tol: float = 1e-5,
    verbose: bool = True
) -> bool:
    """
    Check gradient by comparing analytical and numerical gradients.
    
    Computes relative error:
    error = ||∇_analytical - ∇_numerical|| / (||∇_analytical|| + ||∇_numerical||)
    
    Args:
        function: Function to test
        x: Test point
        analytical_grad: Gradient from backprop
        eps: Finite difference step size
        tol: Tolerance for relative error
        verbose: Print results
        
    Returns:
        True if gradient check passes
    """
    numerical_grad = numerical_gradient(function, x, eps)
    
    # Compute relative error
    diff = np.linalg.norm(analytical_grad - numerical_grad)
    denom = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    relative_error = diff / (denom + 1e-10)
    
    passed = relative_error < tol
    
    if verbose:
        print(f"Gradient Check: {'PASSED' if passed else 'FAILED'}")
        print(f"  Relative Error: {relative_error:.2e}")
        print(f"  Tolerance: {tol:.2e}")
        if not passed:
            print(f"  Max abs diff: {np.max(np.abs(analytical_grad - numerical_grad)):.2e}")
    
    return passed
