"""
Stage 1: Tensor Engine

A multi-dimensional array wrapper around NumPy with shape management,
broadcasting, and support for automatic differentiation.

A Tensor is the fundamental data structure:
- Stores data as NumPy arrays
- Tracks shape and dtype
- Accumulates gradients during backpropagation
- Manages computational graph for autograd
"""

import numpy as np
from typing import Union, Tuple, Optional, List


class Tensor:
    """
    Multi-dimensional array supporting automatic differentiation.
    
    Attributes:
        data: NumPy array containing tensor values
        grad: Accumulated gradient (None if no gradient computed)
        requires_grad: Whether to track gradients
        is_leaf: Whether this is a leaf node (parameter) in computation graph
        _backward: Function to compute upstream gradients
        shape: Shape of the tensor
        dtype: Data type of elements
    """
    
    def __init__(
        self, 
        data: Union[np.ndarray, list, float, int],
        requires_grad: bool = False,
        is_leaf: bool = True,
        dtype: np.dtype = np.float32
    ):
        """
        Initialize a tensor.
        
        Args:
            data: Array-like data or scalar
            requires_grad: Whether to compute gradients
            is_leaf: Whether this is a leaf node in computational graph
            dtype: Data type (default float32)
        """
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=dtype)
        elif isinstance(data, (int, float)):
            self.data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype)
        elif isinstance(data, Tensor):
            self.data = data.data.astype(dtype)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        self.grad = None
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self._backward = lambda: None
        self._prev = set()  # For computational graph
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Tensor({self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get data type."""
        return self.data.dtype
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.data.size
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim
    
    def numpy(self) -> np.ndarray:
        """Get underlying NumPy array."""
        return self.data
    
    def zero_grad(self):
        """Reset gradient to None."""
        self.grad = None
    
    def backward(self, gradient: Optional[np.ndarray] = None):
        """
        Backpropagate through computational graph.
        
        Args:
            gradient: Upstream gradient (default ones_like for scalars)
        """
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("Gradient must be provided for non-scalar tensors")
            gradient = np.ones_like(self.data)
        
        # Accumulate gradient
        if self.grad is None:
            self.grad = gradient.copy()
        else:
            self.grad += gradient
    
    # ============================================================================
    # Shape Operations
    # ============================================================================
    
    def reshape(self, *shape: int) -> 'Tensor':
        """
        Reshape tensor to new shape (no copy).
        
        Args:
            shape: New shape
            
        Returns:
            Reshaped tensor
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            def _backward(grad):
                self.backward(grad.reshape(self.shape))
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """
        Transpose tensor.
        
        Args:
            axes: Permutation of axes (default reverses all axes)
            
        Returns:
            Transposed tensor
        """
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))
        
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            # Reverse the axis permutation
            reverse_axes = [0] * len(axes)
            for i, ax in enumerate(axes):
                reverse_axes[ax] = i
            
            def _backward(grad):
                self.backward(np.transpose(grad, reverse_axes))
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    def flatten(self) -> 'Tensor':
        """
        Flatten tensor to 1D.
        
        Returns:
            Flattened tensor
        """
        original_shape = self.shape
        out = Tensor(self.data.flatten(), requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            def _backward(grad):
                self.backward(grad.reshape(original_shape))
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    def squeeze(self, axis: Optional[int] = None) -> 'Tensor':
        """
        Remove axes of length 1.
        
        Args:
            axis: Specific axis to squeeze (default all)
            
        Returns:
            Squeezed tensor
        """
        if axis is None:
            out = Tensor(np.squeeze(self.data), requires_grad=self.requires_grad, is_leaf=False)
        else:
            out = Tensor(np.squeeze(self.data, axis), requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            original_shape = self.shape
            def _backward(grad):
                # Need to expand dimensions back
                if axis is None:
                    expanded = grad
                    for i, s in enumerate(original_shape):
                        if s == 1:
                            expanded = np.expand_dims(expanded, i)
                else:
                    expanded = np.expand_dims(grad, axis)
                self.backward(expanded)
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    def expand_dims(self, axis: int) -> 'Tensor':
        """
        Add new axis at position.
        
        Args:
            axis: Position to add axis
            
        Returns:
            Expanded tensor
        """
        out = Tensor(np.expand_dims(self.data, axis), requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            def _backward(grad):
                self.backward(np.squeeze(grad, axis))
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    # ============================================================================
    # Indexing and Slicing
    # ============================================================================
    
    def __getitem__(self, key) -> 'Tensor':
        """Index or slice tensor."""
        out = Tensor(self.data[key], requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            def _backward(grad):
                grad_full = np.zeros_like(self.data)
                grad_full[key] = grad
                self.backward(grad_full)
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    # ============================================================================
    # Arithmetic Operations
    # ============================================================================
    
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise addition."""
        if isinstance(other, Tensor):
            other_data = other.data
            requires_grad = self.requires_grad or other.requires_grad
        else:
            other_data = np.array(other, dtype=self.dtype)
            requires_grad = self.requires_grad
        
        out = Tensor(self.data + other_data, requires_grad=requires_grad, is_leaf=False)
        
        if requires_grad:
            def _backward(grad):
                # Sum over broadcast dimensions
                grad_self = grad
                grad_other = grad
                
                # Reduce gradient to match original shapes
                while grad_self.ndim > self.ndim:
                    grad_self = np.sum(grad_self, axis=0)
                for i, (s1, s2) in enumerate(zip(self.shape, grad_self.shape)):
                    if s1 == 1 and s2 > 1:
                        grad_self = np.sum(grad_self, axis=i, keepdims=True)
                
                if self.requires_grad:
                    self.backward(grad_self)
                
                if isinstance(other, Tensor) and other.requires_grad:
                    while grad_other.ndim > other.ndim:
                        grad_other = np.sum(grad_other, axis=0)
                    for i, (s1, s2) in enumerate(zip(other.shape, grad_other.shape)):
                        if s1 == 1 and s2 > 1:
                            grad_other = np.sum(grad_other, axis=i, keepdims=True)
                    other.backward(grad_other)
            
            out._backward = _backward
            out._prev.add(self)
            if isinstance(other, Tensor):
                out._prev.add(other)
        
        return out
    
    def __radd__(self, other: Union[float, np.ndarray]) -> 'Tensor':
        """Right addition."""
        return self + other
    
    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise subtraction."""
        return self + (-other if isinstance(other, Tensor) else -other)
    
    def __rsub__(self, other: Union[float, np.ndarray]) -> 'Tensor':
        """Right subtraction."""
        return other + (-self)
    
    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise multiplication."""
        if isinstance(other, Tensor):
            other_data = other.data
            requires_grad = self.requires_grad or other.requires_grad
        else:
            other_data = np.array(other, dtype=self.dtype)
            requires_grad = self.requires_grad
        
        out = Tensor(self.data * other_data, requires_grad=requires_grad, is_leaf=False)
        
        if requires_grad:
            def _backward(grad):
                grad_self = grad * other_data
                grad_other = grad * self.data
                
                # Reduce to original shapes
                while grad_self.ndim > self.ndim:
                    grad_self = np.sum(grad_self, axis=0)
                for i, (s1, s2) in enumerate(zip(self.shape, grad_self.shape)):
                    if s1 == 1 and s2 > 1:
                        grad_self = np.sum(grad_self, axis=i, keepdims=True)
                
                if self.requires_grad:
                    self.backward(grad_self)
                
                if isinstance(other, Tensor) and other.requires_grad:
                    while grad_other.ndim > other.ndim:
                        grad_other = np.sum(grad_other, axis=0)
                    for i, (s1, s2) in enumerate(zip(other.shape, grad_other.shape)):
                        if s1 == 1 and s2 > 1:
                            grad_other = np.sum(grad_other, axis=i, keepdims=True)
                    other.backward(grad_other)
            
            out._backward = _backward
            out._prev.add(self)
            if isinstance(other, Tensor):
                out._prev.add(other)
        
        return out
    
    def __rmul__(self, other: Union[float, np.ndarray]) -> 'Tensor':
        """Right multiplication."""
        return self * other
    
    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise division."""
        if isinstance(other, Tensor):
            return self * (other ** -1)
        else:
            return self * (other ** -1)
    
    def __rtruediv__(self, other: float) -> 'Tensor':
        """Right division."""
        return other * (self ** -1)
    
    def __pow__(self, power: Union[int, float]) -> 'Tensor':
        """Element-wise power."""
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            def _backward(grad):
                grad_self = grad * power * (self.data ** (power - 1))
                self.backward(grad_self)
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    def __neg__(self) -> 'Tensor':
        """Negation."""
        return self * (-1)
    
    # ============================================================================
    # Matrix Operations
    # ============================================================================
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication.
        
        For 2D: standard matrix multiplication
        For 3D+: batch matrix multiplication
        """
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, is_leaf=False)
        
        if self.requires_grad or other.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    grad_self = grad @ np.swapaxes(other.data, -2, -1)
                    self.backward(grad_self)
                if other.requires_grad:
                    grad_other = np.swapaxes(self.data, -2, -1) @ grad
                    other.backward(grad_other)
            
            out._backward = _backward
            out._prev.add(self)
            out._prev.add(other)
        
        return out
    
    # ============================================================================
    # Reduction Operations
    # ============================================================================
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Sum over axis."""
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), 
                     requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            original_shape = self.shape
            def _backward(grad):
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis)
                grad = np.broadcast_to(grad, original_shape)
                self.backward(grad)
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Mean over axis."""
        result = self.sum(axis=axis, keepdims=True)
        if axis is None:
            count = self.data.size
        else:
            count = self.data.shape[axis]
        result = result * (1.0 / count)
        
        if not keepdims and axis is not None:
            result = result.squeeze(axis)
        
        return result
    
    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Maximum over axis. Returns (values, indices) if needing backward."""
        max_val = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(max_val, requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            if axis is None:
                # Global max
                max_indices = np.unravel_index(np.argmax(self.data), self.shape)
            else:
                max_indices = np.argmax(self.data, axis=axis)
            
            original_shape = self.shape
            def _backward(grad):
                grad_full = np.zeros_like(self.data)
                if axis is None:
                    grad_full[max_indices] = grad
                else:
                    if keepdims:
                        grad_expanded = grad
                    else:
                        grad_expanded = np.expand_dims(grad, axis)
                    # This is a simplified version; full implementation needs index_put
                    np.add.at(grad_full, (slice(None),) * axis + (max_indices,) + (slice(None),) * (self.ndim - axis - 1), grad_expanded)
                self.backward(grad_full)
            
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    # ============================================================================
    # Element-wise Functions
    # ============================================================================
    
    def exp(self) -> 'Tensor':
        """Element-wise exponential."""
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            def _backward(grad):
                grad_self = grad * np.exp(self.data)
                self.backward(grad_self)
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    def log(self) -> 'Tensor':
        """Element-wise natural logarithm."""
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            def _backward(grad):
                grad_self = grad / self.data
                self.backward(grad_self)
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    def sqrt(self) -> 'Tensor':
        """Element-wise square root."""
        out = Tensor(np.sqrt(self.data), requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            def _backward(grad):
                grad_self = grad / (2 * np.sqrt(self.data))
                self.backward(grad_self)
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    def abs(self) -> 'Tensor':
        """Element-wise absolute value."""
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad, is_leaf=False)
        
        if self.requires_grad:
            def _backward(grad):
                grad_self = grad * np.sign(self.data)
                self.backward(grad_self)
            out._backward = _backward
            out._prev.add(self)
        
        return out
    
    # ============================================================================
    # Tensor Creation Helpers
    # ============================================================================
    
    @staticmethod
    def zeros(*shape: int, requires_grad: bool = False, dtype=np.float32) -> 'Tensor':
        """Create tensor of zeros."""
        return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)
    
    @staticmethod
    def ones(*shape: int, requires_grad: bool = False, dtype=np.float32) -> 'Tensor':
        """Create tensor of ones."""
        return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)
    
    @staticmethod
    def randn(*shape: int, requires_grad: bool = False, dtype=np.float32) -> 'Tensor':
        """Create tensor with random normal values."""
        return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)
    
    @staticmethod
    def rand(*shape: int, requires_grad: bool = False, dtype=np.float32) -> 'Tensor':
        """Create tensor with random uniform values in [0, 1)."""
        return Tensor(np.random.rand(*shape).astype(dtype), requires_grad=requires_grad)
    
    @staticmethod
    def arange(start, stop, step=1, requires_grad: bool = False, dtype=np.float32) -> 'Tensor':
        """Create tensor with evenly spaced values."""
        return Tensor(np.arange(start, stop, step, dtype=dtype), requires_grad=requires_grad)
    
    