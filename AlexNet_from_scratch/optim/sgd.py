"""
Optimizers: SGD and SGD with Momentum

SGD (Stochastic Gradient Descent):
θ_{t+1} = θ_t - α * ∇L(θ_t)

SGD with Momentum:
v_t = β * v_{t-1} + ∇L(θ_t)
θ_{t+1} = θ_t - α * v_t

Momentum helps by:
1. Smoothing gradient noise
2. Accelerating convergence in consistent directions
3. Escaping local minima
"""

import numpy as np
from typing import List, Optional
from tensor import Tensor


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    Parameter update rule:
    θ_{t+1} = θ_t - learning_rate * ∇L
    """
    
    def __init__(
        self,
        params: List[Tensor],
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
    ):
        """
        Initialize SGD optimizer.
        
        Args:
            params: List of tensors to optimize
            learning_rate: Step size for gradient descent
            weight_decay: L2 regularization strength (λ)
        """
        self.params = params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def step(self):
        """
        Perform a single optimization step.
        
        Updates all parameters based on their gradients.
        """
        for param in self.params:
            if param.grad is None:
                continue
            
            # Gradient with L2 regularization
            grad = param.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            # Update parameter
            param.data -= self.learning_rate * grad
    
    def zero_grad(self):
        """Reset all gradients to None."""
        for param in self.params:
            param.zero_grad()
    
    def set_learning_rate(self, lr: float):
        """Change learning rate."""
        self.learning_rate = lr
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SGD(lr={self.learning_rate}, weight_decay={self.weight_decay})"


class SGD_Momentum:
    """
    SGD with Momentum optimizer.
    
    Maintains velocity (moving average of gradients) for smoother, faster convergence.
    
    Parameter update rule:
    v_t = β * v_{t-1} + ∇L
    θ_{t+1} = θ_t - α * v_t
    
    where β is momentum coefficient (typically 0.9).
    """
    
    def __init__(
        self,
        params: List[Tensor],
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        """
        Initialize SGD with Momentum optimizer.
        
        Args:
            params: List of tensors to optimize
            learning_rate: Step size
            momentum: Momentum coefficient (0.9 or 0.99)
            weight_decay: L2 regularization strength
            nesterov: Use Nesterov momentum variant (lookahead)
        """
        self.params = params
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Initialize velocity for each parameter
        self.velocity = {}
        for i, param in enumerate(params):
            self.velocity[i] = np.zeros_like(param.data)
    
    def step(self):
        """
        Perform a single optimization step with momentum.
        """
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            # Gradient with L2 regularization
            grad = param.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            # Update velocity
            v = self.velocity[i]
            v = self.momentum * v + grad
            self.velocity[i] = v
            
            if self.nesterov:
                # Nesterov momentum: lookahead to where momentum would take us
                update = self.momentum * v + grad
            else:
                # Standard momentum
                update = v
            
            # Update parameter
            param.data -= self.learning_rate * update
    
    def zero_grad(self):
        """Reset all gradients."""
        for param in self.params:
            param.zero_grad()
    
    def set_learning_rate(self, lr: float):
        """Change learning rate."""
        self.learning_rate = lr
    
    def __repr__(self) -> str:
        """String representation."""
        nesterov_str = ", nesterov=True" if self.nesterov else ""
        return f"SGD_Momentum(lr={self.learning_rate}, momentum={self.momentum}, weight_decay={self.weight_decay}{nesterov_str})"


# Aliases for convenience
Momentum = SGD_Momentum
Adam_like_stub = SGD_Momentum  # Placeholder - real Adam would be more complex


# ============================================================================
# Tests
# ============================================================================

def test_sgd_update():
    """Test basic SGD parameter update."""
    print("Testing SGD Parameter Update...")
    
    # Create a simple parameter
    param = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    param.grad = np.array([0.1, 0.2, 0.3])
    
    optimizer = SGD([param], learning_rate=0.01)
    
    # Expected update: param -= 0.01 * grad
    expected = np.array([1.0, 2.0, 3.0]) - 0.01 * np.array([0.1, 0.2, 0.3])
    
    optimizer.step()
    
    assert np.allclose(param.data, expected), f"Expected {expected}, got {param.data}"
    print(f"✓ SGD update correct: {param.data}")


def test_sgd_multiple_params():
    """Test SGD with multiple parameters."""
    print("\nTesting SGD with Multiple Parameters...")
    
    param1 = Tensor(np.array([1.0, 2.0]))
    param1.grad = np.array([0.1, 0.2])
    
    param2 = Tensor(np.array([3.0, 4.0]))
    param2.grad = np.array([0.3, 0.4])
    
    optimizer = SGD([param1, param2], learning_rate=0.1)
    optimizer.step()
    
    expected1 = np.array([1.0, 2.0]) - 0.1 * np.array([0.1, 0.2])
    expected2 = np.array([3.0, 4.0]) - 0.1 * np.array([0.3, 0.4])
    
    assert np.allclose(param1.data, expected1)
    assert np.allclose(param2.data, expected2)
    print(f"✓ Multiple parameters updated correctly")


def test_sgd_weight_decay():
    """Test SGD with L2 regularization."""
    print("\nTesting SGD with Weight Decay...")
    
    param = Tensor(np.array([2.0, 4.0]))
    param.grad = np.array([0.0, 0.0])  # No gradient
    
    optimizer = SGD([param], learning_rate=0.1, weight_decay=0.01)
    
    # With weight decay: grad_effective = grad + weight_decay * param
    # grad_eff = [0.0, 0.0] + 0.01 * [2.0, 4.0] = [0.02, 0.04]
    # update = param - lr * grad_eff = [2.0, 4.0] - 0.1 * [0.02, 0.04]
    expected = np.array([2.0, 4.0]) - 0.1 * (np.array([0.0, 0.0]) + 0.01 * np.array([2.0, 4.0]))
    
    optimizer.step()
    
    assert np.allclose(param.data, expected)
    print(f"✓ Weight decay applied: {param.data}")


def test_momentum_accumulation():
    """Test momentum gradient accumulation."""
    print("\nTesting Momentum Accumulation...")
    
    param = Tensor(np.array([0.0]))
    optimizer = SGD_Momentum([param], learning_rate=0.1, momentum=0.9)
    
    # Simulate constant gradient direction
    for step in range(3):
        param.grad = np.array([1.0])
        optimizer.step()
        print(f"  Step {step+1}: param = {param.data[0]:.4f}")
    
    # With momentum, updates should accelerate
    # Step 1: v = 0.9 * 0 + 1 = 1, param = 0 - 0.1 * 1 = -0.1
    # Step 2: v = 0.9 * 1 + 1 = 1.9, param = -0.1 - 0.1 * 1.9 = -0.29
    # Step 3: v = 0.9 * 1.9 + 1 = 2.71, param = -0.29 - 0.1 * 2.71 = -0.561
    
    expected = -0.561
    assert np.isclose(param.data[0], expected, atol=0.001), \
        f"Expected {expected}, got {param.data[0]}"
    print(f"✓ Momentum accumulates velocity")


def test_momentum_vs_sgd():
    """Compare momentum and plain SGD convergence."""
    print("\nTesting Momentum vs Plain SGD...")
    
    # Test function: minimize x^2
    # Gradient: 2x
    # Minimum at x=0
    
    # SGD
    param_sgd = Tensor(np.array([10.0]))
    optimizer_sgd = SGD([param_sgd], learning_rate=0.1)
    
    # Momentum
    param_mom = Tensor(np.array([10.0]))
    optimizer_mom = SGD_Momentum([param_mom], learning_rate=0.1, momentum=0.9)
    
    for _ in range(10):
        # SGD
        param_sgd.grad = 2 * param_sgd.data
        optimizer_sgd.step()
        
        # Momentum
        param_mom.grad = 2 * param_mom.data
        optimizer_mom.step()
    
    print(f"  SGD final: {param_sgd.data[0]:.6f}")
    print(f"  Momentum final: {param_mom.data[0]:.6f}")
    
    # Both should converge to 0
    assert abs(param_sgd.data[0]) < 0.1
    assert abs(param_mom.data[0]) < 0.1
    print(f"✓ Both optimizers converge")


def test_nesterov_momentum():
    """Test Nesterov momentum variant."""
    print("\nTesting Nesterov Momentum...")
    
    param = Tensor(np.array([0.0]))
    optimizer = SGD_Momentum([param], learning_rate=0.1, momentum=0.9, nesterov=True)
    
    # Simulate gradient
    param.grad = np.array([1.0])
    optimizer.step()
    
    # Nesterov should give different update than standard momentum
    assert param.data[0] < 0
    print(f"✓ Nesterov momentum: param = {param.data[0]:.4f}")


def test_learning_rate_schedule():
    """Test changing learning rate during training."""
    print("\nTesting Learning Rate Schedule...")
    
    param = Tensor(np.array([1.0]))
    optimizer = SGD([param], learning_rate=0.1)
    
    param.grad = np.array([1.0])
    
    # First update with lr=0.1
    optimizer.step()
    value_1 = param.data[0]
    
    # Change learning rate to 0.01
    optimizer.set_learning_rate(0.01)
    param.grad = np.array([1.0])
    optimizer.step()
    value_2 = param.data[0]
    
    # Difference with new lr should be smaller
    diff_1 = 1.0 - value_1  # Should be 0.1
    diff_2 = value_1 - value_2  # Should be 0.01
    
    assert np.isclose(diff_1, 0.1) and np.isclose(diff_2, 0.01)
    print(f"✓ Learning rate schedule works: {diff_1:.3f} then {diff_2:.3f}")


if __name__ == "__main__":
    test_sgd_update()
    test_sgd_multiple_params()
    test_sgd_weight_decay()
    test_momentum_accumulation()
    test_momentum_vs_sgd()
    test_nesterov_momentum()
    test_learning_rate_schedule()
    print("\n✓ All optimizer tests passed!")
