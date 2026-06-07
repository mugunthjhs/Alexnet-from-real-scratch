# AlexNet from Scratch: A Complete Deep Learning Framework

## Table of Contents
1. [Project Overview](#project-overview)
2. [Complete Roadmap](#complete-roadmap)
3. [Stage 1: Tensor Engine](#stage-1-tensor-engine)
4. [Stage 2: Automatic Differentiation](#stage-2-automatic-differentiation)
5. [Stage 3: Linear Layers](#stage-3-linear-layers)
6. [Stage 4: Softmax and Cross-Entropy](#stage-4-softmax-and-cross-entropy)
7. [Stage 5: Optimizers](#stage-5-optimizers)
8. [Stage 6: Conv2D Forward](#stage-6-conv2d-forward)
9. [Stage 7: Conv2D Backward](#stage-7-conv2d-backward)
10. [Stage 8: Pooling and LRN](#stage-8-pooling-and-lrn)
11. [Stage 9: LeNet Architecture](#stage-9-lenet-architecture)
12. [Stage 10: AlexNet Architecture](#stage-10-alexnet-architecture)
13. [Validation and Testing Strategy](#validation-and-testing-strategy)
14. [Common Mistakes and Debugging](#common-mistakes-and-debugging)

---

## Project Overview

This project builds AlexNet (Krizhevsky et al., 2012) completely from scratch using only Python and NumPy. We evolve from Micrograd (scalar-based automatic differentiation) into a tensor-based deep learning framework capable of training AlexNet on ImageNette.

### Why Build from Scratch?

1. **Deep Understanding**: Understanding every operation, forward and backward
2. **Educational Value**: See exactly how backpropagation flows through convolutions
3. **No Black Boxes**: Full control over every computation
4. **Performance**: Direct NumPy operations without framework overhead

### Key Principles

- **Mathematical Rigor**: Every operation is mathematically derived
- **Complete Code**: No pseudocode, every file is fully functional
- **Numerical Validation**: Every operation has gradient checks
- **Educational Comments**: Code explains what and why

---

## Project Folder Structure

```
AlexNet_from_scratch/
│
├── README.md                      # Main documentation (start here)
├── learning.md                    # This file - Complete mathematical guide
│
├── tensor.py                      # Stage 1: Tensor engine (400 LOC)
│   └── Core: shape management, broadcasting, operations, autodiff
│
├── autograd.py                    # Stage 2: Automatic differentiation (600 LOC)
│   └── Core: computational graphs, backpropagation, gradient checking
│
├── layers/                        # Neural network layers (2,000+ LOC)
│   ├── __init__.py               # Package exports
│   ├── linear.py                 # Stage 3: Fully connected (250 LOC)
│   │   └── Forward: y = xW + b | Backward: proper gradient computation
│   ├── relu.py                   # Activation functions (200 LOC)
│   │   └── ReLU, LeakyReLU with mask-based gradients
│   ├── conv2d.py                 # Stages 6-7: 2D Convolution (450 LOC)
│   │   └── Forward: naive O(K²HW) | Backward: transposed conv + im2col-like
│   ├── maxpool.py                # Stage 8: Max pooling (250 LOC)
│   │   └── Forward: max operation | Backward: gradient routing via argmax
│   ├── lrn.py                    # Stage 8: Local Response Norm (300 LOC)
│   │   └── Channel-wise normalization with lateral inhibition
│   └── dropout.py                # Regularization (200 LOC)
│       └── Inverted dropout with train/eval modes
│
├── losses/                        # Loss functions (300+ LOC)
│   ├── __init__.py               # Package exports
│   └── cross_entropy.py          # Stage 4: Cross-entropy loss (250 LOC)
│       └── Softmax + cross-entropy with numerical stability
│
├── optim/                         # Optimizers (400+ LOC)
│   ├── __init__.py               # Package exports
│   └── sgd.py                    # Stage 5: SGD & Momentum (350 LOC)
│       └── SGD, SGD-M, learning rate scheduling, weight decay
│
├── models/                        # Model architectures (800+ LOC)
│   ├── __init__.py               # Package exports
│   ├── lenet.py                  # Stage 9: LeNet-5 (350 LOC)
│   │   └── Input: 28×28×1 | Conv(6)→Pool→Conv(16)→Pool→FC(120)→FC(84)→FC(10)
│   │       Parameters: ~44K | Expected: 99% on MNIST
│   └── alexnet.py                # Stage 10: AlexNet (450 LOC)
│       └── Input: 224×224×3 | Conv(96→256→384→384→256)→FC(4096)→FC(4096)→FC(N)
│           Parameters: ~60M | Expected: 85% on ImageNette | With ReLU, LRN, Dropout
│
├── datasets/                      # Data loading (530+ LOC)
│   ├── __init__.py               # Package exports
│   ├── mnist.py                  # MNIST loader (250 LOC)
│   │   └── 60K train, 10K test | 28×28 grayscale | Auto-download + batch loading
│   └── imagenette.py             # ImageNette loader (280 LOC)
│       └── 9K train, 3.6K val | 224×224 RGB | 10-class ImageNet subset
│
├── tests/                         # Testing (1,000+ LOC)
│   ├── __init__.py               # Package exports
│   ├── test_tensor.py            # Tensor tests (200 LOC)
│   ├── test_layers.py            # Layer tests (250 LOC)
│   └── test_models.py            # Model tests (200 LOC)
│
├── test_all.py                    # Integration test suite (300 LOC)
│   └── Comprehensive tests for all components
│
├── train_lenet.py                # Stage 9: LeNet training (350 LOC)
│   └── Complete training pipeline with validation, checkpointing, metrics
│       Command: python train_lenet.py
│       Expected: 99% on MNIST in 20 epochs (~5-10 min)
│
├── train_alexnet.py              # Stage 10: AlexNet training (450 LOC)
│   └── Complete training with argparse, LR scheduling, metrics
│       Command: python train_alexnet.py --data-path ./imagenette2 --epochs 40
│       Expected: 85% top-1 on ImageNette (~4-8 hrs per epoch)
│
├── utils/                         # Utilities (reserved for future)
│   └── __init__.py               # Package exports
│
└── __init__.py                    # Main package initialization

Total: 30+ files | 8,500+ LOC | 100+ tests
```

### File Organization Philosophy

1. **Core Foundation**
   - `tensor.py`: Everything starts here
   - `autograd.py`: Builds on tensor
   
2. **Modular Layers**
   - Each layer in separate file for clarity
   - All follow same interface: `forward()`, `backward()`
   
3. **Grouped Components**
   - `losses/`: Loss functions (currently just cross-entropy, but extensible)
   - `optim/`: Optimizers (SGD variants, extensible to Adam, etc.)
   - `models/`: Complete architectures that use layers
   - `datasets/`: Data loaders (currently MNIST and ImageNette)
   
4. **Training & Testing**
   - Separate training scripts for each model
   - Comprehensive test suite covering all components
   
5. **Documentation**
   - `README.md`: Quick reference and overview
   - `learning.md`: This file - Complete mathematical guide with derivations

### Import Hierarchy

```
User Code
    ↓
models/ (high-level)
    ├── layers/ (components)
    │   ├── tensor.py (base)
    │   └── autograd.py (differentiation)
    ├── losses/ (objectives)
    └── optim/ (parameter updates)
    ↓
datasets/ (data)
    └── Tensor (data structure)
```

### Stage-to-File Mapping

| Stage | File | Purpose | LOC |
|-------|------|---------|-----|
| Stage 0 | learning.md | Educational guide | 1,200 |
| Stage 1 | tensor.py | Tensor engine | 400 |
| Stage 2 | autograd.py | Auto-differentiation | 600 |
| Stage 3 | layers/linear.py | Fully connected | 250 |
| Stage 4 | losses/cross_entropy.py | Loss function | 250 |
| Stage 5 | optim/sgd.py | Optimizers | 350 |
| Stage 6 | layers/conv2d.py | Conv forward | 450 |
| Stage 7 | layers/conv2d.py | Conv backward | (same file) |
| Stage 8 | layers/maxpool.py, lrn.py | Pooling + LRN | 550 |
| Stage 9 | models/lenet.py, train_lenet.py | LeNet + training | 700 |
| Stage 10 | models/alexnet.py, train_alexnet.py | AlexNet + training | 900 |

---

## Complete Roadmap

### Progression
```
Micrograd (scalar) 
    → Tensor Engine (multidimensional arrays)
    → Automatic Differentiation (computational graphs)
    → Linear Layers (fully connected)
    → Softmax + Cross-Entropy (classification)
    → Optimizers (SGD, Momentum)
    → Convolution (2D spatial operations)
    → Pooling (dimensionality reduction)
    → LeNet (first CNN architecture)
    → AlexNet (deeper, with regularization)
```

### Component Dependency Graph
```
tensor.py (base)
    ↓
autograd.py (builds on tensor)
    ↓
layers/ (uses autograd)
    ├── linear.py
    ├── conv2d.py
    ├── maxpool.py
    ├── relu.py
    ├── dropout.py
    └── lrn.py
    ↓
losses/ (uses layers/autograd)
    └── cross_entropy.py
    ↓
optim/ (uses autograd)
    ├── sgd.py
    └── momentum.py
    ↓
models/ (uses layers + losses)
    ├── lenet.py
    └── alexnet.py
    ↓
train.py (uses everything)
```

---

## Stage 1: Tensor Engine

### Why Do We Need a Tensor Engine?

Micrograd worked with scalars (0D). Neural networks need multi-dimensional arrays:
- Images: (height, width, channels) = 3D
- Batches: (batch_size, height, width, channels) = 4D
- Weights: (out_features, in_features) = 2D

A Tensor Engine provides:
1. **Shape Management**: Track dimensions through computations
2. **Broadcasting**: Align shapes for element-wise operations
3. **View Operations**: Reshape without copying (efficiency)
4. **NumPy Integration**: Use NumPy for fast computation

### Mathematical Foundation

#### Tensor Representation

A tensor is a multi-dimensional array. For a 3D tensor (batch, height, width):

$$\mathbf{X} \in \mathbb{R}^{B \times H \times W}$$

Operations must respect shape constraints:
- Element-wise: Shapes must match (or broadcast)
- Linear (matmul): Last dim of A must equal first dim of B
- Convolution: Special sliding window computation

#### Broadcasting Rule

NumPy broadcasting aligns shapes from the right:
```
Shape 1: (5, 1, 3)
Shape 2: (   4, 3)  <- implicit leading 1
Result:  (5, 4, 3)
```

Each dimension is compatible if:
1. Equal, OR
2. One of them is 1

#### Shape Calculations

**Matrix Multiplication**:
- Input: $(M \times K)$ @ $(K \times N)$
- Output: $(M \times N)$

**Convolution**:
- Input: $(B \times C_{in} \times H \times W)$
- Kernel: $(C_{out} \times C_{in} \times K_h \times K_w)$
- Output: $(B \times C_{out} \times H_{out} \times W_{out})$

where: $H_{out} = \lfloor (H + 2P - K_h) / S \rfloor + 1$

### Key Operations

1. **Creation**: Create tensors with proper shapes
2. **Reshaping**: Change shape without copying data
3. **Transpose**: Swap dimensions
4. **Indexing**: Extract subsets
5. **Concatenation**: Join along axis

### Implementation Plan

The Tensor class wraps NumPy arrays and adds:
- Shape validation
- Operation tracking (for autograd)
- Gradient storage
- Device management (CPU only for now)

---

## Stage 2: Automatic Differentiation

### Why Autograd?

We need to compute gradients for millions of parameters. Manual backpropagation is:
1. **Error-prone**: Easy to make mistakes in complex operations
2. **Tedious**: Must manually derive every operation
3. **Inflexible**: Adding new operations requires new backward code

Automatic differentiation (AD) computes gradients exactly to machine precision by:
1. Building a computational graph
2. Applying chain rule automatically
3. Supporting arbitrary combinations of operations

### Computational Graph

For $y = f(x)$, we build a DAG:

```
        x
        |
    [Linear layer]
        |
        h
        |
    [ReLU]
        |
        z
        |
    [Loss]
        |
        L (scalar)
```

During backward pass, we traverse from output to inputs using chain rule.

### Chain Rule Derivation

For composite functions:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

For function $y = f(g(h(x)))$:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

### Gradient Computation

Each operation must implement:

1. **Forward Pass**: $y = f(x)$
2. **Backward Pass**: $\frac{\partial L}{\partial x} = \text{upstream\_grad} \times \frac{\partial f}{\partial x}$

Example for $y = 2x$:
- Forward: $y = 2x$
- Backward: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \times 2$

### Implementation Strategy

Each Tensor has:
- **data**: The actual NumPy array
- **grad**: Accumulated gradients (initially None)
- **_backward**: Function to compute upstream gradients
- **requires_grad**: Flag indicating if we need gradients

Computation graph is built implicitly as we perform operations.

### Common Operations and Their Gradients

#### Addition: $y = x + w$
- Forward: $y = x + w$
- Backward: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}$, $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y}$

#### Multiplication (Element-wise): $y = x \odot w$
- Forward: $y = x \cdot w$
- Backward: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot w$, $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \odot x$

#### Matrix Multiplication: $y = x @ w$
- Forward: $y = x @ w$ (standard matmul)
- Backward: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} @ w^T$, $\frac{\partial L}{\partial w} = x^T @ \frac{\partial L}{\partial y}$

#### Summation: $y = \sum x$
- Forward: $y = \text{sum}(x)$
- Backward: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \times \mathbf{1}$ (broadcast upstream grad)

---

## Stage 3: Linear Layers

### Fully Connected Layer

The fundamental building block of neural networks.

#### Mathematical Formulation

$$y = xW + b$$

where:
- $x \in \mathbb{R}^{B \times D_{in}}$: Input (batch_size, input_features)
- $W \in \mathbb{R}^{D_{in} \times D_{out}}$: Weights
- $b \in \mathbb{R}^{D_{out}}$: Bias
- $y \in \mathbb{R}^{B \times D_{out}}$: Output

**Computation**:
$$y_i = \sum_{j=1}^{D_{in}} x_{ij} W_{jk} + b_k$$

#### Forward Pass Derivation

```python
# Batched matrix multiplication
z = x @ W  # (B, D_in) @ (D_in, D_out) = (B, D_out)
y = z + b  # Broadcast b across batch
```

#### Backward Pass Derivations

**Gradient w.r.t. Input** $(∂L/∂x)$:

Using chain rule: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} @ W^T$

Proof:
$$L = f(y) = f(xW + b)$$
$$\frac{\partial L}{\partial x_{ik}} = \sum_j \frac{\partial L}{\partial y_{ij}} \frac{\partial y_{ij}}{\partial x_{ik}}$$
$$= \sum_j \frac{\partial L}{\partial y_{ij}} W_{kj}$$
$$= \left(\frac{\partial L}{\partial y} @ W^T\right)_{ik}$$

**Gradient w.r.t. Weight** $(∂L/∂W)$:

Using chain rule: $\frac{\partial L}{\partial W} = x^T @ \frac{\partial L}{\partial y}$

Proof:
$$\frac{\partial L}{\partial W_{kj}} = \sum_i \frac{\partial L}{\partial y_{ij}} \frac{\partial y_{ij}}{\partial W_{kj}}$$
$$= \sum_i \frac{\partial L}{\partial y_{ij}} x_{ik}$$
$$= \left(x^T @ \frac{\partial L}{\partial y}\right)_{kj}$$

**Gradient w.r.t. Bias** $(∂L/∂b)$:

Since $\frac{\partial y_{ij}}{\partial b_j} = 1$:
$$\frac{\partial L}{\partial b} = \sum_i \frac{\partial L}{\partial y}_{ij} \text{ (sum over batch)}$$

```python
grad_b = np.sum(dL_dy, axis=0)  # Sum over batch dimension
```

#### Typical Shapes

For a layer with input_features=784, output_features=128, batch_size=32:
- Input x: (32, 784)
- Weight W: (784, 128)
- Bias b: (128,)
- Output y: (32, 128)
- Grad dL/dy: (32, 128) incoming
- Grad dL/dx: (32, 784) to propagate back
- Grad dL/dW: (784, 128) for updating weights
- Grad dL/db: (128,) for updating bias

#### Implementation Checklist

- [ ] Forward pass with batched matmul
- [ ] Bias addition with broadcasting
- [ ] Weight gradient: x.T @ dL/dy
- [ ] Input gradient: dL/dy @ W.T
- [ ] Bias gradient: sum over batch
- [ ] Numerical gradient check

---

## Stage 4: Softmax and Cross-Entropy

### Softmax: From Logits to Probabilities

Softmax converts raw network outputs (logits) into probability distributions.

#### Mathematical Definition

For logits $z \in \mathbb{R}^{D}$:

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{D} e^{z_j}}$$

Properties:
- Output: $\sigma(z)_i \in [0, 1]$
- Sum: $\sum_i \sigma(z)_i = 1$ (probability distribution)
- Differentiable: Can compute gradients

#### Numerical Stability

Computing $e^{z_i}$ directly can overflow/underflow. Solution: subtract max before exponentiation.

$$\sigma(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{D} e^{z_j - \max(z)}}$$

Since $e^{a - b} = e^a / e^b$, subtracting a constant from all elements doesn't change the result, but prevents overflow.

#### Forward Pass

```
Input: z shape (B, D)
Compute: z_max = max(z) along D dimension, shape (B,)
Compute: z_shift = z - z_max[:, None]  # Broadcast
Compute: exp_z = exp(z_shift)
Compute: sum_exp = sum(exp_z) along D dimension
Compute: probs = exp_z / sum_exp[:, None]
Output: probs shape (B, D)
```

### Cross-Entropy Loss

Measures difference between predicted and true distributions.

#### Mathematical Definition

For predicted probabilities $\hat{p} \in \mathbb{R}^{D}$ and true label $y \in \{1, ..., D\}$ (one-hot encoded as $p$):

$$L = -\sum_{i=1}^{D} p_i \log(\hat{p}_i)$$

For one-hot label with true class $c$:
$$L = -\log(\hat{p}_c)$$

#### Why Negative Log Likelihood?

- Maximum likelihood estimation: Maximize $P(\text{true class})$
- Equivalent to: Minimize $-\log(P(\text{true class}))$
- Interpretation: Penalizes wrong predictions more when confident in wrong answer

#### Gradient Derivation

**dL/d(logits)** - The key insight: Softmax + Cross-Entropy has remarkably simple gradient!

$$\frac{\partial L}{\partial z} = \hat{p} - p$$

where $\hat{p}$ is predicted probabilities and $p$ is true (one-hot) label.

Derivation:
$$L = -\log(\hat{p}_c) = -\log\left(\frac{e^{z_c}}{\sum_j e^{z_j}}\right)$$
$$= -z_c + \log\left(\sum_j e^{z_j}\right)$$

Taking derivative w.r.t. $z_i$:
$$\frac{\partial L}{\partial z_i} = \begin{cases}
-1 + \frac{e^{z_i}}{\sum_j e^{z_j}} = -1 + \hat{p}_i & \text{if } i = c \\
\frac{e^{z_i}}{\sum_j e^{z_j}} = \hat{p}_i & \text{if } i \neq c
\end{cases}$$

This can be written compactly as:
$$\frac{\partial L}{\partial z} = \hat{p} - \text{one\_hot}(c)$$

**For batched data**:
$$\frac{\partial L}{\partial Z} = \frac{1}{B}(\hat{P} - \mathbb{1}_c) \quad \text{where } \mathbb{1}_c \text{ is one-hot matrix}$$

#### Implementation Checkpoints

1. **Softmax Forward**: Shift by max, exp, normalize
2. **Softmax Backward**: Jacobian matrix computation
3. **Cross-Entropy Forward**: -log(p[true_class])
4. **Cross-Entropy Backward**: pred_probs - one_hot
5. **Combined Loss + dL/dz**: Should be ~softmax output minus truth

---

## Stage 5: Optimizers

### Stochastic Gradient Descent (SGD)

#### Parameter Update Rule

$$\theta_{t+1} = \theta_t - \eta \frac{\partial L}{\partial \theta_t}$$

where:
- $\theta$: Parameters (weights, biases)
- $\eta$: Learning rate
- $\frac{\partial L}{\partial \theta}$: Gradient of loss w.r.t. parameters

#### Why It Works

1. Gradient points in direction of steepest increase
2. Negative gradient points toward decrease
3. Following negative gradient reduces loss

#### Implementation

```python
for param in params:
    if param.grad is not None:
        param.data -= learning_rate * param.grad
        param.grad = 0  # Reset for next iteration
```

### Momentum SGD

Plain SGD oscillates heavily. Momentum adds inertia.

#### Update Rule

$$v_t = \beta v_{t-1} + \frac{\partial L}{\partial \theta_t}$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

where $\beta \in [0.9, 0.99]$ (typically 0.9).

#### Intuition

- Accumulate gradient direction over time
- Smooth out noise in mini-batch gradients
- Move faster in consistent directions

#### Shape Consistency

Velocity must match parameter shape for accumulation.

---

## Stage 6: Conv2D Forward Pass

### Convolution: The Core of Modern Computer Vision

#### Why Convolution?

1. **Local Connectivity**: Each neuron sees only local patch
2. **Weight Sharing**: Same filter applied across image
3. **Translation Invariance**: Detects same features anywhere
4. **Efficiency**: Vastly fewer parameters than fully connected

#### Mathematical Definition

**Discrete 2D Convolution**:
$$Y[i,j] = \sum_{dy=0}^{K_h-1} \sum_{dx=0}^{K_w-1} X[i+dy, j+dx] \cdot W[dy, dx] + b$$

For batches and multiple channels:
$$Y[b,c_{out},i,j] = \sum_{c_{in}=0}^{C_{in}-1} \sum_{dy=0}^{K_h-1} \sum_{dx=0}^{K_w-1} X[b,c_{in},i+dy,j+dx] \cdot W[c_{out},c_{in},dy,dx] + b[c_{out}]$$

#### Shape Calculations

**Output spatial dimensions**:
$$H_{out} = \lfloor (H_{in} + 2P - K_h) / S \rfloor + 1$$
$$W_{out} = \lfloor (W_{in} + 2P - K_w) / S \rfloor + 1$$

where:
- $P$: Padding (same on all sides)
- $K_h, K_w$: Kernel height and width
- $S$: Stride

**Full output shape**: $(B, C_{out}, H_{out}, W_{out})$

#### Kernel Interpretation

Each kernel $W[c_{out}, c_{in}, :, :]$ is a feature detector:
- Detects edges, textures, patterns
- Different kernels learn different patterns
- Earlier layers: low-level (edges)
- Later layers: high-level (objects)

#### Common Architectures

**No Padding, Stride 1** (loses spatial dimensions):
- Input: 28x28, Kernel: 5x5
- Output: (28-5)+1 = 24x24

**Same Padding, Stride 1** (preserves spatial dimensions):
- Padding P = (K-1)/2
- For 5x5 kernel: P = 2
- Output: Same spatial size

**Striding** (downsampling):
- Stride S = 2
- Output dimensions halved (approximately)

#### Implementation Strategy

**Naive Approach** (slow but clear):
```
for each output position (i, j):
    for each input channel:
        for each kernel row dy:
            for each kernel col dx:
                accumulate X[i+dy, j+dx] * W[dy, dx]
```

**Optimized Approach** (using im2col):
1. Unfold input into columns
2. Reshape kernels into rows
3. Single matrix multiplication
4. Reshape output back

For now, we implement naive approach with optimization via NumPy broadcasting.

#### Padding Implementation

Pad input with zeros before convolution:
```
if padding > 0:
    X_padded = pad(X, padding)
else:
    X_padded = X
```

---

## Stage 7: Conv2D Backward Pass

### Gradient Computation for Convolution

This is the most complex operation to backprop through.

#### Gradient w.r.t. Input (dL/dX)

**Forward pass**: $Y = \text{Conv}(X, W) + b$

**Backward pass**: $\frac{\partial L}{\partial X}$ is a transposed convolution!

$$\frac{\partial L}{\partial X[i, j]} = \sum_{dy,dx} \frac{\partial L}{\partial Y[i-dy, j-dx]} W[dy, dx]$$

This is **transposed convolution** (deconvolution):
- Input: dL/dY with shape (B, C_out, H_out, W_out)
- Kernel: W with shape (C_out, C_in, K_h, K_w)
- Output: dL/dX with shape (B, C_in, H_in, W_in)

#### Gradient w.r.t. Weights (dL/dW)

$$\frac{\partial L}{\partial W[dy, dx]} = \sum_{i,j} X[i+dy, j+dx] \frac{\partial L}{\partial Y[i, j]}$$

Implementation using convolution with flipped filters:
1. For each kernel position (dy, dx)
2. Multiply corresponding input patch with upstream gradient
3. Sum the products

Can be computed as convolution of X with dL/dY (with appropriate flipping).

#### Gradient w.r.t. Bias (dL/dB)

$$\frac{\partial L}{\partial b[c_{out}]} = \sum_{b,i,j} \frac{\partial L}{\partial Y[b, c_{out}, i, j]}$$

Sum upstream gradients over all spatial positions and batch.

#### Numerical Complexity

- Forward: $O(B \cdot C_{out} \cdot H_{out} \cdot W_{out} \cdot C_{in} \cdot K_h \cdot K_w)$
- Backward (same complexity)

For AlexNet: Forward ~700M operations, Backward ~1.4B (includes all 3 gradients).

---

## Stage 8: Pooling and Local Response Normalization

### Max Pooling

#### Definition

Partition feature maps into non-overlapping regions, output maximum:

$$Y[i,j] = \max_{dy,dx \in \text{pool}} X[iS+dy, jS+dx]$$

where:
- Pool size: typically 2x2 or 3x3
- Stride: often equals pool size (no overlap)

#### Why Max Pooling?

1. **Translation Invariance**: Small shifts don't affect max
2. **Dimensionality Reduction**: Reduce spatial dimensions
3. **Noise Robustness**: Ignores non-maximum activations
4. **Feature Concentration**: Focus on strongest responses

#### Forward Pass

Shape transformation:
- Input: $(B, C, H, W)$
- Pool: $P \times P$ with stride $S$
- Output: $(B, C, H/S, W/S)$

#### Backward Pass

Key insight: Gradient flows only to position that had maximum value.

For forward: $y = \max(x_0, x_1, ..., x_n)$

Backward:
$$\frac{\partial L}{\partial x_i} = \begin{cases} \frac{\partial L}{\partial y} & \text{if } x_i = \max(...) \\ 0 & \text{otherwise} \end{cases}$$

**Implementation**: Store argmax indices during forward pass, use to route gradients back.

```python
# Forward: store argmax locations
max_indices = argmax(input_patches)

# Backward: gradient flows only to argmax position
for each output position j:
    gradient[argmax_indices[j]] += upstream_gradient[j]
```

### Local Response Normalization (LRN)

**From AlexNet Paper**: "The neurons with the largest activities have a damping effect on neurons in neighboring feature maps at the same spatial location."

#### Mathematical Definition

$$Y[i,j,k] = X[i,j,k] / \left(1 + \frac{\alpha}{n} \sum_{m=\max(0,k-\frac{n}{2})}^{\min(K-1,k+\frac{n}{2})} X[i,j,m]^2\right)^\beta$$

where:
- $k$: Channel index
- $n$: Number of adjacent channels (5 in AlexNet)
- $\alpha$: Scaling parameter (0.0001)
- $\beta$: Exponent (0.75)

#### Intuition

Normalize across channels (same spatial location):
1. Square activations in local channel neighborhood
2. Multiply by $\alpha/n$ and add 1
3. Raise to power $\beta$
4. Divide original activation

**Effect**: Normalizes across channels, implements lateral inhibition (biological inspiration).

#### Forward Pass

```python
for each position (i,j):
    for each channel k:
        # Sum squares in channel neighborhood
        local_sum = sum(X[i,j,m]^2) for m in [k-n//2, k+n//2]
        # Normalize
        Y[i,j,k] = X[i,j,k] / (1 + alpha/n * local_sum)^beta
```

#### Backward Pass

Using chain rule:
$$\frac{\partial L}{\partial X} = \text{derivative of normalization w.r.t. input}$$

For each position:
$$\frac{\partial L}{\partial X[i,j,k]} = \frac{\partial L}{\partial Y[i,j,k]} \frac{\partial Y[i,j,k]}{\partial X[i,j,k]} + \sum_m \frac{\partial L}{\partial Y[i,j,m]} \frac{\partial Y[i,j,m]}{\partial X[i,j,k]}$$

The second term accounts for how X[i,j,k] appears in normalization of other channels.

---

## Stage 9: LeNet Architecture

### LeNet-5 (LeCun et al., 1998)

The first successful CNN, trained on handwritten digit recognition (MNIST).

#### Architecture

```
Input (28x28x1)
    ↓
Conv2D (6 filters, 5x5) + ReLU
    ↓ [24x24x6]
MaxPool (2x2, stride 2)
    ↓ [12x12x6]
Conv2D (16 filters, 5x5) + ReLU
    ↓ [8x8x16]
MaxPool (2x2, stride 2)
    ↓ [4x4x16] = 256 features
Flatten
    ↓
Linear (256 → 120) + ReLU
    ↓
Linear (120 → 84) + ReLU
    ↓
Linear (84 → 10) + Softmax
    ↓
Output (10 classes)
```

#### Shape Calculations

**Conv1**: Input 28x28x1, Kernel 5x5, 6 filters, no padding, stride 1
$$H_{out} = \lfloor (28 + 0 - 5) / 1 \rfloor + 1 = 24$$
Output: 24x24x6

**Pool1**: Input 24x24x6, Pool 2x2, stride 2
$$H_{out} = 24 / 2 = 12$$
Output: 12x12x6

**Conv2**: Input 12x12x6, Kernel 5x5, 16 filters, no padding, stride 1
$$H_{out} = \lfloor (12 + 0 - 5) / 1 \rfloor + 1 = 8$$
Output: 8x8x16

**Pool2**: Input 8x8x16, Pool 2x2, stride 2
$$H_{out} = 8 / 2 = 4$$
Output: 4x4x16 = 256 neurons

#### Parameter Count

- Conv1: (5×5×1+1) × 6 = 156
- Conv2: (5×5×6+1) × 16 = 2,416
- FC1: (256+1) × 120 = 30,840
- FC2: (120+1) × 84 = 10,164
- FC3: (84+1) × 10 = 850
- **Total**: ~44K parameters

#### Training Details

- Optimizer: SGD with momentum
- Learning Rate: 0.01 (scheduled to decrease)
- Batch Size: 32
- Epochs: 20
- Loss: Cross-entropy

---

## Stage 10: AlexNet Architecture

### AlexNet (Krizhevsky et al., 2012)

First "deep" CNN that won ImageNet by huge margin (85% top-5 vs 74% previous best).

#### Key Innovations

1. **Depth**: 8 layers (5 conv, 3 FC)
2. **ReLU**: Activation function enabling deeper networks
3. **Dropout**: Regularization to prevent overfitting
4. **LRN**: Local response normalization (lateral inhibition)
5. **GPU Training**: Trained on GPUs (NVIDIA), split across 2 GPUs
6. **Data Augmentation**: Random crops and flips
7. **Large Filters**: Early layers use 11x11, 5x5 kernels

#### Full Architecture

```
Input: 224x224x3
    ↓
Conv2D (96 filters, 11x11, stride 4) + ReLU + LRN
    ↓ [(224-11)/4 + 1 = 54] = 54x54x96
MaxPool (3x3, stride 2) + LRN
    ↓ [(54-3)/2 + 1 = 26] = 26x26x96
Dropout (p=0.5)
    ↓
Conv2D (256 filters, 5x5, stride 1, pad 2) + ReLU + LRN
    ↓ [(26+4-5)/1 + 1 = 26] = 26x26x256
MaxPool (3x3, stride 2)
    ↓ [(26-3)/2 + 1 = 12] = 12x12x256
Dropout (p=0.5)
    ↓
Conv2D (384 filters, 3x3, stride 1, pad 1) + ReLU
    ↓ [(12+2-3)/1 + 1 = 12] = 12x12x384
Conv2D (384 filters, 3x3, stride 1, pad 1) + ReLU
    ↓ [(12+2-3)/1 + 1 = 12] = 12x12x384
Conv2D (256 filters, 3x3, stride 1, pad 1) + ReLU
    ↓ [(12+2-3)/1 + 1 = 12] = 12x12x256
MaxPool (3x3, stride 2)
    ↓ [(12-3)/2 + 1 = 5] = 5x5x256 = 6400 features
Dropout (p=0.5)
    ↓
Flatten → 6400
    ↓
FC (4096 units) + ReLU + Dropout (p=0.5)
    ↓
FC (4096 units) + ReLU + Dropout (p=0.5)
    ↓
FC (1000 units) + Softmax
    ↓
Output: 1000 classes (ImageNet)
```

#### Parameter Count

- Conv1: (11×11×3+1) × 96 = 34,944
- Conv2: (5×5×96+1) × 256 = 614,656
- Conv3: (3×3×384+1) × 384 = 1,327,488
- Conv4: (3×3×384+1) × 384 = 1,327,488
- Conv5: (3×3×256+1) × 256 = 590,080
- FC1: (6400+1) × 4096 = 26,214,656
- FC2: (4096+1) × 4096 = 16,781,312
- FC3: (4096+1) × 1000 = 4,097,000
- **Total**: ~60M parameters

#### Key Architectural Details

**"Overlapping Pooling"**:
- Pool size: 3×3
- Stride: 2 (not 3)
- Creates overlap, improves generalization

**Dropout** (p=0.5):
- Applied after Conv1, Conv2, Pool2
- Applied after FC1, FC2
- Not applied to Conv3, Conv4, Conv5
- Reduces overfitting (model trains longer)

**Local Response Normalization (LRN)**:
- Applied after Conv1, Conv2
- Implements lateral inhibition
- Reduces top-1 error by ~1.4%

**Modifications for ImageNette** (10 classes instead of 1000):
- Change final FC layer to 10 units
- Rest of architecture unchanged

---

## Validation and Testing Strategy

### Gradient Checking

Essential for verifying correct backpropagation implementation.

#### Numerical Gradient

Using central differences:
$$\frac{\partial f}{\partial x}_{\text{num}} \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

where $\epsilon \sim 10^{-5}$.

#### Implementation

```python
def numerical_gradient(function, x, epsilon=1e-5):
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += epsilon
        x_minus = x.copy()
        x_minus.flat[i] -= epsilon
        grad.flat[i] = (function(x_plus) - function(x_minus)) / (2 * epsilon)
    return grad
```

#### Gradient Check Procedure

1. Initialize random parameters
2. Compute analytical gradient via backprop
3. Compute numerical gradient via finite differences
4. Compare: relative error should be < 1e-5

```
relative_error = ||grad_analytical - grad_numerical|| / (||grad_analytical|| + ||grad_numerical||)
```

If error > 1e-5: bug in backprop!

### Shape Checking

Every operation must have correct shapes.

#### Conv2D Shapes
- Input: (B, C_in, H, W)
- Kernel: (C_out, C_in, K_h, K_w)
- Output: (B, C_out, H_out, W_out)

#### Gradient Shapes (should match forward shapes)
- dL/dX: Same as X
- dL/dW: Same as W
- dL/dB: Same as B

### Activation Distribution Checking

After initialization and each training step:
- Check activation means and stds
- Dead ReLU check: How many ReLUs output 0?
- Gradient flow check: Do gradients diminish in early layers?

### Accuracy Metrics

**For Classification**:
```
Accuracy = # Correct / # Total
Top-5 Error = # Not in top 5 / # Total
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

### Learning Curves

Plot during training:
- Training loss (should decrease)
- Validation loss (should decrease, then plateau)
- Training accuracy
- Validation accuracy

Watch for:
- **Overfitting**: Training loss << validation loss
- **Underfitting**: Both losses remain high
- **Divergence**: Loss becomes NaN (learning rate too high)

### Sanity Checks

Before serious training:

1. **Overfit a single batch**: Loss should reach ~0 in few iterations
2. **Loss with random labels**: Should decrease to log(num_classes)
3. **No regularization**: Loss should drop faster than with L2
4. **Learning rate sensitivity**: Large change in learning rate should dramatically affect loss

---

## Common Mistakes and Debugging

### Mistake 1: Transpose Errors in Backprop

**Symptom**: Shape mismatch errors or very strange gradients.

**Solution**: Carefully check dimensions in every backward equation.
- Matrix multiplication backward: $(dL/dy @ W^T)$ has correct shape
- Be careful with axes in summation

### Mistake 2: Gradient Accumulation vs. Reset

**Symptom**: Gradients grow unboundedly, training becomes unstable.

**Solution**: 
```python
# Reset gradients before backward pass
optimizer.zero_grad()

# Accumulate gradients
loss.backward()

# Update parameters
optimizer.step()
```

### Mistake 3: Uninitialized Gradients

**Symptom**: NaN or undefined behavior.

**Solution**: Gradients should be None initially, created during backward pass.

### Mistake 4: Incorrect Broadcasting in Bias Addition

**Symptom**: Shape errors or silent broadcasting bugs.

**Correct**:
```python
y = x @ W  # (B, D_out)
y += b     # b shape (D_out,) broadcasts to (B, D_out)
```

### Mistake 5: Wrong Padding Implementation

**Symptom**: Output spatial dimensions are wrong.

**Formula**: P = (K - 1) / 2 for "same" padding.

For K=3: P=1, For K=5: P=2, For K=11: P=5

### Mistake 6: Forgetting to Average Batch Gradient

**Symptom**: Gradients are sum over batch instead of mean.

**Solution**: 
```python
# For loss that doesn't automatically reduce
grad_batch = compute_gradient(batch)
grad_mean = grad_batch / batch_size
```

### Mistake 7: MaxPool Backward Routing

**Symptom**: Gradient magnitudes are wrong.

**Solution**: Store argmax indices in forward pass, route to correct locations in backward.

### Mistake 8: ReLU Killing Gradients

**Symptom**: Loss stops decreasing, many dead ReLUs.

**Solution**: 
- Check activation statistics
- ReLU should pass ~50% of neurons in early training
- Use Leaky ReLU if severe

### Mistake 9: Conv2D Stride Implementation

**Symptom**: Output spatial dimensions calculated incorrectly.

**Formula**: $H_{out} = \lfloor (H + 2P - K) / S \rfloor + 1$

Test with concrete examples: Input 5, Kernel 3, Stride 2, Padding 1
$$H_{out} = \lfloor (5 + 2 - 3) / 2 \rfloor + 1 = \lfloor 2 / 2 \rfloor + 1 = 1 + 1 = 2$$

### Mistake 10: Data Normalization

**Symptom**: Training is very slow or loss oscillates wildly.

**Solution**: Normalize data: $(X - \mu) / \sigma$

For images: Subtract ImageNet mean, divide by std per channel.

### Debugging Workflow

1. **Check shapes** at every operation
2. **Gradient check** critical operations
3. **Visualize activations** (histogram, max/min)
4. **Overfit a small batch** to verify code correctness
5. **Check learning curves** for divergence or stalling
6. **Validate on test set** to detect overfitting

---

## Milestones and Expected Outputs

### Stage 1 (Tensor Engine)
✓ Create tensors, reshape, transpose
✓ Broadcasting works
✓ Slicing and indexing work
Expected: Basic shape manipulation complete

### Stage 2 (Autograd)
✓ Forward pass builds computational graph
✓ Backward pass computes gradients
✓ Gradient shapes match forward shapes
Expected: Gradient check passes for all operations

### Stage 3 (Linear Layer)
✓ Forward pass: y = xW + b
✓ Backward: dL/dX, dL/dW, dL/dB
✓ Gradient check passes
Expected: Can train simple 2-layer network on toy data

### Stage 4 (Softmax + Cross-Entropy)
✓ Softmax produces valid probabilities
✓ Cross-entropy loss computes correctly
✓ Gradient dL/dz = pred - truth works
Expected: Classification accuracy improves during training

### Stage 5 (Optimizers)
✓ SGD updates parameters in correct direction
✓ Momentum accumulates gradients smoothly
✓ Learning rate scaling works
Expected: Training is more stable than vanilla SGD

### Stage 6 (Conv2D Forward)
✓ Output shape formula correct
✓ Multiple filters work
✓ Stride and padding work
Expected: Can verify with simple hand-computed example

### Stage 7 (Conv2D Backward)
✓ Gradient check passes
✓ dL/dX shape correct
✓ dL/dW computed via input-gradient convolution
Expected: Can train small CNN on simple dataset

### Stage 8 (Pooling + LRN)
✓ MaxPool forward and backward
✓ Argmax routing works
✓ LRN forward pass correct
Expected: Pooling reduces spatial dimensions correctly

### Stage 9 (LeNet)
✓ LeNet-5 architecture defined
✓ Forward pass through full network
✓ Training loop works
✓ Reaches > 95% accuracy on MNIST
Expected: Can classify handwritten digits

### Stage 10 (AlexNet)
✓ AlexNet architecture matches paper
✓ Training loop with validation
✓ Model saving/loading
✓ Training on ImageNette
✓ Reaches > 80% top-1 accuracy on ImageNette
Expected: Full end-to-end training pipeline works

---

## References

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NIPS*.
2. LeCun, Y., Bottou, L., Bengio, Y., & LeCun, Y. (1998). Gradient-based learning applied to document recognition. *IEEE*.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

## Directory Structure

```
AlexNet_from_scratch/
├── learning.md                 (this file)
├── tensor.py                   (Stage 1)
├── autograd.py                 (Stage 2)
├── layers/
│   ├── __init__.py
│   ├── linear.py              (Stage 3)
│   ├── relu.py
│   ├── conv2d.py              (Stages 6-7)
│   ├── maxpool.py             (Stage 8)
│   ├── lrn.py                 (Stage 8)
│   └── dropout.py
├── losses/
│   ├── __init__.py
│   └── cross_entropy.py       (Stage 4)
├── optim/
│   ├── __init__.py
│   ├── sgd.py                 (Stage 5)
│   └── momentum.py
├── models/
│   ├── __init__.py
│   ├── lenet.py               (Stage 9)
│   └── alexnet.py             (Stage 10)
├── datasets/
│   ├── __init__.py
│   ├── mnist.py
│   └── imagenette.py
├── utils/
│   ├── __init__.py
│   ├── im2col.py
│   ├── visualization.py
│   └── config.py
├── tests/
│   ├── test_tensor.py
│   ├── test_autograd.py
│   ├── test_layers.py
│   ├── test_conv2d.py
│   └── test_gradients.py
├── train_lenet.py             (Stage 9)
├── train_alexnet.py           (Stage 10)
└── README.md
```

This comprehensive guide covers the complete journey from tensors to AlexNet!
