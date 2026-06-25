# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **from-scratch implementation of AlexNet and LeNet using only NumPy**—no PyTorch, TensorFlow, or external ML frameworks. It's an educational project that demonstrates:

- Building a complete autograd engine (Tensor with backpropagation)
- Implementing neural network layers (Conv2D, Linear, Dropout, etc.)
- Loss functions and optimizers (SGD, Adam)
- Training deep models on real datasets (ImageNette, MNIST)

Two main directories:
- **`alexnet and lenet from scratch/`** — Core framework and training scripts
- **`learning_alexnet/`** — Jupyter notebooks and demos for experimentation

## Code Architecture

### Layer 1: Tensor & Autograd (`tensor.py`)

A NumPy-backed `Tensor` class that records all operations and enables automatic differentiation via `backward()`.

**Key concepts:**
- `requires_grad=True` enables gradient tracking
- `_prev` set stores the input Tensors that produced this Tensor
- `_backward` is a closure that applies the chain rule to compute gradients
- `backward()` builds a topological sort of the computation graph, then traverses in reverse order

**Common operations:**
- Arithmetic: `+`, `-`, `*`, `/`, `**`
- Linear algebra: `@` (matmul), `.T` (transpose), reshape, squeeze, expand
- Reductions: sum, mean
- Activation-friendly: relu, softmax (backward implemented)

### Layer 2: NN Layers (`nn/` directory)

Base class `Module` (similar to PyTorch) that:
- Auto-registers `Tensor` attributes with `requires_grad=True` as parameters
- Auto-registers other `Module` instances as submodules
- Provides `parameters()` iterator and `zero_grad()` for training loops
- Supports `train()/eval()` modes (affects Dropout, BatchNorm if added)

**Available layers:**
- **Conv2D** — 2D convolution with padding/stride/groups
- **Linear** — Fully connected layer
- **MaxPool2D, AvgPool2D** — Pooling operations
- **ReLU, Softmax** — Activations
- **Dropout** — Stochastic regularization (respects `training` flag)
- **Flatten** — Reshape batch to vectors
- **LocalResponseNorm (LRN)** — AlexNet's local response normalization
- **Sequential** — Compose layers (forward passes through each in order)

**Pattern:** Each layer inherits `Module` and implements `forward(x)`, which may call other layers or raw Tensor ops.

### Layer 3: Loss Functions (`losses/`)

Standalone functions that take predictions and targets, return a scalar Tensor with gradients:
- **cross_entropy(logits, targets)** — For classification
- **mse(predictions, targets)** — For regression

### Layer 4: Optimizers (`optim/`)

**SGD** with momentum and weight decay, **Adam** for adaptive learning rates.

Both iterate over `model.parameters()`, accumulate gradients, then update weights.

### Layer 5: Models (`models/`)

**AlexNet** and **LeNet** as `Module` subclasses. Each:
- Stacks layers in `__init__`
- Implements `forward()` to compose them
- Optionally includes a `_feature_output_size()` static method to compute flattened sizes after convolutions

**AlexNet:** 5 conv layers (with LRN and pooling) + 3 FC layers for 10-class (ImageNette) or 1000-class (full ImageNet) classification.

**LeNet:** 2 conv layers + 2 FC layers, designed for MNIST (28×28 grayscale).

### Layer 6: Datasets (`datasets/`)

Data loaders that yield (batch_images, batch_labels) tuples:
- **imagenette.py** — ImageNette (10-class ImageNet subset, 224×224 RGB)
- **mnist.py** — MNIST (10-class handwritten digits, 28×28 grayscale)

Loaders handle image normalization and optional augmentation.

## Common Commands

### Run AlexNet Training

```bash
# Download ImageNette first
# wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz && tar -xf imagenette2.tgz

cd "alexnet and lenet from scratch"
python train_alexnet.py --data-root /path/to/imagenette2 --epochs 30 --batch-size 8
```

**Parameters:**
- `--data-root` (required) — Path to imagenette2 directory
- `--epochs` (default 30) — Number of training epochs
- `--batch-size` (default 16) — Batch size per iteration
- `--lr` (default 0.01) — Learning rate
- `--momentum` (default 0.9) — SGD momentum
- `--weight-decay` (default 5e-4) — L2 regularization
- `--input-size` (default 224) — Input image dimension (224 or 227)

**Expected performance:** ~1–2 minutes per epoch on modern CPU (batch_size=8), much faster on GPU (if NumPy is swapped for a GPU-accelerated backend).

### Run LeNet Training

```bash
cd "alexnet and lenet from scratch"
python train_lenet.py
```

LeNet trains on MNIST automatically (downloads if needed). Much faster (~seconds per epoch).

### Explore with Jupyter

```bash
# AlexNet demo notebook
jupyter notebook "learning_alexnet/alexnet_demo.ipynb"

# Backprop tutorial
jupyter notebook "backprogation_from scratch.ipynb"
```

## Key Implementation Details

### Tensor Operations & Gradient Flow

Every Tensor operation that supports differentiation returns a new Tensor whose `_backward` closure knows the chain-rule formula. For example:

```python
z = x @ y  # matrix multiplication
# z._backward is a closure that, when called with z.grad, computes x.grad and y.grad
z.backward()  # runs topological sort, then applies chain rule in reverse
print(x.grad, y.grad)
```

**Broadcasting:** Tensor ops follow NumPy broadcasting rules. Gradients are summed over broadcast dimensions using `_sum_to_shape()`.

### Conv2D Implementation

Convolution is implemented as an im2col (image-to-column) transformation + matrix multiply + reshape:
1. Unfold input patches into columns (shape: `[ks*ks*C_in, H_out*W_out]`)
2. Reshape weights as `[C_out, ks*ks*C_in]`
3. Perform `weights @ patches` → `[C_out, H_out*W_out]`
4. Reshape back to `[B, C_out, H_out, W_out]`

Backward pass computes input and weight gradients via transpose and re-cols.

### Training Loop Pattern

```python
for epoch in range(epochs):
    for images, labels in train_loader():
        # Forward pass
        logits = model(images)
        loss = cross_entropy(logits, Tensor(labels))
        
        # Backward pass
        loss.backward()
        
        # Update
        optimiser.step()
        optimiser.zero_grad()
```

- `loss.backward()` populates all `param.grad` values
- `optimiser.step()` updates params: `p.data -= lr * p.grad`
- `optimiser.zero_grad()` clears grads for the next iteration

### No Batch Normalization

This implementation omits BatchNorm (which would require running statistics). Models use Dropout and LRN instead.

## Important Notes

### Performance

- **NumPy-only:** Pure Python loops and NumPy operations are *slow*. AlexNet training takes ~1–2 min/epoch on CPU.
- To speed up, swap NumPy calls for CuPy (GPU) or use a compiled backend (Cython, Numba), but the API remains unchanged.
- Start with small batches (batch_size=8, 1–2 epochs) to verify correctness before full runs.

### Memory

- Large models (AlexNet) on large batches can exhaust RAM. Monitor memory usage.
- The entire computation graph is kept in memory during `backward()`. For very large graphs, gradient accumulation or checkpointing would help (not implemented).

### Path Handling

- Training scripts insert their parent directory into `sys.path` for relative imports.
- Always run scripts from their own directory or adjust paths if moving code around.

### Testing & Validation

- No dedicated test suite; correctness is verified via training curves and comparison with reference implementations.
- The learning.md file in the main module directory documents each component—read it to understand design choices.

## Repo Structure

```
.
├── README.md
├── CLAUDE.md
├── backprogation_from scratch.ipynb      # Tutorial on backprop math
├── alexnet and lenet from scratch/
│   ├── tensor.py                          # Tensor + autograd engine
│   ├── train_alexnet.py                   # Training script for AlexNet
│   ├── train_lenet.py                     # Training script for LeNet
│   ├── learning.md                        # Detailed design documentation
│   ├── nn/                                # Layer implementations
│   │   ├── module.py                      # Base Module class
│   │   ├── conv2d.py, linear.py, ...
│   ├── models/                            # Model definitions
│   │   ├── alexnet.py, lenet.py
│   ├── losses/                            # Loss functions
│   ├── optim/                             # Optimizers
│   └── datasets/                          # Data loaders
├── learning_alexnet/
│   ├── tensor.py                          # Newer variant (may differ from main)
│   ├── alexnet_demo.ipynb                 # Interactive demo
│   ├── draw.py                            # Visualization helper
│   └── nn/                                # Alternate layer implementations
└── venv/                                  # Virtual environment (excluded from version control)
```

## Development Workflow

1. **Modify tensor.py or an nn layer** → Run a training script with `--epochs 1 --batch-size 8` to quickly verify correctness.
2. **Add a new loss or optimizer** → Test on LeNet first (MNIST trains in seconds).
3. **Experiment interactively** → Use the Jupyter notebooks in `learning_alexnet/`.
4. **Understand design rationale** → Read `learning.md` before modifying the Tensor or backward pass.

## Links

- **learning.md** — High-level guide explaining why each component exists and what to implement first.
- **alexnet_demo.ipynb** — Live notebook walking through AlexNet's forward and backward passes.
- **backprogation_from scratch.ipynb** — Detailed math and implementation guide for backpropagation.

## Notes for Future Contributors

- When modifying Tensor operations, add both `forward` and `backward` logic; the backward part is critical for training.
- If you add a new layer, implement it in `nn/`, inherit from `Module`, and test it with a full training loop.
- Conv2D's im2col approach is not the fastest; consider more efficient algorithms (e.g., Winograd) if performance becomes critical.
- Synchronize implementations across `alexnet and lenet from scratch/` and `learning_alexnet/` if you edit shared code.
