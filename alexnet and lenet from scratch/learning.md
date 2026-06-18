# Building LeNet and AlexNet from Scratch — Complete Learning Guide

A step-by-step guide for understanding **what** to build, **why** each piece exists, and **in what order** to build it. Every concept here is implemented in this repository using only NumPy — no PyTorch, no TensorFlow.

---

## The Big Picture

A neural network framework has four layers of abstraction:

```
1. Tensor          — the data container + autograd engine
2. nn layers       — parameterised operations (Conv, Linear, …)
3. Loss + Optim    — how to measure error and update weights
4. Model           — a composition of layers for a specific task
```

You build from the bottom up. You cannot train a model without loss functions. You cannot define a loss without tensors. Each layer depends only on what came before.

---

## Phase 1 — Tensor and Autograd (already done: tensor.py)

**What it is:** A `Tensor` wraps a NumPy array and records every operation applied to it. When you call `.backward()`, it walks the recorded graph in reverse and computes gradients automatically.

**Key concepts to understand:**

### 1.1 Why store `_prev` and `_backward`?
Every operation (add, multiply, matmul, …) creates a new Tensor and attaches:
- `_prev` — the input Tensors that produced it
- `_backward` — a closure that knows how to push gradients *back* to those inputs

```
    x ─── [mul] ─── z
    y ─┘
```
When `z.backward()` runs, it calls `z._backward()`, which computes `x.grad` and `y.grad` using the chain rule, then calls `x._backward()` and `y._backward()` recursively.

### 1.2 Topological sort
`backward()` builds a topological order of the entire graph before traversing it. This ensures each node's gradient is fully accumulated before it is used as input to the next backward pass.

```python
# tensor.py — backward()
def _build_topo(v):
    if id(v) in visited: return
    visited.add(id(v))
    for child in v._prev:
        _build_topo(child)
    topo.append(v)
```

### 1.3 `_sum_to_shape` — why broadcasting needs a special backward
When NumPy broadcasts `(3,) + (3, 3)`, the smaller tensor is implicitly repeated. The backward must *undo* that repetition by summing along the broadcast axes.

### 1.4 Key operations already implemented
| Op | Forward | Backward |
|---|---|---|
| `+`, `-`, `*`, `/` | element-wise | chain rule |
| `**` | power | power rule |
| `@` (matmul) | GEMM | `dA = dOut @ B.T`, `dB = A.T @ dOut` |
| `sum`, `mean` | reduce | broadcast back |
| `exp`, `log`, `sqrt` | element-wise | analytic derivative |
| `reshape`, `flatten` | shape only | reshape grad back |

---

## Phase 2 — The Module Base Class (nn/module.py)

**Build this second** — before any layer.

**What it is:** A base class that every layer inherits from. It provides automatic parameter tracking and the `train`/`eval` mode switch.

**Why you need it:**
- You need a uniform way to collect *all* learnable parameters (weights, biases) from a complex model so the optimiser can update them all.
- Layers like Dropout behave differently during training vs evaluation.

**How it works:**
```python
class Module:
    def __setattr__(self, name, value):
        if isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value   # auto-registered!
        elif isinstance(value, Module):
            self._modules[name] = value       # sub-module registered
        object.__setattr__(self, name, value)

    def parameters(self):
        yield from self._parameters.values()
        for mod in self._modules.values():
            yield from mod.parameters()       # recursive!
```
When you write `self.W = Tensor(..., requires_grad=True)` inside a layer's `__init__`, `__setattr__` intercepts it and records `W` as a learnable parameter. No extra registration needed.

---

## Phase 3 — Layers (nn/)

Build these in dependency order. Each layer is a `Module` subclass.

### 3.1 Linear (nn/linear.py) — simplest layer

```
forward:  out = x @ W + b
backward: dW = x.T @ dout,  dx = dout @ W.T
```

Uses the `@` operator from tensor.py which already has the right backward. The bias addition uses `+` which also has its backward. So the entire backward is **free** — it flows through the tensor operations automatically.

**Initialisation:** Kaiming He (`std = sqrt(2/fan_in)`) — designed for ReLU activations. Prevents gradients from vanishing or exploding in deep networks.

### 3.2 ReLU (nn/relu.py) — first custom backward

```
forward:  f(x) = max(0, x)
backward: df/dx = 1 if x > 0 else 0
```

This is the first place where you cannot rely on tensor.py's existing ops — there is no `max(scalar, tensor)` with grad. You must write the backward manually.

```python
result = np.maximum(0, x.data)
# In _backward:
grad = out.grad * (result > 0).astype(dtype)
```

The mask `(result > 0)` is captured at forward time and reused in backward. This is the pattern for all custom layers.

### 3.3 Conv2D (nn/conv2d.py) — the hardest layer

**Conceptual insight:** Convolution is just matrix multiplication in disguise.

**The im2col trick:**
Instead of nested loops over every output position, rearrange the input into a 2-D matrix where each *row* contains all the input values that contribute to one output pixel, then do a single matrix multiply.

```
input:   (N, C, H, W)
col:     (N*H_out*W_out,  C*kH*kW)     ← im2col
W_flat:  (C_out,          C*kH*kW)     ← reshape weights
output:  col @ W_flat.T → (N*H_out*W_out, C_out) → reshape → (N, C_out, H_out, W_out)
```

**Output size formula:**
```
H_out = floor((H + 2*padding - kernel_size) / stride) + 1
```

**Backward:**
```
dW   = dout_flat.T @ col         # how much each weight contributed
dcol = dout_flat  @ W_flat       # how much each input patch contributed
dx   = col2im(dcol, ...)         # scatter patch grads back to input positions
```

`col2im` is the exact inverse of `im2col` — it accumulates overlapping patch gradients back into the input.

**Bias:** Added *after* the manual backward by reusing the Tensor `+` operator:
```python
out = out + self.b.reshape(1, C_out, 1, 1)
```
This piggybacks on tensor.py's `__add__` backward, which correctly sums the bias gradient via `_sum_to_shape`.

### 3.4 MaxPool2D (nn/maxpool.py)

```
forward:  take the max over each k×k window
backward: route gradient only to the max position(s)
```

Key detail: if multiple positions tie for the max, split the gradient equally (mask / count_of_maxes). This avoids artificially amplifying gradients.

### 3.5 AvgPool2D (nn/avgpool.py)

```
forward:  average over each k×k window
backward: distribute gradient equally (÷ k²) over the window
```

Used in the original LeNet-5. MaxPool is more common in modern networks.

### 3.6 Dropout (nn/dropout.py)

**Why:** A regularisation technique. During training, randomly zero some activations so the network cannot memorise specific neuron patterns.

**Inverted dropout (the modern version):**
- Zero activations with probability `p`
- Scale surviving activations by `1/(1-p)` during training
- No scaling needed at test time — the expected value is already correct

**The training flag:** Dropout is a no-op during `.eval()` mode (inference). The `Module.training` flag, set by `.train()`/`.eval()`, controls this.

### 3.7 LocalResponseNorm (nn/lrn.py) — AlexNet only

Normalises each activation across a neighbourhood of *channels*:
```
b_i = a_i / (k + α * Σ_{j=i±n/2} a_j²)^β
```

Inspired by lateral inhibition in the brain: strong activations suppress neighbouring ones. It fell out of use after BatchNorm proved more effective, but AlexNet uses it.

**Backward:** Two-term derivative (product rule on the denominator):
```
dx_i = dout_i / scale_i^β  -  2αβ·a_i · Σ_{k: i∈window(k)} (dout_k · b_k / scale_k)
```

### 3.8 Flatten / Softmax (nn/flatten.py, nn/softmax.py)

Flatten delegates entirely to `tensor.reshape` — the backward is already handled.

Softmax is used standalone when you need probabilities. For *training*, always prefer raw logits + `cross_entropy` which fuses the operations for numerical stability.

---

## Phase 4 — Loss Functions (losses/)

### 4.1 Cross-Entropy (losses/cross_entropy.py)

The standard loss for multi-class classification.

**Why the fused implementation?**

Naively: `loss = -log(softmax(logits)[correct_class])`

The problem: `softmax` involves `exp(logits)` which can overflow for large logits. The fix is to subtract the row maximum before `exp` — this shifts the values but does not change the softmax output.

```python
shifted = logits - logits.max(axis=1, keepdims=True)
exp_x   = np.exp(shifted)
probs   = exp_x / exp_x.sum(axis=1, keepdims=True)
loss    = -log(probs[range(N), targets]).mean()
```

**Gradient (the elegant result):**
```
d(loss)/d(logit_i) = (p_i - 1_[i==y]) / N
```
The gradient is just `softmax_output - one_hot_label`, divided by batch size. This is why softmax + cross-entropy is always implemented together.

### 4.2 MSE (losses/mse.py)

For regression tasks. Implemented entirely via tensor operations (subtract, square, mean) — no custom backward needed.

---

## Phase 5 — Optimisers (optim/)

Optimisers read `.grad` off each parameter and update `.data`. They do not interact with the computation graph at all.

### 5.1 SGD (optim/sgd.py)

```
theta ← theta - lr * grad
```

With momentum:
```
v ← momentum * v + grad
theta ← theta - lr * v
```

Momentum gives SGD "inertia" — it accelerates in consistent gradient directions and dampens oscillations. Used for AlexNet in the original paper (`momentum=0.9`).

With weight decay (L2 regularisation):
```
grad ← grad + weight_decay * theta
```

Penalises large weights, reducing overfitting.

### 5.2 Adam (optim/adam.py)

Maintains per-parameter adaptive learning rates by tracking the first moment (mean of gradients) and second moment (variance of gradients):

```
m_t = β₁·m_{t-1} + (1-β₁)·g          # exponential moving average of grad
v_t = β₂·v_{t-1} + (1-β₂)·g²         # exponential moving average of grad²
m̂ = m_t / (1 - β₁ᵗ)                  # bias correction (important early on)
v̂ = v_t / (1 - β₂ᵗ)
θ ← θ - lr * m̂ / (√v̂ + ε)
```

Adam is the go-to default: it converges faster than SGD on most tasks and is less sensitive to learning rate choice. Use `lr=1e-3` to start.

---

## Phase 6 — Models (models/)

Models are just Modules that compose layers. No new concepts — just architecture.

### 6.1 LeNet-5 (models/lenet.py)

Original paper: LeCun et al., 1989/1998. Designed for 32×32 greyscale digit images. This version adapts it for MNIST (28×28).

```
Input: (N, 1, 28, 28)

Conv1(1→6,  5×5, pad=2) → ReLU → MaxPool(2×2)   → (N, 6,  14, 14)
Conv2(6→16, 5×5,      ) → ReLU → MaxPool(2×2)   → (N, 16,  5,  5)
Flatten                                           → (N, 400)
FC(400→120) → ReLU
FC(120→84)  → ReLU
FC(84→10)                                         → (N, 10) logits
```

**Why pad=2 on Conv1?** To keep the output 28×28 after a 5×5 kernel. Without padding: `(28-5)/1 + 1 = 24`.

**Why no activation on the final FC?** The last layer outputs raw logits. The `cross_entropy` loss applies softmax internally.

### 6.2 AlexNet (models/alexnet.py)

Original paper: Krizhevsky, Sutskever, Hinton — 2012 ImageNet winner.

```
Input: (N, 3, 224, 224)

Conv1(3→96,   11×11, s=4)  → ReLU → LRN → MaxPool(3,s=2)  → (N, 96,  26, 26)
Conv2(96→256,  5×5,  pad=2)→ ReLU → LRN → MaxPool(3,s=2)  → (N, 256, 12, 12)
Conv3(256→384, 3×3,  pad=1)→ ReLU                          → (N, 384, 12, 12)
Conv4(384→384, 3×3,  pad=1)→ ReLU                          → (N, 384, 12, 12)
Conv5(384→256, 3×3,  pad=1)→ ReLU → MaxPool(3,s=2)         → (N, 256,  5,  5)
Flatten                                                     → (N, 6400)
Dropout(0.5) → FC(6400→4096) → ReLU
Dropout(0.5) → FC(4096→4096) → ReLU
FC(4096→10)                                                 → (N, 10) logits
```

**Why three consecutive convolutions (Conv3-5) without pooling?** Stacking smaller 3×3 kernels gives a larger effective receptive field (3 layers of 3×3 = 7×7 equivalent) with fewer parameters and more non-linearities.

**Why Dropout before the FC layers?** AlexNet introduced Dropout as a regulariser. Without it, the 4096-dim FC layers overfit badly on any dataset.

---

## Phase 7 — Training Loop

The training loop is the same pattern for both models:

```python
for epoch in range(epochs):
    model.train()                           # enable Dropout etc.
    for images, labels in train_loader():
        optimiser.zero_grad()               # 1. clear old gradients
        logits = model(images)              # 2. forward pass (builds graph)
        loss   = cross_entropy(logits, Tensor(labels))  # 3. compute loss
        loss.backward()                     # 4. backprop (fills .grad on params)
        optimiser.step()                    # 5. update weights

    model.eval()                            # disable Dropout
    # ... evaluate on val set
```

These five lines — zero → forward → loss → backward → step — are the heartbeat of every gradient-based learning algorithm.

---

## Build Order Summary

```
Step 1  tensor.py              ← already done
Step 2  nn/module.py           ← base class, needed by every layer
Step 3  nn/linear.py           ← simplest layer, uses tensor ops
Step 4  nn/relu.py             ← first custom backward
Step 5  nn/conv2d.py           ← hardest layer (im2col)
Step 6  nn/maxpool.py          ← custom backward, no parameters
Step 7  nn/avgpool.py          ← variant of maxpool
Step 8  nn/dropout.py          ← uses training flag
Step 9  nn/lrn.py              ← complex backward, AlexNet-specific
Step 10 nn/activations.py      ← Sigmoid, Tanh (same pattern as ReLU)
Step 11 nn/softmax.py          ← Jacobian-vector product backward
Step 12 nn/flatten.py          ← wraps tensor.reshape
Step 13 nn/sequential.py       ← container, composes layers
Step 14 losses/cross_entropy.py← fused softmax-NLL for stability
Step 15 losses/mse.py          ← trivial, built from tensor ops
Step 16 optim/sgd.py           ← reads .grad, writes .data
Step 17 optim/adam.py          ← adaptive lr, moment estimates
Step 18 models/lenet.py        ← compose layers for MNIST
Step 19 models/alexnet.py      ← compose layers for ImageNette
Step 20 train_lenet.py         ← end-to-end training script
Step 21 train_alexnet.py       ← end-to-end training script
```

---

## How to Run

### LeNet on MNIST (quick experiment — ~10 min on CPU)

```bash
cd "alexnet and lenet from scratch"
python train_lenet.py --epochs 10 --batch-size 64 --lr 0.001
```

Expected: reaches ~98% test accuracy after 10 epochs.

### AlexNet on ImageNette (serious experiment — GPU recommended)

Download dataset first:
```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -xf imagenette2.tgz
```

Then:
```bash
python train_alexnet.py --data-root ./imagenette2 --epochs 30 --batch-size 8
```

Note: the pure-NumPy backend is educational, not production-fast. For real-world use, switch to PyTorch and substitute `nn.Conv2d`, `nn.Linear`, etc. — the architecture and training loop are identical.

---

## Common Mistakes and Why They Happen

| Mistake | Symptom | Fix |
|---|---|---|
| Forgetting `zero_grad()` before backward | Gradients accumulate across batches; loss diverges | Always call before forward |
| Returning logits instead of applying softmax to loss | Loss is NaN | Use `cross_entropy(logits, …)`, not `cross_entropy(softmax(logits), …)` |
| Using softmax output as logits in cross_entropy | Double-softmax; probabilities collapse to near-uniform | Pass raw logits |
| Not switching to `model.eval()` for validation | Dropout randomly zeros activations during val; metrics are noisy | Always `model.eval()` before val loop |
| Conv backward uses wrong dimension ordering | Gradients are wrong shapes | Remember: `dout.transpose(0,2,3,1)` to move C_out to last dim before reshape |
| `broadcast_to` without `.copy()` | `ValueError: assignment destination is read-only` | Add `.copy()` after any `broadcast_to` before in-place ops |

---

## Key Mathematical Identities to Remember

| Concept | Formula |
|---|---|
| Chain rule | `d(f∘g)/dx = df/dg · dg/dx` |
| Matmul grad | `dL/dA = dL/dC @ B.T`,  `dL/dB = A.T @ dL/dC` |
| Softmax grad | `dL/dz_i = s_i(dL/ds_i - Σ_j dL/ds_j · s_j)` |
| Cross-entropy grad | `dL/dz_i = (s_i - y_i) / N` |
| Conv output size | `floor((H + 2p - k) / s) + 1` |
| Im2col converts | conv → GEMM (matrix multiply) |
| Inverted dropout scale | `1 / (1 - p)` at train time |

---

## What to Explore Next

1. **Batch Normalisation** — replaces LRN in modern networks; normalises activations across the batch, enabling much higher learning rates.
2. **ResNet skip connections** — `out = F(x) + x`; makes 100+ layer networks trainable by giving gradients a "shortcut".
3. **Adam variants** — AdamW decouples weight decay from the moment estimates for better regularisation.
4. **Data augmentation** — random crop, flip, colour jitter; improves generalisation without extra data.
5. **Learning rate schedules** — cosine annealing, step decay; most gains in top-1 accuracy come from the LR schedule, not the architecture.
