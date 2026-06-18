"""
Train LeNet-5 on MNIST.

Usage:
    python train_lenet.py                  # defaults: 10 epochs, batch=64, lr=1e-3
    python train_lenet.py --epochs 20 --lr 0.001 --batch-size 128
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensor import Tensor
from models import LeNet5
from losses import cross_entropy
from optim import Adam
from datasets.mnist import get_mnist_loaders


# --------------------------------------------------------------------------- #

def accuracy(logits: Tensor, labels: np.ndarray) -> float:
    preds = logits.data.argmax(axis=1)
    return float((preds == labels).mean())


def evaluate(model: LeNet5, loader_fn) -> tuple:
    model.eval()
    total_loss = total_acc = n = 0
    for images, labels in loader_fn():
        logits = model(images)
        loss = cross_entropy(logits, Tensor(labels))
        total_loss += float(loss.data)
        total_acc  += accuracy(logits, labels)
        n += 1
    return total_loss / n, total_acc / n


def train(epochs: int = 10, batch_size: int = 64, lr: float = 1e-3, data_root: str = './data'):
    print("=" * 60)
    print("Training LeNet-5 on MNIST")
    print("=" * 60)

    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, root=data_root)
    model = LeNet5(num_classes=10)
    optimiser = Adam(model.parameters(), lr=lr)

    print(f"Parameters: {sum(p.data.size for p in model.parameters()):,}")
    print()

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = t_acc = n = 0

        for images, labels in train_loader():
            optimiser.zero_grad()
            logits = model(images)
            loss = cross_entropy(logits, Tensor(labels))
            loss.backward()
            optimiser.step()

            t_loss += float(loss.data)
            t_acc  += accuracy(logits, labels)
            n += 1

        val_loss, val_acc = evaluate(model, test_loader)

        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"| train loss {t_loss/n:.4f}  acc {t_acc/n:.4f}  "
            f"| val loss {val_loss:.4f}  acc {val_acc:.4f}"
        )

    print()
    print("Training complete.")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--data-root",  type=str,   default="./data")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_root=args.data_root,
    )
