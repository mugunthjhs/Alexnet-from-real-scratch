"""
Train AlexNet on ImageNette (10-class ImageNet subset).

Usage:
    python train_alexnet.py --data-root /path/to/imagenette2
    python train_alexnet.py --data-root /path/to/imagenette2 --epochs 30 --batch-size 16

NOTE: AlexNet is large. With a pure-NumPy backend expect ~1-2 min/epoch on CPU
for batch_size=8 on a modern laptop.  Start with a small batch to verify
correctness before a full run.

Download ImageNette:
    wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
    tar -xf imagenette2.tgz
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensor import Tensor
from models import AlexNet
from losses import cross_entropy
from optim import SGD
from datasets.imagenette import get_imagenette_loaders


# --------------------------------------------------------------------------- #

def accuracy(logits: Tensor, labels: np.ndarray) -> float:
    preds = logits.data.argmax(axis=1)
    return float((preds == labels).mean())


def evaluate(model: AlexNet, loader_fn) -> tuple:
    model.eval()
    total_loss = total_acc = n = 0
    for images, labels in loader_fn():
        logits = model(images)
        loss = cross_entropy(logits, Tensor(labels))
        total_loss += float(loss.data)
        total_acc  += accuracy(logits, labels)
        n += 1
    return total_loss / n, total_acc / n


def train(
    data_root: str,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    input_size: int = 224,
):
    print("=" * 60)
    print("Training AlexNet on ImageNette")
    print("=" * 60)

    train_loader, val_loader = get_imagenette_loaders(
        root=data_root, batch_size=batch_size, size=input_size
    )

    model = AlexNet(num_classes=10, input_size=input_size)
    optimiser = SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    print(f"Parameters: {sum(p.data.size for p in model.parameters()):,}")
    print(f"FC input size: {AlexNet._feature_output_size(input_size)}")
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

            if n % 10 == 0:
                print(f"  [{epoch}/{epochs}] batch {n:>4}  "
                      f"loss {t_loss/n:.4f}  acc {t_acc/n:.4f}", end="\r")

        val_loss, val_acc = evaluate(model, val_loader)

        print(
            f"\nEpoch {epoch:>3}/{epochs}  "
            f"| train loss {t_loss/n:.4f}  acc {t_acc/n:.4f}  "
            f"| val loss {val_loss:.4f}  acc {val_acc:.4f}"
        )

    print()
    print("Training complete.")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",    type=str,   required=True,
                        help="Path to imagenette2 directory")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--momentum",     type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--input-size",   type=int,   default=224)
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        input_size=args.input_size,
    )
