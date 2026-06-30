"""Train and evaluate AlexNet or VGG-19 on ImageNette.

Run from the project root, e.g.::

    python training/train.py --model alexnet --download
    python training/train.py --model vgg --epochs 20 --batch-size 16 --lr 0.01

The tinytorch framework is pure NumPy autograd, so convolution is slow on
224x224 images. For a quick smoke test, cap the work per epoch::

    python training/train.py --model alexnet --download --limit-batches 5

ImageNette is a 10-class subset of ImageNet, so num_classes defaults to 10.
"""

import argparse
import os
import sys
import time

import numpy as np

# tinytorch uses flat imports (from tensor import Tensor, from nn import ...),
# which rely on the package root being on sys.path. This script lives in
# training/, so tinytorch sits one directory up.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "tinytorch"))

from models import AlexNet, VGG19          # noqa: E402
from nn import CrossEntropyLoss            # noqa: E402
from optim import SGD                      # noqa: E402
from datasets import get_imagenette_loaders  # noqa: E402


# ===========================================================================
#  CONFIG — edit these to run the file directly (▶ Run / `python train.py`).
#  Every value can still be overridden on the command line, e.g.
#      python training/train.py --model vgg --epochs 20
# ===========================================================================
CONFIG = {
    "model":         "alexnet",   # "alexnet" or "vgg"
    "data_root":     "./data",    # where ImageNette lives / downloads to
    "download":      True,        # fetch ImageNette if missing (safe to leave True)
    "size":          160,         # download variant: 160, 320, or "full"
    "image_size":    224,         # input resize; must stay 224 for these models
    "num_classes":   10,          # ImageNette has 10 classes
    "epochs":        10,
    "batch_size":    16,
    "lr":            1e-2,
    "momentum":      0.9,
    "weight_decay":  5e-4,
    "limit_batches": None,        # set to e.g. 5 for a quick smoke test
    "checkpoint_dir": "./checkpoints",
    "save_every":    1,           # save every N epochs (0 disables)
    "resume":        None,        # path to a .npz checkpoint to resume from
}


def build_model(name: str, num_classes: int):
    """Construct the requested architecture."""
    name = name.lower()
    if name == "alexnet":
        return AlexNet(num_classes=num_classes)
    if name in ("vgg", "vgg19"):
        return VGG19(num_classes=num_classes)
    raise ValueError(f"unknown model '{name}' (choose 'alexnet' or 'vgg')")


def save_checkpoint(path, model, optimizer, epoch, best_val_acc, model_name):
    """Write model weights, optimizer momentum buffers and metadata to `path`.

    Parameters and velocities are stored by position in model.parameters()
    order, which is deterministic for a fixed architecture.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blob = {"epoch": np.int64(epoch),
            "best_val_acc": np.float64(best_val_acc),
            "model_name": np.array(model_name)}
    for i, p in enumerate(model.parameters()):
        blob[f"p{i}"] = p.data
    for i, v in enumerate(optimizer._velocities):
        if v is not None:
            blob[f"v{i}"] = v
    np.savez(path, **blob)
    print(f"  saved checkpoint -> {path}")


def load_checkpoint(path, model, optimizer, model_name):
    """Restore weights, momentum buffers and metadata saved by save_checkpoint.

    Returns (next_epoch, best_val_acc) so training can resume.
    """
    ckpt = np.load(path, allow_pickle=True)
    saved_name = str(ckpt["model_name"])
    if saved_name != model_name:
        raise ValueError(
            f"checkpoint is for model '{saved_name}' but --model is '{model_name}'"
        )
    params = list(model.parameters())
    for i, p in enumerate(params):
        p.data = ckpt[f"p{i}"]
    optimizer._velocities = [
        ckpt[f"v{i}"] if f"v{i}" in ckpt.files else None
        for i in range(len(params))
    ]
    epoch = int(ckpt["epoch"])
    best_val_acc = float(ckpt["best_val_acc"])
    print(f"  resumed from {path} (epoch {epoch}, best val acc {best_val_acc:.4f})")
    return epoch + 1, best_val_acc


def accuracy(logits, labels) -> float:
    """Fraction of correct top-1 predictions for one batch."""
    preds = logits.data.argmax(axis=1)
    return float((preds == labels).mean())


def run_epoch(model, loader, criterion, optimizer=None, limit_batches=None):
    """One pass over a loader. Trains if an optimizer is given, else evaluates.

    Returns (average loss, average accuracy) over the batches seen.
    """
    train = optimizer is not None
    model.train(train)

    n_batches = len(loader) if limit_batches is None else min(limit_batches, len(loader))
    loss_sum = acc_sum = 0.0
    seen = 0

    for i, (x, labels) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break

        logits = model(x)
        loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.data)
        acc_sum += accuracy(logits, labels)
        seen += 1

        phase = "train" if train else "val"
        print(
            f"\r  [{phase}] batch {seen}/{n_batches}  "
            f"loss {loss_sum / seen:.4f}  acc {acc_sum / seen:.4f}",
            end="",
            flush=True,
        )

    print()
    if seen == 0:
        return 0.0, 0.0
    return loss_sum / seen, acc_sum / seen


def main():
    # argparse defaults come straight from CONFIG, so editing CONFIG and
    # running the file is equivalent to passing the corresponding flags.
    parser = argparse.ArgumentParser(description="Train AlexNet/VGG-19 on ImageNette.")
    parser.add_argument("--model", default=CONFIG["model"], choices=["alexnet", "vgg", "vgg19"],
                        help="architecture to train")
    parser.add_argument("--data-root", default=CONFIG["data_root"],
                        help="directory for the ImageNette dataset")
    parser.add_argument("--download", action="store_true", default=CONFIG["download"],
                        help="download ImageNette if not already present")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="never download (overrides CONFIG)")
    parser.add_argument("--size", default=CONFIG["size"], type=lambda v: v if v == "full" else int(v),
                        help="download variant: 160, 320, or full")
    parser.add_argument("--image-size", default=CONFIG["image_size"], type=int,
                        help="resize target; must be 224 for these models")
    parser.add_argument("--num-classes", default=CONFIG["num_classes"], type=int,
                        help="number of output classes (ImageNette has 10)")
    parser.add_argument("--epochs", default=CONFIG["epochs"], type=int)
    parser.add_argument("--batch-size", default=CONFIG["batch_size"], type=int)
    parser.add_argument("--lr", default=CONFIG["lr"], type=float)
    parser.add_argument("--momentum", default=CONFIG["momentum"], type=float)
    parser.add_argument("--weight-decay", default=CONFIG["weight_decay"], type=float)
    parser.add_argument("--limit-batches", default=CONFIG["limit_batches"], type=int,
                        help="cap batches per epoch for a quick smoke test")
    parser.add_argument("--checkpoint-dir", default=CONFIG["checkpoint_dir"],
                        help="directory for saved checkpoints")
    parser.add_argument("--save-every", default=CONFIG["save_every"], type=int,
                        help="save a checkpoint every N epochs (0 disables)")
    parser.add_argument("--resume", default=CONFIG["resume"],
                        help="path to a checkpoint .npz to resume from")
    args = parser.parse_args()

    if args.image_size != 224:
        print(f"warning: AlexNet/VGG-19 expect a 224x224 input; "
              f"image_size={args.image_size} will break the classifier shapes.")

    print(f"Loading ImageNette from '{args.data_root}' ...")
    train_loader, val_loader = get_imagenette_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        size=args.size,
        image_size=args.image_size,
        download=args.download,
        normalize=True,
    )

    model = build_model(args.model, args.num_classes)
    criterion = CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    print(f"\nModel: {args.model}  |  classes: {args.num_classes}  |  "
          f"batch: {args.batch_size}  |  lr: {args.lr}\n")

    start_epoch = 1
    best_val_acc = 0.0
    if args.resume:
        start_epoch, best_val_acc = load_checkpoint(
            args.resume, model, optimizer, args.model,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, args.limit_batches,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, args.limit_batches,
        )

        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)
        print(
            f"  -> train loss {train_loss:.4f} acc {train_acc:.4f}  |  "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}  |  "
            f"{time.time() - t0:.1f}s\n"
        )

        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"{args.model}_last.npz"),
                model, optimizer, epoch, best_val_acc, args.model,
            )
        if is_best:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"{args.model}_best.npz"),
                model, optimizer, epoch, best_val_acc, args.model,
            )

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
