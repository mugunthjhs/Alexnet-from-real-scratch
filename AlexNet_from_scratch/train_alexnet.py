"""
Training Script for AlexNet on ImageNette

This script trains AlexNet on the ImageNette dataset (10-class subset of ImageNet).

Architecture: AlexNet with modifications for ImageNette
- Input: 224×224 RGB images
- Output: 10 class predictions
- Optimizations:
  * Dropout (0.5) to prevent overfitting
  * Local Response Normalization (LRN)
  * ReLU activations

Expected results:
- After 20 epochs: ~80% top-1 accuracy
- After 40 epochs: ~85% top-1 accuracy

Usage:
python train_alexnet.py --data-path /path/to/imagenette2 --epochs 40 --batch-size 128

Requirements:
- Download ImageNette: https://github.com/fastai/imagenette
- Extract to a directory
- Pass path with --data-path argument
"""

import numpy as np
import sys
import os
import argparse
from pathlib import Path
from time import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tensor import Tensor
from losses.cross_entropy import CrossEntropyLoss
from optim.sgd import SGD_Momentum
from models.alexnet import AlexNet
from datasets.imagenette import get_imagenette_loaders


class AlexNetTrainer:
    """Training loop for AlexNet."""
    
    def __init__(self, model, loss_fn, optimizer):
        """
        Initialize trainer.
        
        Args:
            model: AlexNet model
            loss_fn: Loss function
            optimizer: Optimizer
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average loss and accuracy for epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        start_time = time()
        
        for batch_idx, (x, y) in enumerate(train_loader()):
            # Forward pass
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += float(loss.data)
            predictions = np.argmax(logits.data, axis=1)
            total_correct += np.sum(predictions == y)
            total_samples += len(y)
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 20 == 0:
                elapsed = time() - start_time
                print(f"  Batch {batch_idx+1:3d}: Loss = {float(loss.data):.4f}, "
                      f"Acc = {100*total_correct/total_samples:.2f}%, "
                      f"Time = {elapsed:.1f}s")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, avg_acc
    
    def eval_epoch(self, val_loader):
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average loss and top-1 accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_correct_top5 = 0
        total_samples = 0
        num_batches = 0
        
        for x, y in val_loader():
            # Forward pass
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            
            # Accumulate metrics
            total_loss += float(loss.data)
            
            # Top-1 accuracy
            predictions = np.argmax(logits.data, axis=1)
            total_correct += np.sum(predictions == y)
            
            # Top-5 accuracy
            top5_preds = np.argsort(logits.data, axis=1)[:, -5:]
            for i, true_label in enumerate(y):
                if true_label in top5_preds[i]:
                    total_correct_top5 += 1
            
            total_samples += len(y)
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        top1_acc = total_correct / total_samples if total_samples > 0 else 0
        top5_acc = total_correct_top5 / total_samples if total_samples > 0 else 0
        
        return avg_loss, top1_acc, top5_acc
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate_schedule=None, save_dir='./models'):
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            learning_rate_schedule: Optional function (epoch) -> learning_rate
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nTraining {self.model} for {num_epochs} epochs")
        print("=" * 80)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Update learning rate
            if learning_rate_schedule is not None:
                new_lr = learning_rate_schedule(epoch)
                self.optimizer.set_learning_rate(new_lr)
                print(f"\nEpoch {epoch+1}/{num_epochs} (LR = {new_lr:.2e})")
            else:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_top1, val_top5 = self.eval_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_top1)
            
            # Print results
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc (Top-1): {100*train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc (Top-1): {100*val_top1:.2f}%, Val Acc (Top-5): {100*val_top5:.2f}%")
            
            # Save best model
            if val_top1 > best_val_acc:
                best_val_acc = val_top1
                print(f"  ✓ New best validation accuracy (Top-1): {100*best_val_acc:.2f}%")
                self.save_model(os.path.join(save_dir, 'best_alexnet.npz'))
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_model(os.path.join(save_dir, f'alexnet_epoch_{epoch+1}.npz'))
        
        print("\n" + "=" * 80)
        print(f"Training complete!")
        print(f"Best validation accuracy (Top-1): {100*best_val_acc:.2f}%")
    
    def save_model(self, filepath):
        """Save model parameters."""
        params = self.model.parameters()
        param_dict = {}
        for i, param in enumerate(params):
            param_dict[f'param_{i}'] = param.data
        np.savez(filepath, **param_dict)
        print(f"  Saved model to {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters."""
        data = np.load(filepath)
        params = self.model.parameters()
        for i, param in enumerate(params):
            if f'param_{i}' in data:
                param.data = data[f'param_{i}']
        print(f"  Loaded model from {filepath}")


# ============================================================================
# Learning Rate Scheduling
# ============================================================================

def exponential_schedule(epoch, initial_lr=0.01, decay_rate=0.9):
    """Exponential decay learning rate schedule."""
    return initial_lr * (decay_rate ** epoch)


def step_schedule(epoch, initial_lr=0.01, step_size=10, gamma=0.1):
    """Step learning rate schedule - decrease by gamma every step_size epochs."""
    return initial_lr * (gamma ** (epoch // step_size))


def cosine_schedule(epoch, num_epochs, initial_lr=0.01, min_lr=1e-4):
    """Cosine annealing learning rate schedule."""
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / num_epochs))


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AlexNet on ImageNette')
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to ImageNette dataset root directory')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay / L2 regularization (default: 1e-4)')
    parser.add_argument('--schedule', type=str, choices=['step', 'exponential', 'cosine'], default='step',
                        help='Learning rate schedule (default: step)')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes (default: 10 for ImageNette)')
    parser.add_argument('--save-dir', type=str, default='./models',
                        help='Directory to save models (default: ./models)')
    
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    """Main training script."""
    args = parse_args()
    
    print("AlexNet Training on ImageNette")
    print("=" * 80)
    print(f"Dataset: {args.data_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Momentum: {args.momentum}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"LR Schedule: {args.schedule}")
    print(f"Num Classes: {args.num_classes}")
    print("=" * 80)
    
    # Load data
    print("\nLoading ImageNette dataset...")
    try:
        train_loader, val_loader = get_imagenette_loaders(
            root=args.data_path,
            batch_size=args.batch_size,
            size=224
        )
    except Exception as e:
        print(f"Error loading ImageNette: {e}")
        print(f"Please check that the data path is correct: {args.data_path}")
        print("Download ImageNette from: https://github.com/fastai/imagenette")
        return
    
    # Create model
    model = AlexNet(num_classes=args.num_classes)
    print(f"\nModel: {model}")
    print(f"Total parameters: {sum(p.size for p in model.parameters()):,}")
    
    # Create loss and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = SGD_Momentum(
        model.parameters(),
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Create trainer
    trainer = AlexNetTrainer(model, loss_fn, optimizer)
    
    # Learning rate schedule
    if args.schedule == 'step':
        schedule = lambda epoch: step_schedule(epoch, args.learning_rate, step_size=10, gamma=0.1)
    elif args.schedule == 'exponential':
        schedule = lambda epoch: exponential_schedule(epoch, args.learning_rate, decay_rate=0.95)
    elif args.schedule == 'cosine':
        schedule = lambda epoch: cosine_schedule(epoch, args.epochs, args.learning_rate)
    else:
        schedule = None
    
    # Train
    try:
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            learning_rate_schedule=schedule,
            save_dir=args.save_dir
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
