"""
Training Script for LeNet-5 on MNIST

This script demonstrates the complete training pipeline:
1. Load dataset
2. Create model
3. Define loss and optimizer
4. Training loop with validation
5. Model saving

Expected results:
- After 10 epochs: >95% accuracy
- After 20 epochs: >98% accuracy

Usage:
python train_lenet.py
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tensor import Tensor
from losses.cross_entropy import CrossEntropyLoss
from optim.sgd import SGD_Momentum
from models.lenet import LeNet5
from datasets.mnist import get_mnist_loaders


class Trainer:
    """Training loop manager."""
    
    def __init__(self, model, loss_fn, optimizer, device='cpu'):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device ('cpu' or 'cuda')
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss and accuracy for epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
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
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}: Loss = {float(loss.data):.4f}, "
                      f"Acc = {100*total_correct/total_samples:.2f}%")
        
        avg_loss = total_loss / (batch_idx + 1)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def eval_epoch(self, val_loader):
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for x, y in val_loader():
            # Forward pass (no gradients needed)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            
            # Accumulate metrics
            total_loss += float(loss.data)
            predictions = np.argmax(logits.data, axis=1)
            total_correct += np.sum(predictions == y)
            total_samples += len(y)
        
        avg_loss = total_loss / (batch_idx + 1) if 'batch_idx' in locals() else 0
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate_schedule=None):
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            learning_rate_schedule: Optional function of (epoch) -> learning_rate
        """
        print(f"\nTraining {self.model} for {num_epochs} epochs")
        print("=" * 70)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Update learning rate if schedule provided
            if learning_rate_schedule is not None:
                new_lr = learning_rate_schedule(epoch)
                self.optimizer.set_learning_rate(new_lr)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.eval_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Print epoch results
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {100*train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {100*val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  ✓ New best validation accuracy: {100*best_val_acc:.2f}%")
                self.save_model('best_model.npz')
        
        print("\n" + "=" * 70)
        print(f"Training complete!")
        print(f"Best validation accuracy: {100*best_val_acc:.2f}%")
    
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
            param.data = data[f'param_{i}']
        print(f"  Loaded model from {filepath}")


# ============================================================================
# Learning Rate Scheduling
# ============================================================================

def step_schedule(epoch, initial_lr=0.01, step_size=5, gamma=0.1):
    """
    Step learning rate schedule.
    
    Decreases learning rate by gamma every step_size epochs.
    """
    return initial_lr * (gamma ** (epoch // step_size))


# ============================================================================
# Main
# ============================================================================

def main():
    """Main training script."""
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 20
    initial_learning_rate = 0.01
    momentum = 0.9
    
    print("LeNet-5 Training on MNIST")
    print("=" * 70)
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {initial_learning_rate}")
    print(f"Momentum: {momentum}")
    print("=" * 70)
    
    # Load data
    print("\nLoading MNIST dataset...")
    try:
        train_loader, val_loader = get_mnist_loaders(
            batch_size=batch_size,
            root='./data/mnist'
        )
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        print("Note: You may need to download MNIST first")
        return
    
    # Create model
    model = LeNet5(num_classes=10)
    print(f"\nModel: {model}")
    
    # Create loss and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = SGD_Momentum(
        model.parameters(),
        learning_rate=initial_learning_rate,
        momentum=momentum,
        weight_decay=1e-4
    )
    
    # Create trainer
    trainer = Trainer(model, loss_fn, optimizer)
    
    # Learning rate schedule: divide by 10 after 10 epochs
    def schedule(epoch):
        return step_schedule(epoch, initial_learning_rate, step_size=10, gamma=0.1)
    
    # Train
    try:
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            learning_rate_schedule=schedule
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
