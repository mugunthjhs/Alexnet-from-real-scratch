"""
MNIST Dataset Loader

Downloads and loads the MNIST dataset for training LeNet-5.

MNIST: 28×28 grayscale images of handwritten digits (0-9)
- Training: 60,000 images
- Testing: 10,000 images

This is a simple test dataset - real training uses ImageNette.
"""

import numpy as np
import os
from urllib.request import urlretrieve
import gzip
import struct
from tensor import Tensor


class MNISTDataset:
    """MNIST dataset loader."""
    
    def __init__(self, root: str = './data', train: bool = True, download: bool = True):
        """
        Initialize MNIST dataset.
        
        Args:
            root: Path to data directory
            train: Load training (True) or test (False) set
            download: Download if not present
        """
        self.root = root
        self.train = train
        
        if download and not self._check_exists():
            self.download()
        
        if self.train:
            self.data, self.targets = self._load_training_data()
        else:
            self.data, self.targets = self._load_test_data()
    
    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        return (os.path.exists(os.path.join(self.root, 'train-images-idx3-ubyte.gz')) and
                os.path.exists(os.path.join(self.root, 'train-labels-idx1-ubyte.gz')) and
                os.path.exists(os.path.join(self.root, 't10k-images-idx3-ubyte.gz')) and
                os.path.exists(os.path.join(self.root, 't10k-labels-idx1-ubyte.gz')))
    
    def download(self):
        """Download MNIST dataset."""
        os.makedirs(self.root, exist_ok=True)
        
        urls = [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        ]
        
        for url in urls:
            filename = url.split('/')[-1]
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                urlretrieve(url, filepath)
    
    def _load_training_data(self):
        """Load training images and labels."""
        return self._load_data(
            os.path.join(self.root, 'train-images-idx3-ubyte.gz'),
            os.path.join(self.root, 'train-labels-idx1-ubyte.gz')
        )
    
    def _load_test_data(self):
        """Load test images and labels."""
        return self._load_data(
            os.path.join(self.root, 't10k-images-idx3-ubyte.gz'),
            os.path.join(self.root, 't10k-labels-idx1-ubyte.gz')
        )
    
    def _load_data(self, images_path: str, labels_path: str):
        """Load images and labels from files."""
        # Load images
        with gzip.open(images_path, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            num_images = struct.unpack('>I', f.read(4))[0]
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, num_rows, num_cols)
        
        # Load labels
        with gzip.open(labels_path, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            num_labels = struct.unpack('>I', f.read(4))[0]
            
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return images, labels
    
    def __len__(self) -> int:
        """Number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """Get single sample."""
        image = self.data[idx].astype(np.float32) / 255.0  # Normalize to [0, 1]
        label = int(self.targets[idx])
        
        # Add channel dimension: (28, 28) -> (1, 28, 28)
        image = np.expand_dims(image, axis=0)
        
        return Tensor(image), label
    
    def get_batch(self, indices: np.ndarray):
        """Get batch of samples."""
        images = []
        labels = []
        
        for idx in indices:
            img, lbl = self[idx]
            images.append(img.data)
            labels.append(lbl)
        
        images = np.stack(images, axis=0)  # (batch, 1, 28, 28)
        labels = np.array(labels, dtype=np.int64)
        
        return Tensor(images), labels
    
    def __repr__(self) -> str:
        """String representation."""
        split = "train" if self.train else "test"
        return f"MNISTDataset(split='{split}', size={len(self)})"


def get_mnist_loaders(batch_size: int = 32, root: str = './data'):
    """
    Get MNIST training and test data loaders.
    
    Args:
        batch_size: Batch size
        root: Data directory
        
    Returns:
        train_loader, test_loader (functions that yield batches)
    """
    train_dataset = MNISTDataset(root=root, train=True)
    test_dataset = MNISTDataset(root=root, train=False)
    
    def train_loader():
        indices = np.arange(len(train_dataset))
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield train_dataset.get_batch(batch_indices)
    
    def test_loader():
        indices = np.arange(len(test_dataset))
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield test_dataset.get_batch(batch_indices)
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing MNIST Dataset...")
    
    dataset = MNISTDataset(train=True, download=False)
    print(f"Dataset: {dataset}")
    print(f"Sample shape: {dataset[0][0].shape}")
    
    img, label = dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")
    
    print("✓ MNIST dataset works!")
