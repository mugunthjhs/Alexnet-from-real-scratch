"""
ImageNette Dataset Loader

ImageNette is a subset of ImageNet with 10 classes:
- tench (fish)
- English springer (dog)
- cassette player
- chain saw
- church (building)
- French horn (instrument)
- garbage truck
- gas pump
- golf ball
- parachute

Images are 224×224 RGB.

This is what we'll use to train AlexNet since full ImageNet is huge.

Dataset: https://github.com/fastai/imagenette
"""

import numpy as np
import os
from PIL import Image
from tensor import Tensor


class ImageNetteDataset:
    """ImageNette dataset loader."""
    
    # Class names
    CLASS_NAMES = [
        'tench', 'english_springer', 'cassette_player', 'chain_saw', 'church',
        'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute'
    ]
    
    # Class IDs (ImageNet IDs)
    CLASS_IDS = [
        'n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
        'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257'
    ]
    
    def __init__(self, root: str, split: str = 'train', size: int = 224):
        """
        Initialize ImageNette dataset.
        
        Args:
            root: Path to dataset root (e.g., '/path/to/imagenette2')
            split: 'train' or 'val'
            size: Image size (224 for AlexNet)
        """
        self.root = root
        self.split = split
        self.size = size
        
        # Build list of image paths and labels
        self.images = []
        self.labels = []
        
        split_dir = os.path.join(root, split)
        
        for class_idx, class_id in enumerate(self.CLASS_IDS):
            class_dir = os.path.join(split_dir, class_id)
            
            if not os.path.isdir(class_dir):
                print(f"Warning: {class_dir} not found")
                continue
            
            # Find all images in this class directory
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(class_dir, filename)
                    self.images.append(filepath)
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images from {split}")
    
    def __len__(self) -> int:
        """Number of samples."""
        return len(self.images)
    
    def __getitem__(self, idx: int):
        """Get single sample."""
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load {image_path}: {e}")
            # Return dummy image if loading fails
            image = Image.new('RGB', (self.size, self.size))
        
        # Resize to (size, size)
        image = image.resize((self.size, self.size), Image.BILINEAR)
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Convert from HxWxC to CxHxW
        image_array = np.transpose(image_array, (2, 0, 1))
        
        # Normalize by ImageNet statistics (optional, can skip for simplicity)
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image_array = (image_array - mean[:, None, None]) / std[:, None, None]
        
        return Tensor(image_array), label
    
    def get_batch(self, indices: np.ndarray):
        """Get batch of samples."""
        images = []
        labels = []
        
        for idx in indices:
            try:
                img, lbl = self[idx]
                images.append(img.data)
                labels.append(lbl)
            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                continue
        
        if len(images) == 0:
            raise RuntimeError("No valid samples in batch")
        
        images = np.stack(images, axis=0)  # (batch, 3, 224, 224)
        labels = np.array(labels, dtype=np.int64)
        
        return Tensor(images), labels
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ImageNetteDataset(split='{self.split}', size={len(self)})"


def get_imagenette_loaders(root: str, batch_size: int = 32, size: int = 224):
    """
    Get ImageNette training and validation data loaders.
    
    Args:
        root: Path to ImageNette dataset root
        batch_size: Batch size
        size: Image size
        
    Returns:
        train_loader, val_loader (functions that yield batches)
    """
    train_dataset = ImageNetteDataset(root=root, split='train', size=size)
    val_dataset = ImageNetteDataset(root=root, split='val', size=size)
    
    def train_loader():
        indices = np.arange(len(train_dataset))
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            try:
                yield train_dataset.get_batch(batch_indices)
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                continue
    
    def val_loader():
        indices = np.arange(len(val_dataset))
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            try:
                yield val_dataset.get_batch(batch_indices)
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                continue
    
    return train_loader, val_loader


def data_augmentation_train(x: np.ndarray) -> np.ndarray:
    """
    Apply data augmentation for training.
    
    Techniques:
    - Random crop: 224 from larger image
    - Horizontal flip: 50% chance
    - Brightness/contrast adjustment (optional)
    
    Args:
        x: Input image (CxHxW)
        
    Returns:
        Augmented image
    """
    # For now, simple augmentation
    # Horizontal flip with 50% probability
    if np.random.rand() > 0.5:
        x = x[:, :, ::-1]  # Flip horizontally
    
    return x


if __name__ == "__main__":
    # Test dataset loading
    print("Testing ImageNette Dataset...")
    
    # This requires the dataset to be downloaded
    try:
        dataset = ImageNetteDataset('/path/to/imagenette2', split='train')
        print(f"Dataset: {dataset}")
        
        if len(dataset) > 0:
            img, label = dataset[0]
            print(f"Image shape: {img.shape}, Label: {label}")
            print(f"Image dtype: {img.data.dtype}")
            print(f"Image range: [{img.data.min():.3f}, {img.data.max():.3f}]")
    except Exception as e:
        print(f"Dataset not available: {e}")
        print("Please download ImageNette from: https://github.com/fastai/imagenette")
