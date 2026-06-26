"""
ImageNette Dataset — auto-download + lazy loader.

10-class subset of ImageNet (fast.ai):
  tench, english_springer, cassette_player, chain_saw, church,
  french_horn, garbage_truck, gas_pump, golf_ball, parachute

Usage (mirrors PyTorch MNIST style):

    from datasets import ImageNetteDataset, ImageNetteLoader

    train_ds = ImageNetteDataset(root='./data', split='train', download=True)
    val_ds   = ImageNetteDataset(root='./data', split='val',   download=True)

    train_loader = ImageNetteLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = ImageNetteLoader(val_ds,   batch_size=32, shuffle=False)

    for X_batch, y_batch in train_loader:   # X: Tensor (B,3,H,W)  y: ndarray (B,)
        ...
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tensor import Tensor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_IDS = [
    'n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
    'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257',
]

CLASS_NAMES = [
    'tench', 'english_springer', 'cassette_player', 'chain_saw', 'church',
    'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute',
]

_URLS = {
    160: 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',
    320: 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz',
    'full': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz',
}

_FOLDER_NAMES = {
    160: 'imagenette2-160',
    320: 'imagenette2-320',
    'full': 'imagenette2',
}

# ImageNet channel mean / std (used when normalize=True)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb  = downloaded / 1_048_576
        print(f'\r  {pct:5.1f}%  ({mb:.1f} MB)', end='', flush=True)


def _download_and_extract(url: str, dest_dir: str, archive_name: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    archive_path = os.path.join(dest_dir, archive_name)

    if not os.path.exists(archive_path):
        print(f'Downloading {url}')
        urllib.request.urlretrieve(url, archive_path, reporthook=_progress_hook)
        print()  # newline after progress bar

    print(f'Extracting {archive_name} ...')
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(dest_dir)
    print('Done.')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageNetteDataset:
    """
    Lazy-loading ImageNette dataset.

    Parameters
    ----------
    root      : directory where the dataset is stored / will be downloaded
    split     : 'train' or 'val'
    size      : download variant — 160 (default, ~94 MB), 320, or 'full'
    image_size: resize target in pixels (default matches `size`)
    download  : if True, fetch + extract the archive when not already present
    normalize : if True, apply ImageNet mean/std normalisation
    augment   : if True, apply random horizontal flip during __getitem__
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        size: int = 160,
        image_size: int = None,
        download: bool = False,
        normalize: bool = True,
        augment: bool = False,
    ):
        if split not in ('train', 'val'):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")
        if size not in _URLS:
            raise ValueError(f"size must be 160, 320, or 'full', got {size!r}")

        self.root       = root
        self.split      = split
        self.size       = size
        self.image_size = image_size if image_size is not None else (size if isinstance(size, int) else 224)
        self.normalize  = normalize
        self.augment    = augment and (split == 'train')

        dataset_dir = os.path.join(root, _FOLDER_NAMES[size])

        if not os.path.isdir(dataset_dir):
            if download:
                archive = _FOLDER_NAMES[size] + '.tgz'
                _download_and_extract(_URLS[size], root, archive)
            else:
                raise RuntimeError(
                    f"Dataset not found at '{dataset_dir}'.\n"
                    f"Pass download=True to fetch it automatically."
                )

        self._images: list[str] = []
        self._labels: list[int] = []

        split_dir = os.path.join(dataset_dir, split)
        for class_idx, class_id in enumerate(CLASS_IDS):
            class_dir = os.path.join(split_dir, class_id)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self._images.append(os.path.join(class_dir, fname))
                    self._labels.append(class_idx)

        print(f"ImageNetteDataset [{split}]: {len(self._images)} images, "
              f"size={self.image_size}px")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        """Returns (Tensor[3,H,W] float32, int label)."""
        path  = self._images[idx]
        label = self._labels[idx]

        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (self.image_size, self.image_size))

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
        arr = arr.transpose(2, 0, 1)                     # (3, H, W)

        if self.normalize:
            arr = (arr - _MEAN[:, None, None]) / _STD[:, None, None]

        if self.augment and np.random.rand() > 0.5:
            arr = arr[:, :, ::-1].copy()

        return Tensor(arr), label

    def __repr__(self) -> str:
        return (
            f"ImageNetteDataset(split='{self.split}', n={len(self)}, "
            f"image_size={self.image_size}, normalize={self.normalize})"
        )


# ---------------------------------------------------------------------------
# Loader  (lazy — reads images from disk per batch)
# ---------------------------------------------------------------------------

class ImageNetteLoader:
    """
    Iterates over an ImageNetteDataset in mini-batches.

    Yields (Tensor[B,3,H,W], ndarray[B] int64) pairs each iteration.

    Parameters
    ----------
    dataset    : ImageNetteDataset instance
    batch_size : samples per batch (default 32)
    shuffle    : shuffle order each epoch (default True for train)
    drop_last  : drop final incomplete batch (default False)
    """

    def __init__(
        self,
        dataset: ImageNetteDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            indices = Tensor.rand(n).data.argsort().astype(np.int64)
        else:
            indices = Tensor.arange(n, dtype=np.int64).data

        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            if self.drop_last and end > len(indices):
                break
            batch_idx = indices[start:end]

            imgs, labels = [], []
            for i in batch_idx:
                img, lbl = self.dataset[int(i)]
                imgs.append(img.data)
                labels.append(lbl)

            yield (
                Tensor(np.stack(imgs, axis=0)),
                np.array(labels, dtype=np.int64),
            )

    def __repr__(self) -> str:
        return (
            f"ImageNetteLoader(n={len(self.dataset)}, "
            f"batch_size={self.batch_size}, shuffle={self.shuffle}, "
            f"drop_last={self.drop_last})"
        )


# ---------------------------------------------------------------------------
# Convenience function  (mirrors get_imagenette_loaders from legacy code)
# ---------------------------------------------------------------------------

def get_imagenette_loaders(
    root: str,
    batch_size: int = 32,
    size: int = 160,
    image_size: int = None,
    download: bool = False,
    normalize: bool = True,
):
    """
    One-liner to get train + val loaders.

    Returns
    -------
    train_loader, val_loader : ImageNetteLoader instances
    """
    train_ds = ImageNetteDataset(
        root, split='train', size=size, image_size=image_size,
        download=download, normalize=normalize, augment=True,
    )
    val_ds = ImageNetteDataset(
        root, split='val', size=size, image_size=image_size,
        download=download, normalize=normalize, augment=False,
    )
    return (
        ImageNetteLoader(train_ds, batch_size=batch_size, shuffle=True),
        ImageNetteLoader(val_ds,   batch_size=batch_size, shuffle=False),
    )
