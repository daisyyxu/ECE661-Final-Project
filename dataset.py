"""CIFAR-10 datasets, DataLoaders, and classifier normalization utilities.

Typical reuse from a training script::

    from dataset import create_dataloaders, normalize_for_classifier

    loaders = create_dataloaders(batch_size=128)
    images, labels = next(iter(loaders.classifier_train))
    logits = model(images)  # This loader has already normalized its images.

The patch and test loaders deliberately return pixels in [0, 1]. Apply a patch
in that space, clamp if appropriate, and only then call
``normalize_for_classifier``. The defaults match the adopted
huyvnphan/PyTorch_CIFAR10 checkpoints. Other checkpoints may require different
statistics; pass their mean and standard deviation explicitly in that case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class CIFAR10Loaders:
    """The three loaders used by the classifier and adversarial-patch stages."""

    classifier_train: DataLoader
    patch_optimization: DataLoader
    test_evaluation: DataLoader


def normalize_for_classifier(
    images: Tensor,
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
) -> Tensor:
    """Differentiably normalize an NCHW or CHW tensor without modifying it.

    Keep adversarial-patch composition in [0, 1] pixel space and call this
    function immediately before the classifier. Gradients flow through all
    operations. Override ``mean`` and ``std`` when a pretrained checkpoint was
    trained with different preprocessing.
    """

    if images.ndim not in (3, 4) or images.shape[-3] != 3:
        raise ValueError(
            f"Expected CHW or NCHW RGB tensor, received shape {tuple(images.shape)}"
        )
    channel_shape = (3, 1, 1) if images.ndim == 3 else (1, 3, 1, 1)
    mean_tensor = images.new_tensor(mean).view(channel_shape)
    std_tensor = images.new_tensor(std).view(channel_shape)
    return (images - mean_tensor) / std_tensor


def create_dataloaders(
    data_dir: str | Path = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = True,
    pin_memory: bool | None = None,
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
) -> CIFAR10Loaders:
    """Create all CIFAR-10 DataLoaders, downloading torchvision archives as needed.

    ``num_workers=0`` is portable across notebooks, macOS, and Windows. Training
    scripts can increase it after adding the usual ``if __name__ == '__main__'``
    guard. The patch loader uses the training split with no augmentation; the
    evaluation loader uses the test split and never shuffles.

    ``mean`` and ``std`` configure only ``classifier_train``. Patch and test
    images always remain unnormalized in [0, 1]. If overriding these statistics,
    pass the same values to ``normalize_for_classifier`` before inference.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    classifier_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    pixel_transform = transforms.ToTensor()

    classifier_train = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=classifier_transform
    )
    patch_train = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=pixel_transform
    )
    test = datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=pixel_transform
    )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    return CIFAR10Loaders(
        classifier_train=DataLoader(classifier_train, shuffle=True, **common),
        patch_optimization=DataLoader(patch_train, shuffle=True, **common),
        test_evaluation=DataLoader(test, shuffle=False, **common),
    )


__all__ = [
    "CIFAR10_CLASSES",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "CIFAR10Loaders",
    "create_dataloaders",
    "normalize_for_classifier",
]
