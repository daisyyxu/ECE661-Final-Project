"""Download and validate the ECE661 CIFAR-10 input pipeline.

Run: ``python test_dataset.py --batch-size 128``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms

from dataset import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    create_dataloaders,
    normalize_for_classifier,
)


def _validate_batch(name: str, images: torch.Tensor, labels: torch.Tensor) -> None:
    assert images.ndim == 4 and images.shape[1:] == (3, 32, 32), (
        f"{name}: bad image shape {tuple(images.shape)}"
    )
    assert labels.ndim == 1 and labels.shape[0] == images.shape[0]
    assert labels.dtype == torch.int64
    assert int(labels.min()) >= 0 and int(labels.max()) < 10


def _save_grid(images: torch.Tensor, labels: torch.Tensor, output: Path) -> None:
    count = min(16, images.shape[0])
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < count:
            axis.imshow(images[index].permute(1, 2, 0).cpu().numpy())
            axis.set_title(CIFAR10_CLASSES[int(labels[index])], fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _check_normalization(images: torch.Tensor) -> None:
    assert CIFAR10_MEAN == (0.4914, 0.4822, 0.4465)
    assert CIFAR10_STD == (0.2471, 0.2435, 0.2616)
    for sample in (images[:2], images[0]):
        original = sample.clone()
        for mean, std in (
            (CIFAR10_MEAN, CIFAR10_STD),
            ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)),
        ):
            probe = sample.clone().requires_grad_(True)
            shape = (1, 3, 1, 1) if probe.ndim == 4 else (3, 1, 1)
            m, s = probe.new_tensor(mean).view(shape), probe.new_tensor(std).view(shape)
            actual = normalize_for_classifier(probe, mean=mean, std=std)
            assert torch.allclose(actual, (probe - m) / s)
            actual.sum().backward()
            assert probe.grad is not None and torch.isfinite(probe.grad).all()
            assert torch.allclose(probe.grad, (1 / s).expand_as(probe))
            assert torch.equal(probe.detach(), original), "Normalization mutated input"
    for shape in ((32, 32), (2, 1, 32, 32), (1, 2, 3, 32, 32)):
        try:
            normalize_for_classifier(torch.zeros(shape))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid shape accepted: {shape}")


def validate(data_dir: Path, batch_size: int, num_workers: int, output: Path,
             download: bool = True) -> None:
    torch.manual_seed(661)
    loaders = create_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers,
        download=download
    )
    counts = {
        "classifier_train": len(loaders.classifier_train.dataset),
        "patch_optimization": len(loaders.patch_optimization.dataset),
        "test_evaluation": len(loaders.test_evaluation.dataset),
    }
    assert counts == {
        "classifier_train": 50_000,
        "patch_optimization": 50_000,
        "test_evaluation": 10_000,
    }, f"Unexpected dataset counts: {counts}"
    assert isinstance(loaders.test_evaluation.sampler, SequentialSampler)
    assert isinstance(loaders.classifier_train.sampler, RandomSampler)
    assert isinstance(loaders.patch_optimization.sampler, RandomSampler)
    assert all(not loader.drop_last and loader.batch_size == batch_size for loader in (
        loaders.classifier_train, loaders.patch_optimization, loaders.test_evaluation))
    pipeline = loaders.classifier_train.dataset.transform.transforms
    assert [type(t) for t in pipeline] == [transforms.RandomCrop,
        transforms.RandomHorizontalFlip, transforms.ToTensor, transforms.Normalize]
    assert pipeline[0].size == (32, 32) and pipeline[0].padding == 4
    assert pipeline[1].p == 0.5
    assert tuple(pipeline[-1].mean) == CIFAR10_MEAN
    assert tuple(pipeline[-1].std) == CIFAR10_STD
    for loader in (loaders.patch_optimization, loaders.test_evaluation):
        assert isinstance(loader.dataset.transform, transforms.ToTensor)
    for name, loader in (("classifier_train", loaders.classifier_train),
                         ("patch_optimization", loaders.patch_optimization),
                         ("test_evaluation", loaders.test_evaluation)):
        targets = torch.tensor(loader.dataset.targets)
        assert targets.dtype == torch.int64
        assert torch.equal(torch.bincount(targets, minlength=10),
                           torch.full((10,), counts[name] // 10))

    classifier_images, classifier_labels = next(iter(loaders.classifier_train))
    patch_images, patch_labels = next(iter(loaders.patch_optimization))
    test_images, test_labels = next(iter(loaders.test_evaluation))
    for name, images, labels in (
        ("classifier_train", classifier_images, classifier_labels),
        ("patch_optimization", patch_images, patch_labels),
        ("test_evaluation", test_images, test_labels),
    ):
        _validate_batch(name, images, labels)
        assert images.shape[0] == batch_size and images.dtype == torch.float32

    for name, images in (("patch", patch_images), ("test", test_images)):
        assert float(images.min()) >= 0.0 and float(images.max()) <= 1.0, (
            f"{name}: pixels must remain in [0, 1]"
        )

    mean = classifier_images.new_tensor(CIFAR10_MEAN)[None, :, None, None]
    std = classifier_images.new_tensor(CIFAR10_STD)[None, :, None, None]
    lower = -mean / std
    upper = (1 - mean) / std
    assert bool(torch.all(classifier_images >= lower - 1e-6))
    assert bool(torch.all(classifier_images <= upper + 1e-6))

    probe = patch_images.detach().clone().requires_grad_(True)
    normalize_for_classifier(probe).sum().backward()
    assert probe.grad is not None and torch.isfinite(probe.grad).all()

    expected = (patch_images - patch_images.new_tensor(CIFAR10_MEAN)[None, :, None, None])
    expected = expected / patch_images.new_tensor(CIFAR10_STD)[None, :, None, None]
    assert torch.allclose(normalize_for_classifier(patch_images), expected)
    _check_normalization(patch_images)

    repeated_images, repeated_labels = next(iter(loaders.test_evaluation))
    assert torch.equal(test_images, repeated_images) and torch.equal(test_labels, repeated_labels)
    scan = {}
    for name, loader in (("patch_optimization", loaders.patch_optimization),
                         ("test_evaluation", loaders.test_evaluation)):
        total, low, high = 0, 1.0, 0.0
        for images, labels in loader:
            _validate_batch(name, images, labels)
            assert torch.isfinite(images).all()
            low, high = min(low, images.min().item()), max(high, images.max().item())
            total += len(labels)
        assert total == counts[name] and 0 <= low <= high <= 1
        scan[name] = {"total": total, "range": [low, high]}

    custom_mean, custom_std = (0.1, 0.2, 0.3), (0.4, 0.5, 0.6)
    custom = create_dataloaders(data_dir=data_dir, batch_size=batch_size,
        num_workers=0, download=download, mean=custom_mean, std=custom_std)
    custom_transform = custom.classifier_train.dataset.transform.transforms[-1]
    assert tuple(custom_transform.mean) == custom_mean and tuple(custom_transform.std) == custom_std
    probe_pixel = torch.ones(3, 32, 32)
    assert torch.allclose(custom_transform(probe_pixel),
                          normalize_for_classifier(probe_pixel, mean=custom_mean, std=custom_std))
    for loader in (custom.patch_optimization, custom.test_evaluation):
        assert isinstance(loader.dataset.transform, transforms.ToTensor)
        image, _ = loader.dataset[0]
        assert 0 <= image.min() and image.max() <= 1
    assert torch.equal(custom.test_evaluation.dataset[0][0], test_images[0])
    try:
        create_dataloaders(data_dir=data_dir, batch_size=0, download=False)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid batch size accepted")

    _save_grid(test_images, test_labels, output)
    print("CIFAR-10 validation PASSED")
    print(f"Counts: {counts}")
    print(f"Classifier batch: shape={tuple(classifier_images.shape)}, dtype={classifier_images.dtype}")
    print(
        "Patch batch: "
        f"shape={tuple(patch_images.shape)}, range=[{patch_images.min().item():.6f}, "
        f"{patch_images.max().item():.6f}]"
    )
    print(
        "Test batch: "
        f"shape={tuple(test_images.shape)}, range=[{test_images.min().item():.6f}, "
        f"{test_images.max().item():.6f}], shuffle=False"
    )
    print("Differentiable normalization: PASSED")
    print("Default/custom normalization, CHW/NCHW gradients, no input mutation: PASSED")
    print("Transforms, class balance, sampler, deterministic evaluation: PASSED")
    print(f"Full pixel-space scans: {scan}")
    print(f"Sample grid: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-download", action="store_true",
                        help="Require existing CIFAR-10 files; never download")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/cifar10_samples.png")
    )
    args = parser.parse_args()
    validate(args.data_dir, args.batch_size, args.num_workers, args.output,
             download=not args.no_download)
