"""Load the adopted CIFAR-10 DenseNet121 without changing its architecture."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import CIFAR10_MEAN, CIFAR10_STD, normalize_for_classifier
from compatibility.vendor.PyTorch_CIFAR10.cifar10_models.densenet import densenet121

UPSTREAM_COMMIT = "641cac24371b17052b9bb6e56af1c83b5e97cd7f"
MODEL_SOURCE = PROJECT_ROOT / "compatibility/vendor/PyTorch_CIFAR10/cifar10_models/densenet.py"
DEFAULT_WEIGHTS = MODEL_SOURCE.parent / "state_dicts/densenet121.pt"
EXPECTED_SHA256 = "8d5c7e8d0383f660de7201d1b29c9964c0f4393394e6066460bee04a68f01173"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_metadata(weights_path: str | Path | None = None) -> dict:
    path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "DenseNet121 weights are missing. Restore the existing checkpoint to "
            "compatibility/vendor/PyTorch_CIFAR10/cifar10_models/state_dicts/densenet121.pt "
            "or supply --weights. This module never downloads weights."
        )
    digest = file_sha256(path)
    if digest != EXPECTED_SHA256:
        raise ValueError("Checkpoint SHA256 does not match the adopted CIFAR-10 DenseNet121 weights")
    # Published evidence uses project-relative paths, not a user's machine path.
    display_path = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else path.name
    return {"file": display_path, "bytes": path.stat().st_size, "sha256": digest}


def load_model(weights_path: str | Path | None = None, device: str | torch.device = "cpu") -> nn.Module:
    """Return an eval-mode DenseNet121 with all parameters frozen.

    The adopted checkpoint is hash-verified and loaded with strict=True.
    The model expects normalized inputs. Use forward_pixels for raw [0,1]
    images. Freezing parameters does NOT disable gradients with respect to
    inputs; no no_grad/inference_mode context or detach is used here.
    """
    checkpoint_metadata(weights_path)
    path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS
    state = torch.load(path.expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, Tensor) for v in state.values()):
        raise TypeError("Expected a tensor-only state_dict")
    model = densenet121(pretrained=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict state_dict mismatch: {incompatible}")
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def forward_pixels(model: nn.Module, images: Tensor) -> Tensor:
    """Normalize raw RGB [B,3,32,32] pixels ONCE and return [B,10] logits.

    Reuses dataset.normalize_for_classifier without mutating or detaching
    images. Do not pass already-normalized classifier_train batches here.
    Future pixel-space modifications must happen before this call.
    """
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32) or images.shape[0] < 1:
        raise ValueError("Expected a nonempty [B,3,32,32] RGB batch")
    if not images.is_floating_point():
        raise TypeError("Expected floating-point pixels in [0,1]")
    if not bool(torch.isfinite(images).all()) or images.min().item() < 0 or images.max().item() > 1:
        raise ValueError("Pixel-space inputs must be finite and in [0,1]")
    return model(normalize_for_classifier(images, mean=CIFAR10_MEAN, std=CIFAR10_STD))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = load_model(args.weights, args.device)
    print(json.dumps({"checkpoint": checkpoint_metadata(args.weights), "strict_loading": "PASSED",
                      "eval_mode": not model.training,
                      "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
                      "parameter_count": sum(p.numel() for p in model.parameters())}, indent=2))


if __name__ == "__main__":
    main()
