"""Offline clean evaluation on the complete CIFAR-10 test split."""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, SequentialSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import CIFAR10_MEAN, CIFAR10_STD, create_dataloaders
from models.densenet121.load_model import (
    DEFAULT_WEIGHTS, MODEL_SOURCE, UPSTREAM_COMMIT, checkpoint_metadata,
    file_sha256, forward_pixels, load_model,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LOGGER = logging.getLogger("member3.densenet121")


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in LOGGER.handlers[:]:
        handler.close()
        LOGGER.removeHandler(handler)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w")):
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(handler)


def get_test_loader(data_dir: str | Path, batch_size: int = 128, num_workers: int = 0) -> DataLoader:
    """Use the existing pipeline; never download, augment or normalize twice."""
    loader = create_dataloaders(data_dir=data_dir, batch_size=batch_size,
                               num_workers=num_workers, download=False, pin_memory=False).test_evaluation
    if len(loader.dataset) != 10_000 or loader.drop_last or not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("Expected all 10000 CIFAR-10 test images, sequentially, without drop_last")
    return loader


def evaluation_context(weights_path: str | Path, device: str, batch_size: int,
                       num_workers: int, threads: int) -> dict:
    executable = Path(sys.executable).resolve()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "DenseNet121 (CIFAR-10)",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "platform": platform.platform(),
                        "device": device, "batch_size": batch_size, "num_workers": num_workers,
                        "threads": threads, "interpreter_name": executable.name},
        "source": {"url": "https://github.com/huyvnphan/PyTorch_CIFAR10",
                   "upstream_commit": UPSTREAM_COMMIT,
                   "model_source_sha256": file_sha256(MODEL_SOURCE),
                   "dataset_py_sha256": file_sha256(PROJECT_ROOT / "dataset.py"),
                   "member3_code_sha256": {p.name: file_sha256(p) for p in
                       sorted(Path(__file__).parent.glob("*.py"))}},
        "checkpoint": checkpoint_metadata(weights_path),
        "normalization": {"mean": list(CIFAR10_MEAN), "std": list(CIFAR10_STD),
                          "input_pixel_space": "[0,1]", "implementation": "dataset.normalize_for_classifier"},
        "downloads_enabled": False,
    }


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: str = "cpu") -> dict:
    """Compute clean accuracy, counting every example exactly once.

    inference_mode is limited to this clean-evaluation function. The reusable
    load_model/forward_pixels functions deliberately keep input autograd enabled.
    """
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Evaluation requires an eval-mode model with frozen parameters")
    if len(loader.dataset) != 10_000 or loader.drop_last or not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("Expected the complete sequential CIFAR-10 test loader")
    correct = total = 0
    low, high = 1.0, 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (images, labels) in enumerate(loader):
            if labels.dtype != torch.int64 or labels.ndim != 1 or labels.min() < 0 or labels.max() >= 10:
                raise ValueError("Invalid CIFAR-10 labels")
            low, high = min(low, images.min().item()), max(high, images.max().item())
            images, labels = images.to(device), labels.to(device)
            logits = forward_pixels(model, images)
            if logits.shape != (len(labels), 10) or not bool(torch.isfinite(logits).all()):
                raise ValueError("Invalid/nonfinite model output")
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += len(labels)
            if step % 20 == 0:
                LOGGER.info("Clean evaluation: %d/10000 images, accuracy %.2f%%", total, 100 * correct / total)
    if total != 10_000:
        raise AssertionError(f"Incomplete evaluation: {total}/10000")
    result = {"correct": correct, "total": total, "clean_accuracy_percent": 100 * correct / total,
              "pixel_range": [low, high], "finite_outputs": True, "shuffle": False,
              "seconds": time.perf_counter() - started}
    LOGGER.info("Clean accuracy: %d/%d = %.2f%%", correct, total, result["clean_accuracy_percent"])
    return result


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "evaluation.json")
    parser.add_argument("--log", type=Path, default=RESULTS_DIR / "test.log")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log)
    torch.set_num_threads(args.threads)
    torch.manual_seed(661)
    report = {"status": "RUNNING", "run_type": "clean_evaluation_only"}
    try:
        report.update(evaluation_context(args.weights, args.device, args.batch_size, args.num_workers, args.threads))
        model = load_model(args.weights, args.device)
        report["evaluation"] = evaluate_model(model, get_test_loader(args.data_dir, args.batch_size, args.num_workers), args.device)
        report["status"] = "PASSED"
        LOGGER.info("DenseNet121 clean evaluation PASSED")
    except Exception as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        LOGGER.exception("DenseNet121 evaluation FAILED")
        raise
    finally:
        save_report(args.output, report)


if __name__ == "__main__":
    main()
