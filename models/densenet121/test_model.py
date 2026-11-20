"""Check the independent loader and input autograd, then run full clean evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import CIFAR10_MEAN, CIFAR10_STD
from models.densenet121.load_model import MODEL_SOURCE, file_sha256, forward_pixels, load_model
from models.densenet121.evaluate import (
    LOGGER, build_parser, configure_logging, evaluate_model, evaluation_context,
    get_test_loader, save_report,
)


def batchnorm_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {f"{name}.{key}": value.detach().clone()
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for key, value in module.named_buffers(recurse=False)}


def run_model_checks(model: torch.nn.Module, pixels: torch.Tensor, labels: torch.Tensor) -> dict:
    assert CIFAR10_MEAN == (0.4914, 0.4822, 0.4465)
    assert CIFAR10_STD == (0.2471, 0.2435, 0.2616)
    assert pixels.shape == (128, 3, 32, 32)
    assert all(not module.training for module in model.modules())
    assert all(not p.requires_grad for p in model.parameters())
    before = batchnorm_snapshot(model)
    assert before, "No BatchNorm buffers found"
    outputs = []
    with torch.inference_mode():
        for batch_size in (1, 2, 128):
            logits = forward_pixels(model, pixels[:batch_size])
            assert logits.shape == (batch_size, 10) and bool(torch.isfinite(logits).all())
            outputs.append({"input": [batch_size, 3, 32, 32],
                            "output": list(logits.shape), "finite": True})
            LOGGER.info("Forward B=%d: %s -> %s PASSED", batch_size,
                        list(pixels[:batch_size].shape), list(logits.shape))
    # No inference_mode/no_grad here: only parameters are frozen, not inputs.
    inputs = pixels[:2].detach().clone().requires_grad_(True)
    original = inputs.detach().clone()
    loss = F.cross_entropy(forward_pixels(model, inputs), labels[:2])
    loss.backward()
    assert inputs.grad is not None and inputs.grad.shape == inputs.shape
    finite = bool(torch.isfinite(inputs.grad).all())
    nonzero = (inputs.grad.abs().flatten(1).sum(dim=1) > 0).tolist()
    assert finite and all(nonzero)
    assert torch.equal(inputs.detach(), original), "Input pixels were mutated"
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    after = batchnorm_snapshot(model)
    assert before.keys() == after.keys() and all(torch.equal(value, after[key]) for key, value in before.items())
    LOGGER.info("Frozen parameters, finite/nonzero input gradients, unchanged BatchNorm: PASSED")
    return {"strict_checkpoint_loading": "PASSED", "eval_mode": True,
            "all_parameters_frozen": True, "parameter_gradients_none": True,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "forward": outputs,
            "input_gradient": {"shape": list(inputs.grad.shape), "finite": finite,
                               "nonzero_per_image": nonzero,
                               "abs_max": inputs.grad.abs().max().item(),
                               "abs_mean": inputs.grad.abs().mean().item()},
            "input_pixels_unchanged": True, "batchnorm_buffers_unchanged": True,
            "batchnorm_buffer_count": len(before)}


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log)
    torch.set_num_threads(args.threads)
    torch.manual_seed(661)
    immutable = [PROJECT_ROOT / "dataset.py", PROJECT_ROOT / "test_dataset.py", MODEL_SOURCE]
    before_hashes = {str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in immutable}
    report = {"status": "RUNNING", "run_type": "model_checks_and_full_clean_evaluation"}
    try:
        report.update(evaluation_context(args.weights, args.device, args.batch_size, args.num_workers, args.threads))
        LOGGER.info("Server environment: %s", report["environment"])
        model = load_model(args.weights, args.device)
        LOGGER.info("Official checkpoint SHA256 and strict=True loading: PASSED")
        probe_loader = get_test_loader(args.data_dir, 128, args.num_workers)
        pixels, labels = next(iter(probe_loader))
        report["tests"] = run_model_checks(model, pixels.to(args.device), labels.to(args.device))
        buffers = batchnorm_snapshot(model)
        loader = probe_loader if args.batch_size == 128 else get_test_loader(args.data_dir, args.batch_size, args.num_workers)
        report["evaluation"] = evaluate_model(model, loader, args.device)
        after = batchnorm_snapshot(model)
        assert buffers.keys() == after.keys() and all(torch.equal(value, after[key]) for key, value in buffers.items())
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        report["tests"]["batchnorm_unchanged_after_full_evaluation"] = True
        assert all(file_sha256(PROJECT_ROOT / p) == digest for p, digest in before_hashes.items())
        assert file_sha256(args.weights) == report["checkpoint"]["sha256"]
        report["unchanged_existing_source_sha256"] = before_hashes
        report["checkpoint_unchanged_after_testing"] = True
        report["status"] = "PASSED"
        LOGGER.info("DenseNet121 independent tests + full 10000-image evaluation PASSED")
    except Exception as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        LOGGER.exception("DenseNet121 independent tests FAILED")
        raise
    finally:
        save_report(args.output, report)


if __name__ == "__main__":
    main()
