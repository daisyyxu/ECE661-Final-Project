"""Reproducible, inference-only checks using the project's existing dataset.py."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).parent / "vendor/PyTorch_CIFAR10")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(661)
    sys.path.insert(0, str(args.repo.resolve()))
    spec = importlib.util.spec_from_file_location("ece661_dataset", args.project / "dataset.py")
    dataset = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dataset
    spec.loader.exec_module(dataset)
    from cifar10_models.resnet import resnet18
    from cifar10_models.vgg import vgg16_bn
    from cifar10_models.densenet import densenet121

    upstream_mean = (0.4914, 0.4822, 0.4465)
    upstream_std = (0.2471, 0.2435, 0.2616)
    baseline_hash = sha256(args.project / "dataset.py")
    report = {
        "environment": {
            "python": sys.version, "executable": sys.executable,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "platform": platform.platform(), "device": args.device, "threads": args.threads,
            "batch_size": args.batch_size,
        },
        "repository": {
            "url": "https://github.com/huyvnphan/PyTorch_CIFAR10",
            # Vendored snapshots have no nested .git directory. Do not record
            # the enclosing team repository's HEAD as the upstream model commit.
            "commit": (args.repo / "UPSTREAM_COMMIT").read_text().strip(),
            "weights_source": "https://drive.google.com/file/d/17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq/view",
        },
        "preprocessing": {
            "dataset_py": str(args.project / "dataset.py"), "sha256_before": baseline_hash,
            "project_mean": dataset.CIFAR10_MEAN, "project_std": dataset.CIFAR10_STD,
            "upstream_mean": upstream_mean, "upstream_std": upstream_std,
            "method": "Existing test_evaluation loader (ToTensor only), then existing normalize_for_classifier with explicit statistics.",
        },
        "models": {},
    }

    def save():
        report["preprocessing"]["sha256_after"] = sha256(args.project / "dataset.py")
        report["preprocessing"]["unchanged"] = baseline_hash == report["preprocessing"]["sha256_after"]
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    loaders = dataset.create_dataloaders(data_dir=args.project / "data", batch_size=args.batch_size,
                                         num_workers=0, download=False, pin_memory=False)
    test_loader = loaders.test_evaluation
    assert len(test_loader.dataset) == 10_000
    assert not test_loader.drop_last
    assert isinstance(test_loader.sampler, torch.utils.data.SequentialSampler)
    probe_images, probe_labels = next(iter(test_loader))
    assert probe_images.shape == (min(args.batch_size, 10_000), 3, 32, 32)
    assert 0 <= probe_images.min() and probe_images.max() <= 1
    probe_images, probe_labels = probe_images.to(args.device), probe_labels.to(args.device)
    save()
    for name, factory in (("resnet18", resnet18), ("vgg16_bn", vgg16_bn), ("densenet121", densenet121)):
        result = report["models"][name] = {}
        started = time.perf_counter()
        try:
            path = args.repo / "cifar10_models/state_dicts" / f"{name}.pt"
            state = torch.load(path, map_location="cpu", weights_only=True)
            assert isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values())
            model = factory(pretrained=False)
            mismatch = model.load_state_dict(state, strict=True)
            assert not mismatch.missing_keys and not mismatch.unexpected_keys
            result["strict_load"] = {"passed": True, "missing_keys": mismatch.missing_keys,
                                     "unexpected_keys": mismatch.unexpected_keys, "state_dict_keys": len(state)}
            # Exercise the repository's unchanged pretrained=True code path too.
            del model, state
            model = factory(pretrained=True, device="cpu").to(args.device).eval()
            model.requires_grad_(False)
            result["upstream_pretrained_factory"] = "PASSED"
            result["checkpoint"] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            result["parameter_count"] = sum(p.numel() for p in model.parameters())
            result["forward"] = []
            with torch.inference_mode():
                for batch in (1, 2, args.batch_size):
                    logits = model(dataset.normalize_for_classifier(probe_images[:batch], mean=upstream_mean, std=upstream_std))
                    assert logits.shape == (batch, 10) and torch.isfinite(logits).all()
                    result["forward"].append({"input": [batch, 3, 32, 32], "output": list(logits.shape), "finite": True})
            frozen_buffers = {n: b.clone() for n, b in model.named_buffers()}
            x = probe_images[:2].clone().requires_grad_(True)
            loss = F.cross_entropy(model(dataset.normalize_for_classifier(x, mean=upstream_mean, std=upstream_std)), probe_labels[:2])
            loss.backward()
            assert x.grad is not None and x.grad.shape == x.shape
            assert torch.isfinite(x.grad).all() and (x.grad.abs().flatten(1).sum(1) > 0).all()
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            assert all(torch.equal(b, frozen_buffers[n]) for n, b in model.named_buffers())
            result["input_gradient"] = {
                "passed": True, "shape": list(x.grad.shape), "finite": True,
                "nonzero_per_image": (x.grad.abs().flatten(1).sum(1) > 0).tolist(),
                "abs_max": x.grad.abs().max().item(), "abs_mean": x.grad.abs().mean().item(),
                "parameter_gradients_none": True, "batchnorm_buffers_unchanged": True,
            }
            del frozen_buffers, x, loss
            print(f"{name}: strict load, original factory, forward, frozen-model input gradients PASSED", flush=True)
            result["accuracy"] = {}
            predictions = {}
            for mode, mean, std in (("upstream", upstream_mean, upstream_std),
                                    ("project_default", dataset.CIFAR10_MEAN, dataset.CIFAR10_STD)):
                correct = total = 0
                mode_started = time.perf_counter()
                all_predictions = []
                with torch.inference_mode():
                    for index, (images, labels) in enumerate(test_loader):
                        images, labels = images.to(args.device), labels.to(args.device)
                        logits = model(dataset.normalize_for_classifier(images, mean=mean, std=std))
                        assert logits.shape == (images.shape[0], 10) and torch.isfinite(logits).all()
                        predicted = logits.argmax(1)
                        correct += int((predicted == labels).sum())
                        total += len(labels)
                        all_predictions.append(predicted)
                        if index % 20 == 0:
                            print(f"{name}/{mode}: {total}/10000, {correct / total:.2%}, elapsed {time.perf_counter()-mode_started:.1f}s", flush=True)
                assert total == 10_000
                predictions[mode] = torch.cat(all_predictions)
                result["accuracy"][mode] = {"correct": correct, "total": total, "percent": 100 * correct / total,
                                            "seconds": time.perf_counter() - mode_started}
                save()
                print(f"{name}/{mode}: {correct}/{total} = {100*correct/total:.2f}%", flush=True)
            result["normalization_prediction_disagreements"] = int((predictions["upstream"] != predictions["project_default"]).sum())
            result["status"] = "PASSED"
            del model, predictions
        except Exception:
            result["status"] = "FAILED"
            result["error"] = traceback.format_exc()
            print(result["error"], flush=True)
        result["seconds"] = time.perf_counter() - started
        save()
        gc.collect()
    assert report["preprocessing"]["unchanged"]
    print(f"Results: {args.output.resolve()}", flush=True)
    if any(r["status"] != "PASSED" for r in report["models"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
