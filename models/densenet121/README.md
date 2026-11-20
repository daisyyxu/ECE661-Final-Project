# Member 3 — CIFAR-10 DenseNet121

Independent loading, clean evaluation and input-gradient checks. Reuses the
existing root `dataset.py` and the adopted vendored model source without changing
data preprocessing or model architecture. No patch, EOT or transferability code.

## Source and local weights

Source: https://github.com/huyvnphan/PyTorch_CIFAR10

Upstream revision: `641cac24371b17052b9bb6e56af1c83b5e97cd7f`.
Source module: `compatibility/vendor/PyTorch_CIFAR10/cifar10_models/densenet.py`.
Its original MIT LICENSE remains at `compatibility/vendor/PyTorch_CIFAR10/LICENSE`.

Default checkpoint location:
`compatibility/vendor/PyTorch_CIFAR10/cifar10_models/state_dicts/densenet121.pt`.
Use the already-downloaded checkpoint (28,435,902 bytes), whose SHA256 is
`8d5c7e8d0383f660de7201d1b29c9964c0f4393394e6066460bee04a68f01173`.
Official archive provenance:
https://drive.google.com/file/d/17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq/view

Neither dataset nor checkpoint is included in Git. These modules never download
them; restore the existing backups or pass `--weights` / `--data-dir`.

## Load and reuse

From the repository root, with its existing `.venv` active:

```python
from models.densenet121.load_model import load_model, forward_pixels
from dataset import create_dataloaders

model = load_model(device="cpu")  # hash check + strict=True + eval + frozen parameters
images, labels = next(iter(create_dataloaders(download=False).test_evaluation))
images.requires_grad_(True)  # optional: input gradients remain available
logits = forward_pixels(model, images)  # [B,3,32,32] -> [B,10]
# loss = ...; loss.backward()  # not implemented as an attack here
```

`forward_pixels` accepts raw RGB float pixels in `[0,1]` and calls the existing
`dataset.normalize_for_classifier` exactly once with:

- mean = `(0.4914, 0.4822, 0.4465)`
- std = `(0.2471, 0.2435, 0.2616)`

The returned model itself expects normalized inputs. Do not pass already
normalized `classifier_train` images through `forward_pixels`. Do not surround
input-gradient computations with `no_grad()` or `inference_mode()`; freezing
parameters alone preserves input autograd. `inference_mode` is used only inside
the clean-evaluation routine.

## Commands

Run these from the repository root; all default paths are script-relative:

```bash
# Load and inspect the frozen model.
.venv/bin/python -m models.densenet121.load_model --device cpu

# Full required test suite AND a fresh full 10000-image clean evaluation.
.venv/bin/python -m models.densenet121.test_model --device cpu --batch-size 128 --threads 4

# Standalone evaluation-only CLI (separate filenames preserve full-suite evidence).
.venv/bin/python -m models.densenet121.evaluate --device cpu --batch-size 128 --threads 4 \
  --output models/densenet121/results/evaluation_only.json \
  --log models/densenet121/results/evaluation_only.log
```

Direct script execution is also supported (for example,
`.venv/bin/python models/densenet121/test_model.py`). `--weights`, `--data-dir`,
`--device`, `--batch-size`, `--num-workers`, `--threads`, `--output` and `--log`
are available for the test/evaluation CLIs. Missing inputs fail explicitly;
`download=False` is always passed to the existing dataset interface.

## Actual server results

Freshly executed on 2026-11-19 (America/New_York; UTC run timestamp recorded in
the JSON), using the server project's existing `.venv`: Python 3.11.17,
PyTorch 2.15.0+cpu, TorchVision 0.29.2+cpu, CPU, 4 threads, batch size 128,
num_workers=0. This is a new independent run, not a copied historical result.

| Check | Actual result |
| --- | --- |
| Official checkpoint SHA256 + strict=True loading | PASSED |
| Standalone load_model CLI | exit code 0; eval mode; all parameters frozen |
| Forward B=1 | [1,3,32,32] → [1,10], finite |
| Forward B=2 | [2,3,32,32] → [2,10], finite |
| Forward B=128 | [128,3,32,32] → [128,10], finite |
| Input gradient after freezing | finite and nonzero for both tested images |
| Parameter gradients | all None |
| BatchNorm buffers | all 363 unchanged after gradient checks AND full evaluation |
| Full clean accuracy | **9406/10000 = 94.06%** |
| Full test-image pixel range | [0.0,1.0] before normalization |
| Full clean-evaluation elapsed time | 37.78 seconds on this CPU run |
| Existing preprocessing, model source and checkpoint | unchanged; SHA256 checked |

The model has 6,956,426 parameters. Input-gradient absolute maximum was
0.008226151578128338; absolute mean was 0.0004578655061777681.
The exact measurements, source hashes, environment and new timestamp are in
[`results/evaluation.json`](results/evaluation.json); the actual run output is in
[`results/test.log`](results/test.log). The full suite exited with code 0.
The full-suite command writes both files; failures are also recorded and cause a
nonzero exit status. The standalone clean-evaluation function was exercised by
this suite; the evaluate CLI uses that same function.
