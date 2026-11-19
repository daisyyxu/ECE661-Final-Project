# ECE661 Adversarial Patch — CIFAR-10 data pipeline

Portable preprocessing component for the ECE661 final project. Run commands from
the folder containing this README; no user-specific absolute paths are required.
This stage contains data preparation and pretrained-model compatibility tests;
it does not train classifiers or implement an adversarial patch.

## Environment

Verified on Linux x86_64, Python **3.11.17**, PyTorch **2.15.0+cpu**,
TorchVision **0.29.2+cpu**, with matplotlib. GPU/CUDA compatibility has not
been tested. If you already have the project environment, reuse it:

```bash
cd ECE661-Final-Project
source .venv/bin/activate
```

For a separate CPU environment (Python 3.11):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install matplotlib==3.11.2
```

`requirements.txt` pins the three direct dependencies; the versions above describe
the actually verified environment. Legacy PyTorch Lightning is not required for
the adopted model inference interfaces.

## Data format and loaders

`data/cifar-10-batches-py/` contains torchvision's original CIFAR-10 Python batch
files (`data_batch_1` through `data_batch_5`, `test_batch`, `batches.meta`).
The original archive may also be present as `data/cifar-10-python.tar.gz`.
There are **50,000 training** and **10,000 test** images, 10 balanced classes.
No per-image PNG conversion is performed. `outputs/cifar10_samples.png` is only
a labeled visualization grid, not a replacement dataset.

Each batch is `(images, labels)`: images are float32 `[B,3,32,32]` RGB tensors;
labels are int64 `[B]` with IDs 0–9. The final batch may be smaller (no drop_last).
Class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

| Loader attribute | Split | Transforms | Shuffle | Output space |
| --- | --- | --- | --- | --- |
| `classifier_train` | train, 50,000 | RandomCrop(32,padding=4), RandomHorizontalFlip, ToTensor, Normalize | True | normalized |
| `patch_optimization` | train, 50,000 | ToTensor only | True | [0,1] |
| `test_evaluation` | test, 10,000 | ToTensor only; no random augmentation | False | [0,1] |

```python
from dataset import create_dataloaders, normalize_for_classifier

loaders = create_dataloaders(
    data_dir="./data", batch_size=128, num_workers=0, download=False,
)
train_images, train_labels = next(iter(loaders.classifier_train))
patch_images, patch_labels = next(iter(loaders.patch_optimization))
test_images, test_labels = next(iter(loaders.test_evaluation))

# Classifier training images are ALREADY normalized. Do not normalize twice.
# logits = model(train_images)

# Clean inference: normalize raw test pixels immediately before the classifier.
# logits = model(normalize_for_classifier(test_images))
```

`create_dataloaders` also accepts `pin_memory`, and optional `mean`/`std` for
classifier-training normalization. Defaults: batch_size=128, num_workers=0,
download=True, pin_memory selected from CUDA availability. Use `download=False`
after restoring an existing dataset backup; no downloads are needed in that case. Future new setups
can use `download=True` to obtain the official dataset through torchvision.
For multiprocessing, place the entry point under `if __name__ == "__main__":`.

## Normalization contract

The defaults now match all three adopted CIFAR-10 checkpoints:

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
```

Normalization is `(images - mean) / std`, independently per RGB channel.
`normalize_for_classifier(images, mean=..., std=...)` supports CHW and NCHW,
preserves device/dtype, does not mutate the input, and propagates input gradients.
Statistics remain configurable for other checkpoints:

```python
from dataset import CIFAR10_MEAN, CIFAR10_STD

mean, std = CIFAR10_MEAN, CIFAR10_STD  # replace only when the checkpoint requires it
loaders = create_dataloaders(download=False, mean=mean, std=std)
images, labels = next(iter(loaders.test_evaluation))
model_inputs = normalize_for_classifier(images, mean=mean, std=std)
```

Changing loader statistics affects only `classifier_train`; patch and test loaders
always return raw `[0,1]` tensors. For future patch integration, the order must be:
raw pixels → compose a pixel-space patch (keep valid pixels in [0,1]) →
`normalize_for_classifier` → model. Do not apply patches to already-normalized
training batches. Patch composition/optimization is intentionally not implemented.
When computing input/patch gradients, freeze model parameters and use `model.eval()`,
but do not wrap that forward pass in `no_grad()` or `inference_mode()`.

## Adopted pretrained models and verified compatibility

Source: [huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10),
commit `641cac24371b17052b9bb6e56af1c83b5e97cd7f`.
Source checkout: `compatibility/vendor/PyTorch_CIFAR10/`.
These are CIFAR-10 models, not torchvision ImageNet checkpoints.
Official weights came from the repository's
[Google Drive backup](https://drive.google.com/file/d/17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq/view).
The Box link returned 404 and the server's direct Drive transfer stalled, so the
official archive was downloaded locally and the three weights transferred to the
server with SHA256 verification. Weights are deliberately NOT uploaded to GitHub.
Restore the three `.pt` files from `ECE661_Pretrained_Models.zip` into
`compatibility/vendor/PyTorch_CIFAR10/cifar10_models/state_dicts/`.
See [pretrained model setup](docs/PRETRAINED_MODELS.md) for hashes and provenance.
The three vendored model modules are unchanged from upstream, with their
[MIT LICENSE](compatibility/vendor/PyTorch_CIFAR10/LICENSE) retained.

Actual server results (CPU; full 10,000-image test set, batch_size=128):

| Model | strict=True state_dict | Forward B=1,2,128 | Frozen-model input gradient | Clean accuracy (adopted normalization) |
| --- | --- | --- | --- | --- |
| ResNet18 | passed, no missing/unexpected keys | [B,3,32,32] → [B,10] | finite, nonzero | 9307/10000 = 93.07% |
| VGG16-BN | passed, no missing/unexpected keys | [B,3,32,32] → [B,10] | finite, nonzero | 9400/10000 = 94.00% |
| DenseNet121 | passed, no missing/unexpected keys | [B,3,32,32] → [B,10] | finite, nonzero | 9406/10000 = 94.06% |

The original `pretrained=True` factories also passed unchanged; parameter gradients
stayed None and BatchNorm buffers stayed unchanged during input-gradient checks.
No model-source modifications or PyTorch downgrade were necessary.

Raw evidence: `compatibility/results.json`, `compatibility/test.log`;
reproduction script: `compatibility/test_pretrained.py`.
Published evidence retains the actual measurements; machine-specific absolute
paths have been made relative. Original unmodified reports remain in the local
code backup. See [publication notes](docs/PUBLICATION_NOTES.md).
That historical report also tested the PREVIOUS project default std[0]=0.2470,
giving 93.07%, 94.00%, and 94.07% respectively (DenseNet predictions differed on
2 images). Its `upstream` results use std[0]=0.2471 and correspond to the newly
adopted defaults above. The historical report's dataset.py hash predates this
intentional normalization update; it is preserved rather than overwritten.

Optional model retest, using existing files only:

```bash
.venv/bin/python compatibility/test_pretrained.py --project . --output compatibility/results_adopted_defaults.json
```

## Preprocessing validation (offline)

Run all preprocessing checks and save the labeled sample grid:

```bash
.venv/bin/python test_dataset.py --no-download --batch-size 128 --num-workers 0
# Additional configurable-batch / multiprocessing regression:
.venv/bin/python test_dataset.py --no-download --batch-size 17 --num-workers 2 --output outputs/cifar10_samples_b17.png
```

The checks cover split counts and class balance, batch shapes/dtypes/labels,
transform order, shuffling/no drop_last, deterministic test data, normalized
classifier bounds, ALL 50,000 patch and 10,000 test images staying in [0,1],
default/custom statistics, CHW/NCHW normalization, analytical input gradients,
non-mutating normalization, and invalid-input rejection. No patch is implemented
by these tests. Server run evidence is saved in `outputs/preprocessing_validation.log`
and `outputs/preprocessing_validation_b17.log`.

Both configurations above were actually run on the server and passed with exit
code 0. Each run scanned all 50,000 patch and 10,000 test images; both ranges were
exactly [0.0, 1.0]. Default/custom normalization, analytical gradients,
deterministic evaluation, transform order, class balance and labels all passed.
These are existing results, not new experiments performed during publication.

## Backup restoration and repository scope

The desktop backups are separate from this Git repository:

- `ECE661_Code.zip`: original project code, tests, documentation and results.
- `ECE661_Dataset.zip`: restore its `data/` directory next to `dataset.py`.
- `ECE661_Pretrained_Models.zip`: restore the three checkpoint files as above.

Each ZIP includes a per-file SHA256 `MANIFEST.json` and the dataset/model packages
include restoration and preprocessing notes. Checkpoint and CIFAR-10 data files,
ZIP archives, virtual environments and caches are excluded by `.gitignore`.
Only the independent `preprocessing` branch is published; no main-branch merge,
model training or adversarial patch implementation is part of this handoff.
