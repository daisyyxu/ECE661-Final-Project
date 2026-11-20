# Pretrained CIFAR-10 Models

## Source and Architectures

All three classifiers come from
[huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10),
upstream revision `641cac24371b17052b9bb6e56af1c83b5e97cd7f`, recorded in
[`UPSTREAM_COMMIT`](../compatibility/vendor/PyTorch_CIFAR10/UPSTREAM_COMMIT).
The CIFAR-10 adaptations accept normalized RGB `[B,3,32,32]` inputs and return
`[B,10]` logits. Do not substitute torchvision's ImageNet architectures or weights.

| Model | Factory | Vendored source | Weight file |
| --- | --- | --- | --- |
| ResNet18 | `resnet18` | [`resnet.py`](../compatibility/vendor/PyTorch_CIFAR10/cifar10_models/resnet.py) | `resnet18.pt` |
| VGG16-BN | `vgg16_bn` | [`vgg.py`](../compatibility/vendor/PyTorch_CIFAR10/cifar10_models/vgg.py) | `vgg16_bn.pt` |
| DenseNet121 | `densenet121` | [`densenet.py`](../compatibility/vendor/PyTorch_CIFAR10/cifar10_models/densenet.py) | `densenet121.pt` |

These model modules are unmodified upstream source, distributed with the original
[`MIT LICENSE`](../compatibility/vendor/PyTorch_CIFAR10/LICENSE).

## Download, Storage, and Restoration

The upstream README provides the official
[Google Drive weight archive](https://drive.google.com/file/d/17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq/view).
For a new setup, download and extract that archive, locate the three filenames
above, and copy only those files into:

```text
compatibility/vendor/PyTorch_CIFAR10/cifar10_models/state_dicts/
├── resnet18.pt
├── vgg16_bn.pt
└── densenet121.pt
```

Alternatively, restore the existing `ECE661_Pretrained_Models.zip` backup.
From the project root, set `MODEL_BACKUP` to its location and extract only the
three checkpoint members (without overwriting any existing files):

```bash
MODEL_BACKUP=/path/to/ECE661_Pretrained_Models.zip
unzip -n "$MODEL_BACKUP" \
  'compatibility/vendor/PyTorch_CIFAR10/cifar10_models/state_dicts/resnet18.pt' \
  'compatibility/vendor/PyTorch_CIFAR10/cifar10_models/state_dicts/vgg16_bn.pt' \
  'compatibility/vendor/PyTorch_CIFAR10/cifar10_models/state_dicts/densenet121.pt' -d .
```

If the files are already present on the shared server, use them without
downloading or restoring again. Keep datasets, checkpoint files, ZIP archives,
and `.venv` out of Git. The existing `.gitignore` excludes these assets.

### Checkpoint Integrity

These are the SHA256 values measured for the adopted checkpoints in
[`compatibility/results.json`](../compatibility/results.json):

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `resnet18.pt` | 44,775,417 | `72d30ca70e7d54e24a26628113604c40bc872eb8dc7a44f50ec58c3a1400f1b0` |
| `vgg16_bn.pt` | 134,652,824 | `9c10607463b05f9bf3b879eeb06f24e5ef80db2c25d7c3a0e642d6e4aa98cf5c` |
| `densenet121.pt` | 28,435,902 | `8d5c7e8d0383f660de7201d1b29c9964c0f4393394e6066460bee04a68f01173` |

Check restored files from the project root:

```bash
sha256sum compatibility/vendor/PyTorch_CIFAR10/cifar10_models/state_dicts/resnet18.pt \
  compatibility/vendor/PyTorch_CIFAR10/cifar10_models/state_dicts/vgg16_bn.pt \
  compatibility/vendor/PyTorch_CIFAR10/cifar10_models/state_dicts/densenet121.pt
```

On macOS, use `shasum -a 256` instead of `sha256sum`. Compare every digest
against the table before loading the files.

## Normalization and Loading

Use the existing `dataset.py` interface with the adopted statistics:

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
```

Patch/test DataLoaders return raw pixels in `[0,1]`. Apply any future patch in
pixel space, then call `dataset.normalize_for_classifier` exactly once before
the classifier. Freeze model parameters and use `eval()` while leaving autograd
enabled for input-gradient computations.

Example strict loading, run from the project root with `.venv` active:

```python
from pathlib import Path
import torch
from dataset import create_dataloaders, normalize_for_classifier
from compatibility.vendor.PyTorch_CIFAR10.cifar10_models.resnet import resnet18
from compatibility.vendor.PyTorch_CIFAR10.cifar10_models.vgg import vgg16_bn
from compatibility.vendor.PyTorch_CIFAR10.cifar10_models.densenet import densenet121

factories = {"resnet18": resnet18, "vgg16_bn": vgg16_bn, "densenet121": densenet121}
name = "resnet18"  # or "vgg16_bn" / "densenet121"
weights_dir = Path("compatibility/vendor/PyTorch_CIFAR10/cifar10_models/state_dicts")
model = factories[name](pretrained=False)
state = torch.load(weights_dir / f"{name}.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval().requires_grad_(False)

images, labels = next(iter(create_dataloaders(download=False).test_evaluation))
images.requires_grad_(True)
logits = model(normalize_for_classifier(images))  # [B,10]; input gradients retained
```

## Reproduce Loading and Compatibility Checks

Activate the existing environment and run from the project root after restoring
CIFAR-10 under `data/` and the weights above:

```bash
source .venv/bin/activate

# Three-model strict loading, forward B=1/2/128, frozen-model input gradients,
# BatchNorm-buffer checks, and full 10,000-image clean evaluation.
# The explicit output name preserves the historical results.json and test.log.
python compatibility/test_pretrained.py --project . --batch-size 128 --threads 4 \
  --device cpu --output compatibility/results_retest.json

# Independent DenseNet121 loader and evaluation/check suite.
python -m models.densenet121.load_model --device cpu
python -m models.densenet121.test_model --device cpu --batch-size 128 --threads 4 \
  --output models/densenet121/results/evaluation_retest.json \
  --log models/densenet121/results/test_retest.log
```

These commands use local data and weights only; they do not train or download.
Verified environment: Python 3.11.17, PyTorch 2.15.0+cpu, TorchVision 0.29.2+cpu.
Recorded full-test clean accuracies are ResNet18 93.07%, VGG16-BN 94.00%, and
DenseNet121 94.06%. The independent DenseNet121 run also measured 94.06%.
See [`models/densenet121/README.md`](../models/densenet121/README.md) for its
evaluation-only command and [`PUBLICATION_NOTES.md`](PUBLICATION_NOTES.md)
for historical report interpretation.
