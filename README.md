# ECE661 Final Project: Adversarial Patch Attacks

This project investigates adversarial patch attacks on CIFAR-10 image
classifiers, including the effects of patch size, placement, transformations,
and transferability across architectures. Planned targeted and untargeted
experiments use ResNet18 as the primary attack model, with VGG16-BN and
DenseNet121 for transferability evaluation.

## Project Status

The CIFAR-10 data pipeline, pretrained model integration, and CPU compatibility
tests are complete. The independent DenseNet121 loading and evaluation module
is also tested. Adversarial patch optimization and attack evaluation are planned;
they are not implemented in this branch.

## Dataset and Preprocessing

CIFAR-10 contains 50,000 training and 10,000 test images across 10 classes.
Images are 32 × 32 RGB. Data stays in torchvision's original Python batch format
under `data/cifar-10-batches-py/`; it is not converted into individual PNG files.

[`dataset.py`](dataset.py) provides three DataLoaders:

| DataLoader | Split | Preprocessing | Shuffle |
| --- | --- | --- | --- |
| `classifier_train` | Train | RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor, Normalize | True |
| `patch_optimization` | Train | ToTensor only | True |
| `test_evaluation` | Test | ToTensor only; no random augmentation | False |

Batches contain float32 images `[B,3,32,32]` and int64 labels `[B]` in 0–9.
The default batch size is 128 and is configurable; the final batch is not dropped.
Class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

### Usage

Run from the project root:

```python
from dataset import create_dataloaders, normalize_for_classifier

loaders = create_dataloaders(
    data_dir="./data", batch_size=128, num_workers=0, download=False,
)
images, labels = next(iter(loaders.patch_optimization))  # raw pixels in [0,1]

# Future patch composition belongs here, before normalization.
model_inputs = normalize_for_classifier(images)
```

Use `download=True` on a new setup to download CIFAR-10 through torchvision;
use `download=False` with an existing dataset. `num_workers` and `pin_memory`
are configurable. Multiprocessing entry points need an
`if __name__ == "__main__":` guard.

### Normalization

All three adopted pretrained models use:

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
```

The required order for patch experiments is:

**Raw Image [0,1] → Patch Application → Normalization → Classifier**

`normalize_for_classifier(images, mean=..., std=...)` is differentiable,
supports CHW/NCHW tensors, and does not modify its input. Statistics remain
configurable for checkpoints that require different preprocessing. Passing
`mean`/`std` to `create_dataloaders` changes only `classifier_train`; patch and
test images always stay in `[0,1]`. Classifier-training batches are already
normalized and must not be normalized twice.

## Pretrained Models

We use CIFAR-10 classifiers from
[huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10),
not torchvision ImageNet checkpoints. Full 10,000-image test-set results with
the normalization above are:

| Model | Correct / Total | Clean Accuracy |
| --- | --- | --- |
| ResNet18 | 9307 / 10000 | 93.07% |
| VGG16-BN | 9400 / 10000 | 94.00% |
| DenseNet121 | 9406 / 10000 | 94.06% |

All three passed strict weight loading, forward propagation with batch sizes
1/2/128, and finite, nonzero input-gradient checks with frozen parameters.
During input-gradient computations, keep `model.eval()` and freeze parameters,
but do not wrap the forward pass in `no_grad()` or `inference_mode()`.

Evidence: [`compatibility/results.json`](compatibility/results.json) and
[`compatibility/test.log`](compatibility/test.log). The independent DenseNet121
module and its additional evaluation are documented in
[`models/densenet121/README.md`](models/densenet121/README.md).
Weights are not included in Git; see
[`docs/PRETRAINED_MODELS.md`](docs/PRETRAINED_MODELS.md) for setup and reproduction.

## Environment Setup

Tests were verified on Linux x86_64 with Python 3.11.17, PyTorch 2.15.0+cpu,
TorchVision 0.29.2+cpu, and matplotlib 3.11.2. Reuse the existing `.venv`
when available. To create a matching CPU environment from the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Legacy PyTorch Lightning is not needed for these inference interfaces.
CUDA and TPU compatibility have not been verified.

## Testing

With `.venv` active and the existing dataset and weights restored, run from the
project root. These commands use separate output names to preserve saved evidence:

```bash
# Preprocessing validation and labeled sample grid.
python test_dataset.py --no-download --batch-size 128 --num-workers 0 \
  --output outputs/cifar10_samples_retest.png

# Optional configurable-batch / multiprocessing regression.
python test_dataset.py --no-download --batch-size 17 --num-workers 2 \
  --output outputs/cifar10_samples_b17_retest.png

# Three-model compatibility and full clean evaluation; no data/weight downloads.
python compatibility/test_pretrained.py --project . \
  --output compatibility/results_retest.json

# Independent DenseNet121 checks and full clean evaluation.
python -m models.densenet121.test_model --device cpu --batch-size 128 --threads 4 \
  --output models/densenet121/results/evaluation_retest.json \
  --log models/densenet121/results/test_retest.log
```

Both preprocessing configurations and the three-model CPU suite passed.
Preprocessing evidence is retained in
[`outputs/preprocessing_validation.log`](outputs/preprocessing_validation.log)
and [`outputs/preprocessing_validation_b17.log`](outputs/preprocessing_validation_b17.log).
The labeled grid is [`outputs/cifar10_samples.png`](outputs/cifar10_samples.png).
Historical result interpretation is documented in
[`docs/PUBLICATION_NOTES.md`](docs/PUBLICATION_NOTES.md).

## Project Structure

```text
ECE661-Final-Project/
├── dataset.py
├── test_dataset.py
├── compatibility/           # Three-model checks, evidence, vendored source + LICENSE
├── models/
│   └── densenet121/         # Independent loading, evaluation, tests, and results
├── docs/
│   ├── PRETRAINED_MODELS.md
│   └── PUBLICATION_NOTES.md
├── outputs/                # Preprocessing evidence and sample grids
├── data/                   # Local CIFAR-10 files; excluded from Git
├── requirements.txt
└── README.md
```

## Planned Experiments

Planned comparisons include targeted/untargeted attacks; patch sizes 3×3, 5×5,
7×7, and 16×16; fixed/random placement; square/circular shapes; random-noise
and real-image baselines; and transfer across ResNet18, VGG16-BN, and DenseNet121.
Evaluation will report attack success rate, clean accuracy, and patched accuracy.
No attack results are reported yet.

## References

- Brown et al., [*Adversarial Patch*](https://arxiv.org/abs/1712.09665), 2017.
- Huy Phan, [*PyTorch CIFAR-10 Pretrained Models*](https://github.com/huyvnphan/PyTorch_CIFAR10).
