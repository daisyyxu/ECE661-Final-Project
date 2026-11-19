# Adopted CIFAR-10 pretrained models

Source: https://github.com/huyvnphan/PyTorch_CIFAR10

Upstream commit: `641cac24371b17052b9bb6e56af1c83b5e97cd7f`.
The vendored `resnet.py`, `vgg.py` and `densenet.py` are unmodified; the upstream
MIT copyright and license are retained alongside them. `UPSTREAM_COMMIT` records
the model-source revision independently of the team's Git commit.

Weights are not included in Git. Use the existing `ECE661_Pretrained_Models.zip`
backup, whose folders already match this repository's layout, or obtain the
original archive from the source repository's official
[Google Drive backup](https://drive.google.com/file/d/17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq/view).
No weights or dataset were downloaded again during GitHub publication.

Required destination: `compatibility/vendor/PyTorch_CIFAR10/cifar10_models/state_dicts/`.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| resnet18.pt | 44775417 | `72d30ca70e7d54e24a26628113604c40bc872eb8dc7a44f50ec58c3a1400f1b0` |
| vgg16_bn.pt | 134652824 | `9c10607463b05f9bf3b879eeb06f24e5ef80db2c25d7c3a0e642d6e4aa98cf5c` |
| densenet121.pt | 28435902 | `8d5c7e8d0383f660de7201d1b29c9964c0f4393394e6066460bee04a68f01173` |

All models take normalized RGB `[B,3,32,32]` tensors and return 10 logits per
image. Normalize raw `[0,1]` pixels with mean `(0.4914,0.4822,0.4465)` and
std `(0.2471,0.2435,0.2616)` immediately before inference. Do not use torchvision's
ImageNet checkpoints or its ImageNet preprocessing as replacements.

CPU checks using PyTorch 2.15.0+cpu and TorchVision 0.29.2+cpu passed strict state
loading, upstream pretrained factories, forward batches 1/2/128 and input
backpropagation with frozen parameters. Full 10,000-image clean accuracies:
ResNet18 93.07%, VGG16-BN 94.00%, DenseNet121 94.06%.
The raw recorded evidence is in `compatibility/results.json`; these were prior
server compatibility tests, not new publication-time experiments.
