# Historical Results and Publication Notes

## Normalization History

The original three-model compatibility run compared two statistics, using the
same complete 10,000-image CIFAR-10 test set and mean
`(0.4914, 0.4822, 0.4465)`:

| Recorded mode | Standard deviation | ResNet18 | VGG16-BN | DenseNet121 |
| --- | --- | --- | --- | --- |
| Historical `project_default` | `(0.2470, 0.2435, 0.2616)` | 9307/10000 (93.07%) | 9400/10000 (94.00%) | 9407/10000 (94.07%) |
| `upstream`, now the project default | `(0.2471, 0.2435, 0.2616)` | 9307/10000 (93.07%) | 9400/10000 (94.00%) | 9406/10000 (94.06%) |

Only the red-channel standard deviation changed: an increase of 0.0001 to match
the adopted checkpoints. The report records two DenseNet121 prediction
disagreements between the settings, with one fewer correct prediction under the
upstream setting; ResNet18 and VGG16-BN predictions did not differ.
Source evidence is preserved in
[`compatibility/results.json`](../compatibility/results.json) and
[`compatibility/test.log`](../compatibility/test.log).

## Reports and Code Versions

The historical `preprocessing.project_std` and `accuracy.project_default`
fields describe the old default, not today's `dataset.py`. The report's
`accuracy.upstream` fields use the current adopted statistics and are the source
of the three-model accuracy table in the root README.

The historical dataset hash is
`1920ab85a3a2d1ccbf0413ffc85afa3d09fad64dbbe89344da0dc061bb658f9d`.
It predates the normalization change; its unchanged before/after hashes mean
the dataset module was not modified *during that historical test*, not that it
matches every later version.

After the default was updated, both preprocessing configurations (batch 128 /
workers 0 and batch 17 / workers 2) passed. Their existing evidence remains in
[`outputs/preprocessing_validation.log`](../outputs/preprocessing_validation.log),
[`outputs/preprocessing_validation_b17.log`](../outputs/preprocessing_validation_b17.log),
and [`outputs/preprocessing_validation.json`](../outputs/preprocessing_validation.json).

An independent DenseNet121 run on 2026-11-19 (America/New_York; recorded in UTC
as 2026-11-20) used the current statistics and again measured 9406/10000
(94.06%). It additionally verified strict loading, batches 1/2/128, frozen
parameters, finite/nonzero input gradients, and all 363 BatchNorm buffers
remaining unchanged through full evaluation. Its evidence is separate:
[`evaluation.json`](../models/densenet121/results/evaluation.json) and
[`test.log`](../models/densenet121/results/test.log).
That report records dataset hash
`d08c3b0b85142e1d5c07e475eff7d56d72ee37c6bfe7cbb99b56f6242a2b61b7`.

The current compatibility script reads whichever project defaults are in
`dataset.py`; rerunning it now will use `std[0]=0.2471` for both modes. It will
not recreate the historical `project_default` condition without an explicit
change to the test configuration. Reproduction commands therefore write new
filenames rather than replacing saved evidence.

## Weight Acquisition History

During the original setup, the upstream Box download returned HTTP 404, and a
direct Google Drive transfer on the server stalled. The official Google Drive
backup archive was downloaded locally; the three adopted checkpoints were then
transferred to the server and checked by SHA256. This describes that setup event,
not a new check of the Box service. No model was retrained or structurally
modified. Current sources, destinations, restoration instructions, and checkpoint
hashes are centralized in [`PRETRAINED_MODELS.md`](PRETRAINED_MODELS.md).

## Original Evidence and Published Files

The initial published snapshot was extracted from `ECE661_Code.zip`; that backup
retains the original server reports and installation/download logs. The dataset
and checkpoint backups are separate from Git.

The original publication made these portability and scope adjustments:

- Machine-specific absolute project paths in historical JSON/log copies were
  replaced with project-relative paths. Recorded measurements, statuses, counts,
  timings, and source/checkpoint hashes were not changed. The published JSON
  includes a `publication` note and the original file's SHA256. Published copies
  therefore need not be byte-identical to the originals.
- The compatibility script reads the vendored `UPSTREAM_COMMIT` file rather than
  querying a nested Git checkout, which is absent from the published snapshot.
  Model checks and accuracy/gradient calculations were retained.
- Direct dependencies were pinned to the verified environment. Only the three
  adopted upstream model modules and their MIT LICENSE were vendored; unused
  training utilities and an upstream report URL with an access-token parameter
  were excluded. Dataset/model files, environments, archives, caches, and local
  transfer utilities are not published.

The current documentation reorganization changes Markdown only. It does not
rerun experiments, replace result JSON, edit or remove any existing test logs,
or alter Python code, checkpoints, or datasets.
