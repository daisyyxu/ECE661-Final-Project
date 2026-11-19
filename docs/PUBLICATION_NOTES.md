# Publication scope and provenance

The GitHub handoff is a reviewed subset extracted from `ECE661_Code.zip`.
The three original desktop backups retain the original server files and include
SHA256 manifests; they are not committed to Git.

No experimental results were recomputed and no experimental logic was changed.
`dataset.py`, `test_dataset.py` and the three adopted upstream model source files
are byte-for-byte identical to the server snapshot. Publication-only adjustments:

- README commands use project-relative paths and document backup restoration.
- Direct dependencies are pinned to the versions installed in the tested server
  environment (Python 3.11.17, torch 2.15.0+cpu, torchvision 0.29.2+cpu,
  matplotlib 3.11.2). GPU/CUDA has not been validated.
- The compatibility script reads the upstream revision from `UPSTREAM_COMMIT`
  instead of invoking Git inside a nested checkout that no longer has `.git`.
  Its model/data checks and accuracy/gradient calculations are unchanged.
- Machine-specific absolute path prefixes in the published historical logs and
  JSON reports are replaced with project-relative paths. Measurement values,
  hashes, counts, statuses and timing fields are not altered; an explanatory
  publication note is added to the JSON copies.
- Only the three adopted model modules are vendored, together with the original
  upstream MIT LICENSE. Unused upstream training code and a public upstream
  report URL containing an accessToken parameter are not included.
- Installation/download logs, transfer patches, old file backups and local
  transfer utilities are omitted from Git. The original code backup preserves
  installation/download history for local reference.

Historical compatibility results precede the intentional default-STD update
from 0.2470 to 0.2471 in the red channel. Their `upstream` accuracy fields already
use the newly adopted defaults; historical `project_default` fields refer to the
old default. Their dataset.py hashes are historical evidence, not hashes of this
README or the current dataset module.

Publication checks consist of file/ZIP integrity checks, secret-pattern and
absolute-path reviews, syntax checks, source-equivalence checks and a review of
the staged Git files. No classifier training, patch optimization or new model
evaluation was run as part of the backup/upload task.
