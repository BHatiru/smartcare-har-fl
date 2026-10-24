# Controlled 15-class HAR federation comparison

This package contains the completed comparison incorporated into manuscript revision `ce52eb0` on 25 October 2026. It supersedes the historical 10-class FL screen as the paper's FL evidence. Its source files and experiment records retain their original bytes and hashes.

## Results and protocol

| Method | Test accuracy, % | Test macro-F1, % | Worst-client test accuracy, % |
|---|---:|---:|---:|
| Centralized | 92.02 +/- 0.55 | 92.03 +/- 0.55 | 90.80 +/- 0.73 |
| Local only | 82.63 +/- 0.39 | 82.66 +/- 0.34 | 76.73 +/- 2.26 |
| FedAvg | 78.44 +/- 1.52 | 77.78 +/- 1.87 | 73.16 +/- 2.23 |
| FedBN | 83.47 +/- 0.85 | 83.31 +/- 1.10 | 81.26 +/- 2.22 |
| FedProx | 78.00 +/- 2.84 | 77.23 +/- 3.21 | 72.83 +/- 3.57 |

Values are mean +/- sample standard deviation across model seeds 42, 43 and 44. There is **one** client-partition seed, not three independent client partitions. See [paper CSV](paper/method_summary.csv) for full precision and [per-seed values](paper/seed_metrics.csv).

Training uses 7,696 clips, development 2,336 and test 4,125, separated by performer. Five clients use a Dirichlet label allocation with alpha 0.5, partition seed 42. Each method starts from the same fresh initialization within a model seed and receives 50 rounds/epoch blocks, one local pass, batch size 16, SGD learning rate 0.01, momentum 0.9 and weight decay 0.0005. Learning-rate multipliers are 1 through round 20, 0.1 through 35, then 0.01. FedProx uses mu 0.01. Optimizers reset each round. FedBN retains every client's BN affine parameters and buffers; there is no pooled calibration or warm start.

Checkpoint selection maximizes pooled development macro-F1, then minimizes cross-entropy, then favors the earliest round. The selected files/states are frozen before final test inference. The original benchmark was examined in earlier development; this is not an untouched external validation set. Label-based allocation constructs synthetic cohorts and is not a deployment routing rule.

The 444,507-parameter custom ST-GCN-style model retains the historical directed graph and preprocessing. It is not a faithful reproduction of upstream ST-GCN++. See [frozen experiment protocol](lab_results/EXPERIMENT_PROTOCOL.json) and [full contract](lab_results/primary/contract.json).

## What is included

- Exact training, preparation, evaluation and test source files, bound by the original source hashes.
- Original train/development split manifests and class-to-client proportions; test allocation and sample identities.
- All 15 selected checkpoints, selected development predictions, complete per-round development histories, final-test logits and metrics.
- Frozen checkpoint selection, original audits, environment versions, manuscript table/curves and full-precision CSVs.
- A [release manifest](RELEASE_MANIFEST.json) distinguishing this public package from the original transfer bundle.

Raw NTU annotations, prepared input tensors, final nonselected checkpoints, resume states, duplicate round JSON files and operational recovery logs are not redistributed here. Complete histories remain in each run's `result.json`. `BUNDLE_MANIFEST.json` describes the original input transfer, including omitted raw data; it is provenance, **not** a claim that every listed input is included in this public release. Original absolute paths in records identify their source machine; they are not required destination paths.

## Verify the released results without training

From this directory, with Python 3.12 and NumPy 2.4.4:

```bash
python verify_release.py
```

This checks the release and original source/checkpoint/prediction hashes, reconstructs pooled/per-class/client/subject metrics independently from saved logits, and checks the paper CSV aggregates. It needs neither a GPU nor raw NTU input tensors. It verifies the saved results, not a rerun of model inference.

For the implementation tests, also install PyTorch 2.11.0 in a suitable environment:

```bash
python -m unittest demo1_fed_skeleton.test_corrected_fl demo1_fed_skeleton.test_development_fl demo1_fed_skeleton.test_resumable_fl demo1_fed_skeleton.test_final_evaluation
```

The recorded training environment used PyTorch 2.11.0+cu128, CUDA 12.8 and an RTX 5080, with TF32 disabled and deterministic algorithms. CPU tests do not establish bit-identical training across platforms.

## Create a new training run

Obtain the public PYSKL NTU-60 HRNet annotations through the [PYSKL data-preparation instructions](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md), subject to the dataset's terms. The annotation file used here has SHA256 `953836d9dcae0b139ba524cee5003346c5cb4cbb26b0371cbf927f0cfa383c6a`.

Run from this directory and choose **new** output paths:

```bash
python -m demo1_fed_skeleton.prepare_development_fl --annotation /path/to/ntu60_hrnet.pkl --output-dir reproduced_data --train-per-class 0 --development-per-class 0 --split-seed 20261001 --subset-seed 20261002 --partition-seed 42 --development-subjects 4 --num-clients 5 --alpha 0.5

python experiments/har_fl_lab_20261024/gpu_queue.py --data-dir reproduced_data --output-dir new_runs --rounds 50 --seeds 42 43 44 --methods centralized local fedavg fedbn fedprox --batch-size 16 --deadline YOUR_FUTURE_ISO8601_DEADLINE_WITH_TIMEZONE --purpose primary
```

The deadline placeholder must be replaced with an actual future timestamp. The CUDA runner refuses CPU fallback and checkpoints completed rounds. A new preparation records its own source path and therefore its own protocol hash; do not overwrite the published protocol to make it appear unchanged.

The original test evaluator is included as `demo1_fed_skeleton/final_evaluation.py`. Replaying inference additionally requires restoring the omitted test tensor file with its original identity/label ordering and hash; the binding files specify that contract. The prediction-based verification above works with this release alone. Do not use the already-published test predictions to select a new model.

## Interpretation

FedBN improves over FedAvg across all three paired seeds; its pooled advantage over local-only training is small, while the worst-client improvement is larger. This supports local normalization under the evaluated label skew, not a universal benefit of federation. Client minima are descriptive for these synthetic cohorts. The experiment does not establish physiological fusion, patient safety, privacy guarantees, federated gateway latency or a new algorithm.

Retained training exposure is matched; a storage interruption discarded an unrecorded partial round. Timings describe retained work rather than total wall-clock cost including recovery. See [resume accounting](lab_results/RESUME_ACCOUNTING.md).

Separate local-head and layer-wise shrinking research began after this comparison. Its incomplete development results are not included in this manuscript-result release.
