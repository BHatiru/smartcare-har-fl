# Smart-Care HAR-FL: skeleton activity recognition and federated-learning artifacts

This repository holds the artifacts for the human activity recognition (HAR) and federated-learning (FL) experiments in the following manuscript:

> **Activity-Aware Federated Multimodal IoMT for Continuous Patient Monitoring**
> B. Bodaubay\*, T. Tursynbekov\*, B. Ongdassynov, T. Bayan, A. Yazici (2026 manuscript; \*equal contribution and shared first authorship).
> School of Computing and Artificial Intelligence (SCAI), Nazarbayev University, Astana, Kazakhstan. T. Bayan is also affiliated with the University of Manchester, Manchester, United Kingdom.
> Supported by the Ministry of Science and Higher Education of the Republic of Kazakhstan, grant AP23487613, "Smart-Care: Innovative Multi-Sensor Technology for Elderly and Disabled Health Management".

The completed **15-class, three-seed HAR-FL comparison** is now published in [controlled_har_fl/](controlled_har_fl/README.md). It contains the exact training/evaluation source, tests, selected checkpoints, split manifests, complete development histories and final-test predictions used in manuscript revision `ce52eb0` (25 October 2026).

The repository also retains the earlier HAR/FL experiments and deployed centralized ONNX model. These evidence blocks have different training and selection protocols.

## Current manuscript artifacts

| Evidence block | Result or scope | Location |
|---|---|---|
| Controlled 15-class FL comparison | Three model seeds, one client partition; development-selected checkpoints, 50 rounds | [Protocol, code and results](controlled_har_fl/README.md) |
| FL paper table and curves | Full-precision per-seed/summary CSVs, table source, all development curves | [Paper assets](controlled_har_fl/paper/) |
| Final-test metrics and cohort assignment | Saved logits, identities, allocation, per-class/client/subject metrics | [Final-test artifacts](controlled_har_fl/lab_results/final_test/) |
| Selected trained models | All 15 selected checkpoints, hashes and complete development histories | [Training artifacts](controlled_har_fl/lab_results/primary/) and [frozen selection](controlled_har_fl/lab_results/frozen_selection.json) |
| Separate archived centralized 2D/3D HAR | Final-epoch accuracy 89.7% / 86.5%; test-selected best 90.6% / 86.8% | `results/cent_{2d,3d}_15cls_results.json`; archived class diagnostics use the selected checkpoints |
| Raspberry Pi 5 measurements | Cold compute 38.9 frames/s; final-minute mean 26.9 frames/s after thermal throttling | [Recorded logs, bound source and analyzer](pi5_benchmark_20261024/README.md) |
| Deployed model | Older centralized 15-class 2D model, distinct from the new matched centralized/FedBN references | `checkpoints/stgcnpp_medical15_cent2d.onnx` and `.onnx.data` |

### Controlled comparison

Mean +/- sample standard deviation across model seeds 42, 43 and 44:

| Method | Test accuracy, % | Test macro-F1, % | Worst-client test accuracy, % |
|---|---:|---:|---:|
| Centralized | 92.02 +/- 0.55 | 92.03 +/- 0.55 | 90.80 +/- 0.73 |
| Local only | 82.63 +/- 0.39 | 82.66 +/- 0.34 | 76.73 +/- 2.26 |
| FedAvg | 78.44 +/- 1.52 | 77.78 +/- 1.87 | 73.16 +/- 2.23 |
| FedBN | 83.47 +/- 0.85 | 83.31 +/- 1.10 | 81.26 +/- 2.22 |
| FedProx | 78.00 +/- 2.84 | 77.23 +/- 3.21 | 72.83 +/- 3.57 |

Five clients use one simulated Dirichlet label partition (alpha 0.5, partition seed 42). Training/development/test performers are disjoint. FedBN retains all client BN state without pooled calibration or warm start. Every method uses paired fresh initialization, matched training exposure and development-only checkpoint selection. Model seeds do not replicate client partitions. Label-based allocation defines benchmark cohorts, not a deployable client router. The historically explored NTU benchmark is not an untouched external validation set.

The result supports local normalization under the evaluated label skew. FedBN's pooled improvement over local-only learning is modest; its worst-client improvement is larger. Neither establishes a universal benefit of federation or a new algorithm.

To verify the released metrics without training or downloading input tensors:

```bash
# In a Python environment with NumPy installed; recorded version: 2.4.4
cd controlled_har_fl
python verify_release.py
```

The [package README](controlled_har_fl/README.md) gives implementation tests, environment versions, data preparation and new-run commands. Raw NTU annotations/prepared input tensors are not redistributed. Public sequence identifiers, prediction logits and selected model weights are included. The manifest identifies exactly which artifacts are published.

## Raspberry Pi measurement scope

The Pi result times a compute loop over pre-decoded frames, with MoveNet SinglePose Thunder int8, four threads and graph-model inference every six frames. It excludes capture, network delivery and alert handling, and uses the older centralized deployment checkpoint. It is not a measured end-to-end alarm rate, an energy result, or a benchmark of the new FedBN checkpoints. The source files actually hashed on the device are in [pi5_benchmark_20261024/](pi5_benchmark_20261024/README.md).

The top-level `edge/` directory remains the older archived harness; use the versioned measurement package when interpreting the Pi numbers.

## Historical FL experiments

The archived `results/phase1_baselines.json` and `notebooks/train_colab_baselines.ipynb` describe a **10-class**, single-run experiment. They do not produce the controlled table above:

| Historical result key | Actual procedure |
|---|---|
| `fedavg` | Global averaging after a pooled warm start, with pooled BN recalibration after each round. |
| `fedbn` | Partial name-based BN exclusion plus pooled calibration; it misses most BN modules and discards client BN state. It is not canonical FedBN. |
| `cluster_fedbn` | The same partial exclusion, with equal averaging of cluster means; unequal clusters give unequal per-client influence. |

Historical `worst_client_acc` values are training-pass minima, and historical `best_acc` values are maxima over test probes. The 10-round notebook screen mixes estimators and does not establish an algorithm ranking. These files remain for provenance and have been superseded as the main FL evidence.

Top-level `train_centralized.py`, `train_federated.py` and `prepare_medical_data.py` are the older development scripts. The Flower script is a separate implementation, not the code path for either the notebook results or the new corrected sequential simulation. Use the versioned package for the current controlled comparison.

## Model

The graph model is a custom ST-GCN-style implementation inspired by [PYSKL/ST-GCN++](https://github.com/kennymckormick/pyskl), not a faithful upstream configuration. The controlled run preserves its directed graph and preprocessing. The 15-class 2D backbone has 444,507 trainable parameters.

## The 15 medical classes

The NTU RGB+D 60 action IDs below match `MEDICAL_CLASS_MAP` in `prepare_medical_data.py`.

| NTU ID | Class | Interface level |
|---|---|---|
| A043 | falling | CRITICAL |
| A042 | staggering | HIGH |
| A048 | nausea/vomiting | HIGH |
| A044 | touch head | MEDIUM |
| A045 | touch chest | MEDIUM |
| A046 | touch back | MEDIUM |
| A047 | touch neck | MEDIUM |
| A041 | sneeze/cough | LOW |
| A009 | standing up | NORMAL |
| A008 | sitting down | NORMAL |
| A059 | walking towards | NORMAL |
| A060 | walking apart | NORMAL |
| A001 | drinking | NORMAL |
| A002 | eating | NORMAL |
| A028 | phone call | NORMAL |

Only CRITICAL (falling) triggers the fall alarm. The levels are interface labels, not clinical severity annotations.

## Other evidence and exploratory work

The PAMAP2 context-alerting analysis and physiology evaluation are not part of this HAR release. Their evidence is available from the authors on reasonable request. `vitals_fl/`, the synthetic-IMU fusion notebook and the historical `docs/` reports are separate exploratory material; their older claims must not be substituted for the current manuscript evidence.

A follow-up study of local classifier heads and adapted layer-wise shrinking began on 25 October 2026. It is still development-stage research and is not included in the paper's results above.

## License

Apache-2.0 (see `LICENSE`). Dataset annotations and externally obtained pose models retain their own terms.
