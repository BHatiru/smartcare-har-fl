# Smart-Care HAR-FL: skeleton activity recognition and federated-learning artifacts

This repository holds the artifacts for the human activity recognition (HAR) and federated-learning (FL) experiments in the following manuscript:

> **Activity-Aware Federated Multimodal IoMT for Continuous Patient Monitoring**
> T. Tursynbekov, B. Bodaubay, B. Ongdassynov, T. Bayan, A. Yazici (2026 manuscript).
> School of Computing and Artificial Intelligence (SCAI), Nazarbayev University, Astana, Kazakhstan. T. Bayan is also affiliated with the University of Manchester, Manchester, United Kingdom.
> Research project SEDS2024025, "Smart-Care: Innovative Multi-Sensor Technology for Elderly and Disabled Health Management".

It contains:

- the HAR training code;
- the archived result files;
- selected checkpoints, including the ONNX model used by the edge node;
- the notebooks in which the archived experiments ran;
- an edge benchmark harness.

The context-alerting (PAMAP2) analysis, the physiology evaluation and the patient-history controls are not in this repository. They are available from the authors on reasonable request.

## How the paper names the archived FL procedures

The archived 50-round FL experiment uses a **10-class** task. It ran as sequential PyTorch simulations in `notebooks/train_colab_baselines.ipynb`. Its result keys do not name canonical algorithms. The paper uses these names instead:

| Key in `results/phase1_baselines.json` | Name in the paper | What the code does |
|---|---|---|
| `fedavg` | Global averaging + calibration | Averages client updates after a 2-epoch pooled warm start. After every round, BN statistics are recalibrated on up to 30×128 pooled training examples. |
| `fedbn` | Partial-BN exclusion + calibration | Skips state keys whose names contain `bn`. This misses 72 of the model's 79 BN modules and keeps no client BN state across rounds. **It is not canonical FedBN.** |
| `cluster_fedbn` | Cluster-balanced averaging + calibration | Applies the same partial-BN exclusion, averages within classifier-weight clusters, then averages the cluster means equally. In the archived four-client/one-client split, the singleton therefore receives half the update. |

Two further points about the result files:

- `worst_client_acc` is the lowest client **training-pass** accuracy at round 50. It is not held-out client performance or a fairness measure.
- `best_acc` values are maxima over per-round **test** evaluations, so they are test-selected. The retained `sampled_history` contains rounds 1, 10, 20, 30, 40 and 50.

The 10-round optimization screen (proximal, global/personal and representation/head variants) ran in `notebooks/train_colab_methods_ran.ipynb`. Its rows use different estimators. Its round-10 contrast between global averaging and partial-BN exclusion has the opposite sign to the 50-round run, so the screen does not rank the procedures.

## Paper item → artifact map

| Paper item | Reported values | Artifact |
|---|---|---|
| Table II, centralized 15-class recognition | 2D final epoch 89.7% (test-selected best 90.6%); 3D final 86.5% (best 86.8%); 444,507 and 450,603 parameters | `results/cent_{2d,3d}_15cls_results.json` |
| Figs. 7–8, fall metrics | Per-class recall for both representations. For the **test-selected 2D checkpoint**, falling recall is 267/275 (97.1%), with zero false falling predictions among 3,850 non-fall clips (100% observed precision). | `results/cent_{2d,3d}_15cls_confusion_matrix.csv`, `checkpoints/cent_{2d,3d}_15cls_best.pt`, `figures/make_figs.py` |
| Table III, 10-class FL | Final 72.0 / 72.6 / 74.6% against 88.1% centralized; round-50 training-pass minima 59.9 / 59.3 / 65.0% | `results/phase1_baselines.json`, `notebooks/train_colab_baselines.ipynb` |
| Fig. 9 | Saved test probes at rounds 1, 10, 20, 30, 40 and 50 | `results/phase1_baselines.json` (`sampled_history`) |
| Table IV, 10-round screen | Best, round-10 and time-ratio values | Printed outputs in `notebooks/train_colab_methods_ran.ipynb` |
| Edge node model | Centrally trained 15-class 2D checkpoint | `checkpoints/stgcnpp_medical15_cent2d.onnx` and its `.onnx.data` file. The file name is historical; the model is the custom graph model described below. |

The paper reports **no device latency, throughput or energy measurements**. `edge/bench.py` is archived benchmark-harness source, not a validated runnable package: it requires separate pose models and runtime dependencies, its default model/label paths refer to the development tree, and its packaged copy contains an unresolved `_ROOT` reference. It is not evidence of any particular frame rate.

## Model

`models/stgcn.py` is a custom ST-GCN-style implementation inspired by ST-GCN++ ([pyskl](https://github.com/kennymckormick/pyskl), Apache-2.0). Its graph construction is part of the evaluated model. It inserts one directed copy of each skeleton edge and initializes the learned matrices from the adjacency, which differs from the reference bidirectional graph. The model is therefore not a faithful reproduction of the published ST-GCN++ configuration.

## Running the provided scripts

The commands below create **new** runs on the 15-class subset. They do not regenerate the archived 10-class FL experiment, whose per-client sample lists were not saved.

```bash
pip install -r requirements.txt

# 1. Data: downloads the public PYSKL NTU RGB+D 60 skeleton annotations and builds the
#    15-class subset plus label-skewed Dirichlet client shards (alpha = 0.5)
python prepare_medical_data.py --num-clients 5 --alpha 0.5 --seed 42

# 2. Centralized 15-class training (2D and 3D)
python train_centralized.py --data-dir data/fed_medical_2d/centralized --graph coco --tag cent_2d_15cls
python train_centralized.py --data-dir data/fed_medical_3d/centralized --graph ntu  --tag cent_3d_15cls

# 3. A new Flower simulation on the 15-class data (note --num-classes; the script defaults to 10)
# The simulation extra supplies Ray, which requirements.txt alone does not install.
python -m pip install "flwr[simulation]>=1.7"
python train_federated.py --data-dir data/fed_medical_2d --graph coco --num-classes 15 \
    --mode fedavg --num-rounds 50

# 4. HAR plots from the archived results
python figures/make_figs.py
```

The centralized commands default to CUDA; add `--device cpu` to run them on a CPU.

`train_federated.py` is a separate Flower implementation (`flwr`). It is not the code path of the archived notebook experiments. Its `--mode` names (`fedbn`, `cluster`, `fedprox`) are command-line labels. Inspect its client-state and normalization handling before treating a run as a canonical baseline.

The Dirichlet split assigns clips to clients **by label only**, so one performer's clips can reach several clients. The partition models label skew, not person-owned devices.

## Limitations of the archived runs

- There is one archived run per configuration and no development split. The FL notebooks set seed 42; the centralized result files do not record a training seed.
- The 10-class FL task is distinct from the 15-class centralized task, and the centralized and federated training budgets differ.
- The FL procedures use pooled warm starts and pooled BN recalibration, so they are not strictly data-isolated.
- FL histories retain rounds 1, 10, 20, 30, 40 and 50; intermediate per-round records are unavailable.
- Clip-level NTU metrics do not measure online alarm sensitivity, latency or false alarms per hour.

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

## Exploratory material not used as paper evidence

- `vitals_fl/` is a temporal vital-sign FL pilot on a different task and model.
- `notebooks/train_colab_multimodal_ran.ipynb` and `docs/report-multimodal-fusion.md` are a synthetic-IMU fusion pilot.
- `docs/` holds historical working notes. They use older method names such as "FedBN" and "worst-client". Where they differ from the manuscript, the manuscript's definitions above apply.

The vitals-FL and synthetic-IMU pilot results are not used as evidence in this manuscript. Historical notes may repeat HAR/FL results that are reported, under the revised definitions above.

## License

Apache-2.0 (see `LICENSE`). The NTU RGB+D 60 annotations are the public PYSKL releases and remain under their own terms.
