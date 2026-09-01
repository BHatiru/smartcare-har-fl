# Smart-Care HAR-FL — Federated Skeleton-Based Medical Activity Recognition

Artifact repository for the HAR and federated-learning experiments in:

> **Federated Multisensor Framework for Vital-Sign Anomaly Detection and Human
> Activity Recognition in Continuous Patient Monitoring**
> T. Tursynbekov, B. Bodaubay, B. Ongdassynov — Nazarbayev University.
> Grant SEDS2024025 "Smart-Care: Multi-Sensor Technology for Elderly and
> Disabled Health Management" (PI Prof. Adnan Yazici).

Every HAR-side number in the paper traces to a file in this repository: the
training code, the archived results it produced, the checkpoints, and the
notebooks that are the original experiment record. All experiments are
reproducible from this repo alone — the data-prep script auto-downloads the
public NTU RGB+D 60 skeleton pickles.

## Layout

```
prepare_medical_data.py   NTU-60 → 15-class medical subset (2D COCO-17 + 3D NTU-25),
                          train/test NPZ + Dirichlet non-IID client shards
prepare_clients.py        standalone client-shard partitioner
train_centralized.py      centralized ST-GCN++ training (80-epoch protocol)
train_federated.py        Flower FL: fedavg | fedbn | cluster | fedprox
export_onnx.py            checkpoint → ONNX for edge deployment
models/stgcn.py           self-contained ST-GCN++ (no mmcv/pyskl dependency)
fl/                       Flower client + clustered-aggregation strategy
notebooks/                the Colab notebooks the experiments actually ran in
results/                  archived result JSONs / confusion CSVs / training curves
checkpoints/              trained .pt weights + exported ONNX (incl. deployed model)
vitals_fl/                temporal vitals model FL experiment (cross-modality check)
edge/                     edge benchmark harness + pose-backend ablation
figures/make_figs.py      regenerates the paper figures from results/
docs/                     experiment log + written reports
```

## Reproduce

```bash
pip install -r requirements.txt

# 1. Data: downloads the public NTU-60 skeleton pickles (~4.5 GB) and builds
#    the 15-class medical subset + 5 non-IID client shards (Dirichlet α=0.5)
python prepare_medical_data.py --num-clients 5 --alpha 0.5 --seed 42

# 2. Centralized 15-class training (Table: 2D vs 3D)
python train_centralized.py --data-dir data/fed_medical_2d/centralized --graph coco --tag cent_2d_15cls
python train_centralized.py --data-dir data/fed_medical_3d/centralized --graph ntu  --tag cent_3d_15cls

# 3. Federated training (Table: FL baselines)
python train_federated.py --data-dir data/fed_medical_2d --graph coco \
    --mode fedavg --num-rounds 50    # also: fedbn | cluster | fedprox

# 4. Paper figures
python figures/make_figs.py
```

The archived runs used a Colab T4 (~16 min/80-epoch centralized run, ~1.5 h/50-round
FL run). CPU works but is slow.

## Paper claim → artifact map

| Paper item | Claim | Backing artifact |
|---|---|---|
| Centralized HAR table | 2D COCO-17 **90.6%** vs 3D NTU-25 86.8% (15 classes); fall **97.1% recall at 100% precision** | `results/cent_{2d,3d}_15cls_results.json`, `results/cent_{2d,3d}_15cls_confusion_matrix.csv`, `checkpoints/cent_{2d,3d}_15cls_best.pt` |
| Per-class figure, confusion figure | per-class accuracies, error structure | `figures/make_figs.py` renders both directly from the files above |
| FL baselines table (50 rounds) | FedAvg 77.2 / FedBN 76.4 / Clustered FedBN 76.8 best acc; worst-client 65.0% for clustered | `results/phase1_baselines.json`, `notebooks/train_colab_baselines.ipynb` |
| FL convergence figure | sampled accuracy vs rounds | `figures/make_figs.py` from `results/phase1_baselines.json` |
| Personalized-FL screening table (10 rounds) | FedProx 73.4% best, 1.0× cost; Ditto 2.2× cost; FedRep underperforms | `docs/experiments-log.md` (2026-04-05 entry), `checkpoints/fedprox_2d_*c_best.pt`, `notebooks/train_colab_methods_ran.ipynb` |
| Edge inference prose | RTMPose path ≈52 ms/frame (~19 FPS laptop CPU); MoveNet MultiPose ≈11 ms; Pi 5 ~25 FPS | `edge/bench.py` (same harness runs unchanged on laptop and Pi 5) |
| Pose-backend quantization finding (URFD) | MoveNet Thunder int8 preserves fall evidence (peak P(fall) 99–100%); Lightning loses it | `edge/pose_backends.py` header notes, `edge/pose_compare.png` |
| Deployed edge model | ST-GCN++ 15-class ONNX running on the Raspberry Pi 5 | `checkpoints/stgcnpp_medical15_cent2d.onnx` (+ required `.onnx.data` sidecar) |
| Vitals temporal FL (cross-modality contrast) | FedBN 86.8% ≈ centralized 87.2% on vitals — opposite of the skeleton case | `vitals_fl/results.json`, `vitals_fl/train_vitals_temporal.py` |
| Multimodal synthetic-IMU fusion (negative result) | late fusion with synthetic IMU: 79.3% → 77.3% (−2.0 pp) | `notebooks/train_colab_multimodal_ran.ipynb`, `docs/report-multimodal-fusion.md` |

## Known limitations of the archived runs

Stated here so the record is honest; the paper's limitations section mirrors them.

- All runs use a **single seed (42)**.
- The 50-round FL baselines were run on the earlier **10-class protocol**, while the
  centralized comparison uses the 15-class protocol; per-round FL histories were lost
  to a storage failure, so only accuracies sampled every 10 rounds survive
  (`phase1_baselines.json` → `sampled_history`).
- The personalized-FL comparison is a **10-round screening**, not a full-length run.
- Clustered FedBN's clusters collapsed to a single split `[0,0,0,0,1]` after round 10.
- Edge latency numbers are measured with `edge/bench.py`; a consolidated
  machine-readable suite output (laptop + Pi 5) is pending and will be added
  under `results/`.

## The 15 medical classes

NTU-60 ids → class: A043 falling · A044 staggering · A049 nausea/vomiting ·
A039–A042 touch head/chest/back/neck · A006 sneeze/cough · A008 standing up ·
A009 sitting down · A054 walking towards · A055 walking apart · A001 drinking ·
A002 eating · A017 phone call. At inference these map to four severity tiers
(emergency / pain / symptom / normal) that drive alerting.

## Related

- The live edge deployment (Raspberry Pi 5: camera → MoveNet MultiPose →
  ST-GCN++ → debounced alerts → dashboard) and the demo apps live in the
  project's development repository; this repository holds the reviewed
  experimental record.
- ST-GCN++ architecture follows [pyskl](https://github.com/kennymckormick/pyskl)
  (Apache-2.0); `models/stgcn.py` is a self-contained reimplementation.
- NTU RGB+D 60 skeleton pickles are the public releases from OpenMMLab/pyskl.

## License

Apache-2.0 (see `LICENSE`).
