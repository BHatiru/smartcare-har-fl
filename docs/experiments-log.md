# Experiment Log

Running record of all experiments for the pyskl-research federated learning project.
Entries are appended chronologically. Each entry follows the template in `.github/skills/research-loop/references/experiment-log-template.md`.

---

## Research Roadmap — 2026-03-24

### Phase 1: Establish Baselines
- [x] Centralized STGCN++ on 2D medical subset (no FL)
- [x] FedAvg baseline (5 clients, α=0.5, 50 rounds)
- [x] FedBN baseline (same, BN kept local)
- [x] Clustered FedBN baseline (2 clusters, re-cluster every 5 rounds)
- [x] Record all metrics in `fl-baselines.md`

### Phase 2: 2D → 3D Skeleton Transition
- [ ] Adapt FL model for NTU RGB+D 3D Kinect data (25 joints instead of 17)
- [ ] Update `demo1_fed_skeleton/models/stgcn.py` graph for NTU 25-joint layout
- [ ] Update `prepare_clients.py` for 3D data loading
- [ ] Re-run all baselines (centralized, FedAvg, FedBN) on 3D data
- [ ] Compare 2D vs 3D — does 3D improve medical action recognition?

### Phase 3: Advanced FL Methods (Quick Scan — 10 rounds each)
- [x] Literature review via scout: FedCFE paper comparison methods analyzed
- [x] Implement FedProx (proximal term μ/2‖w-w_g‖², μ=0.01)
- [x] Implement Ditto (personalized model + L2 reg toward global, λ=0.1)
- [x] Implement FedRep (shared body + local head, alternating training)
- [x] Implement APFL (adaptive α mixing global/local, α₀=0.5)
- [x] Run quick scan: 10 rounds each, compare all 6 methods (5 of 6 complete — APFL pending)
- [ ] Deep dive on top-performing method — FedProx (50 rounds)

### Phase 4: Multimodal Exploration
- [x] Scout datasets: UTD-MHAD, SmartFallMM, MMAct, C-MHAD, K-Fall
- [x] Design synthetic IMU from NTU 3D skeleton (discrete 2nd derivative)
- [x] Implement Late Fusion: STGCN++ (skeleton) + 1D-CNN (IMU) → concat → FC
- [x] Run quick centralized comparison: skeleton-only vs late fusion → **Late fusion -2.0 pp (hurt)**
- [x] Create multimodal v2 notebook: Fusion-GCN + extended sensors (4 sensors, 12ch)
- [ ] Run Fusion-GCN experiment (virtual graph nodes)
- [ ] Evaluate if graph-level fusion adds value vs late fusion

---

## Project Setup — 2026-03-24

**Status:** Agent system initialized. Infrastructure ready (data shards, Flower strategy, ONNX export).

**Current state:**
- 2D STGCN++ trained with FedAvg (50 rounds) — outputs exist in `demo1_fed_skeleton/outputs/`
- ONNX pipeline working at 15-20 FPS on CPU
- 3D pipeline exists in pyskl configs but not used in FL yet
- FL modes implemented: FedAvg, FedBN, Clustered FedBN

**Next:** Run Phase 1 baselines systematically and record all metrics.

---

## Phase 1: Baselines Notebook Created — 2026-04-01

**File:** `demo1_fed_skeleton/train_colab_baselines.ipynb` (29 cells)

**Experiments planned:**
1. **Centralized** — 20 epochs on all combined data (upper bound)
2. **FedAvg** — 50 rounds, all params aggregated
3. **FedBN** — 50 rounds, BN params kept local
4. **Clustered FedBN** — 50 rounds, 2 clusters, re-cluster every 5 rounds

**Shared config:** SEED=42, 5 clients, Dirichlet α=0.5, STGCN++ (64ch, 6 blocks, ~443K params), SGD lr=0.01, warm-start 2 epochs, LR schedule (×0.1 @R20, ×0.01 @R35)

**Status:** COMPLETE. Results extracted from notebook cell outputs.

---

## Phase 1: Baseline Results — 2026-04-05

**Notebook:** `demo1_fed_skeleton/train_colab_baselines.ipynb` (executed on Colab T4)
**Runtime:** ~3.8 hours total (25 min centralized, ~75 min per FL method)
**Results file:** `demo1_fed_skeleton/outputs/phase1_baselines.json`

| Method | Best Acc | Final Acc | Gap vs Centralized |
|--------|----------|-----------|--------------------|
| Centralized | **0.8949** | 0.8807 | — (ceiling) |
| FedAvg | 0.7719 | 0.7195 | -0.1230 |
| FedBN | 0.7636 | 0.7264 | -0.1313 |
| Clustered FedBN | 0.7683 | 0.7461 | -0.1266 |

**Key observations:**
- Large FL-to-centralized gap (~12-13 pp) — significant room for improvement
- All FL methods peak around R20, then degrade (overfitting / LR schedule issue)
- FedAvg slightly outperforms FedBN (surprising — BN local didn't help here)
- Clustered FedBN best final accuracy (0.7461) but cluster collapsed to [0,0,0,0,1] after R10
- Worst-client accuracy: Clustered FedBN most stable (~0.65), FedAvg worst (~0.59)
- Warm-start only reaches 48.7% after 2 epochs — very early in convergence

**Data stats:** Train=6695, Test=2749, Client sizes=[1798, 1078, 1648, 1253, 918]

**Missing:** Per-class F1, confusion matrix, fall detection recall (Drive save failed — full per-round histories lost, only sampled at R1/10/20/30/40/50)

**Next:** Analyze results with @analyst, investigate accuracy degradation after R20, plan Phase 2.

---

## Phase 3+4: Methods Scan Notebook Created — 2026-04-05

**File:** `demo1_fed_skeleton/train_colab_methods.ipynb` (32 cells)
**Source:** FedCFE paper (IEEE IoT Journal 2026) comparison methods

### Part A — FL Methods Quick Scan (10 rounds each)

| # | Method | Core Idea | Hyperparams |
|---|--------|-----------|-------------|
| 1 | FedAvg | Standard averaging | baseline |
| 2 | FedBN | BN kept local | baseline |
| 3 | **FedProx** | Proximal term μ/2‖w-w_g‖² | μ=0.01 |
| 4 | **Ditto** | Personalized model + L2 reg toward global | λ=0.1 |
| 5 | **FedRep** | Shared body + local head (alternating) | head_ep=1, body_ep=1 |
| 6 | **APFL** | Adaptive α mixing global/local | α₀=0.5 |

**Quick config:** 10 rounds (not 50), same data partition, warm-start 2 epochs

### Part B — Multimodal Exploration

- Synthetic IMU from NTU 3D skeleton (25 joints, physics-based 2nd derivative)
- 2 virtual sensors: right wrist (joint 11) + waist (joint 0) → 6 channels
- Low-pass Butterworth filter + Gaussian noise
- Late Fusion: STGCN++ features (256-d) + 1D-CNN IMU features (128-d) → FC(384, 10)

**Status:** Ready to run on Colab T4. Estimated ~15 min per FL method, ~10 min for multimodal.

---

## Phase 4: Multimodal v1 Results — 2026-04-05

**Notebook:** `demo1_fed_skeleton/train_colab_multimodal_ran.ipynb` (executed on Colab T4)
**File:** `demo1_fed_skeleton/train_colab_multimodal.ipynb` (clean source)

| Method | Best Acc | Δ vs Skel-only |
|--------|----------|----------------|
| Skeleton-only (STGCN++, 10ep centralized) | **0.793** | — |
| Late Fusion (2 sensors, 6ch, STGCN++ + 1D-CNN) | 0.773 | **-2.0 pp** |

**Key findings:**
- Late Fusion with synthetic IMU **hurt** accuracy by -2.0 pp
- Synthetic IMU is derived from same skeleton positions (2nd derivative) → information is redundant
- Extra model complexity (~500K vs ~443K) without new information → worse generalization
- IMU visualization shows plausible accelerometer patterns but they're a deterministic function of skeleton

**Conclusion:** Late fusion with synthetic IMU is not viable. Next: test Fusion-GCN (graph-level integration) and extended sensors.

---

## Phase 4: Multimodal v2 Notebook Created — 2026-04-05

**File:** `demo1_fed_skeleton/train_colab_multimodal_v2.ipynb` (28 cells)

| # | Method | Key Change | Hypothesis |
|---|--------|-----------|-----------|
| A | Skeleton-only (repeat) | Baseline on same data subset | ~79% |
| B | Late Fusion 12ch | 4 sensors (wrist×2, waist, ankle) → 12ch | More coverage, marginal ↑ |
| C | **Fusion-GCN** | IMU as 4 virtual nodes in 21-node skeleton graph | Best — cross-modal GCN |

**Fusion-GCN architecture:**
- Extends COCO-17 graph with 4 virtual IMU nodes → 21-node graph
- Virtual nodes connected to anatomical joints: node 17→joint 11, node 18→joint 0, node 19→joint 7, node 20→joint 15
- Same STGCN++ architecture, just larger graph — GCN learns cross-modal spatial relationships
- Reference: Duhme et al. "Fusion-GCN" (GCPR 2021) reported +12.4% F1

**Status:** Ready to run on Colab T4. Estimated ~30 min total.

---

## Phase 3: PFL Quick-Scan Results — 2026-04-05

**Notebook:** `demo1_fed_skeleton/train_colab_methods_ran.ipynb` (executed on Colab T4)
**Source:** `demo1_fed_skeleton/train_colab_methods.ipynb` (clean source)
**Config:** 10 rounds, 5 clients, Dirichlet α=0.5, warm-start 2 epochs, SGD lr=0.01

| Method | Best Acc | R5 Acc | R10 Acc | Time |
|--------|----------|--------|---------|------|
| FedAvg | 0.7083 | 0.6726 | 0.6570 | 843 s |
| FedBN | 0.7188 | 0.6715 | 0.7126 | 841 s |
| **FedProx** | **0.7341** | 0.6493 | **0.7341** | 883 s |
| Ditto | 0.7134 | — | 0.7050 (global) | 1,882 s |
| FedRep | 0.6405 | 0.6091 | 0.6405 | 1,348 s |
| APFL | — | — | — | not run |

**Ditto additional metrics:** personal_mean=0.5756 at R10, global=0.7050

**Key findings:**
- **FedProx is the clear winner** (+2.58 pp over FedAvg, +1.53 pp over FedBN)
- FedProx still improving at R10 while FedAvg already declining (65.70% at R10)
- Ditto achieves 71.34% but at 2.2× compute cost — personal models need more rounds
- FedRep underperforms (64.05%) — 10 rounds likely insufficient for representation–head decoupling
- APFL not run due to time constraints

**Next:** Full 50-round FedProx run, APFL quick scan, per-class F1 analysis

---

## Combined Emergency System — Architecture Design — 2026-04-08

### Overview

The full emergency prevention/notification system combines **two federated subsystems**:

| Subsystem | Input | Output | Model | FL Status |
|-----------|-------|--------|-------|-----------|
| **A: Skeleton HAR** | Video → skeleton keypoints | 10 medical actions | STGCN++ (443K) | FedProx best (73.4%) |
| **B: Vitals Anomaly** | Physiological sensors (HR, SpO2, etc.) | 6 vitals conditions | MLP (128→64) | Centralized ~79%, FL ~53% |

### Combined Dataset (`ntu_action_dataset/`)

| File | Shape | Contents |
|------|-------|----------|
| `sequences.npy` | (9462, 64, 25, 3) | Raw 3D NTU-25 skeleton keypoints |
| `labels.npy` | (9462,) | Binary: medical=1, daily=0 |
| `features.parquet` | 9462 × 370 | Per-sequence derived features (speeds, accels, COM) |
| `integrated_v2/*.parquet` | ~1.2M rows × 77 cols | **Paired** vitals + NTU data: 3,600 sequences, 300 subjects |
| `timeseries_chunk_*.parquet` | ~1.5M rows × 83 cols | Per-frame 25-joint 3D coords |

All 3,600 integrated sequences have matching raw skeleton data in sequences.npy.

### Teammate's Vitals FL Code (`federated_v2/`)

Three implementations: FedAvg, FedAvgM, FedProx — all train HealthModel (MLP) on 5 vitals features.
Saved models: centralized, fed-10, fed-20, fed-50 clients.

**Issues identified:**
- Point-wise MLP (no temporal modeling) — misses time-series patterns
- IID split (`np.array_split`) instead of non-IID Dirichlet
- High NaN rates: etCO2 83%, awRR 78% — need explicit masking
- FedProx stuck at ~35% → model/setup needs debugging

### Fusion Architecture Options (Ranked by feasibility)

**Option 1: Decision-Level Fusion (chosen for demo)**
- Each subsystem predicts independently
- Rule-based or lightweight classifier combines: (action_class, action_conf, vitals_class, vitals_conf) → emergency_level
- Pro: subsystems stay independent, easiest to deploy
- Con: no cross-modal learning

**Option 2: Embedding-Level Fusion**
- Extract embeddings from both backbones (STGCN++ 256-d, vitals encoder 128-d)
- Train a fusion head on integrated_v2 paired data
- Pro: learns cross-modal correlations
- Requires: synchronized inference, paired training data (have it)

**Option 3: Semantic Alignment via Shared Latent Space**
- Project both embeddings into shared emergency-severity space
- Contrastive learning on integrated_v2 pairs
- Pro: works when only one modality is available (graceful degradation)
- Most research-novel but highest implementation effort

### Demo Plan

`demo_combined/demo_emergency_system.py` — self-contained inference demo:
1. Train quick classifiers on integrated_v2 (seconds on tabular data)
2. Decision-level fusion with emergency scoring
3. Patient monitoring dashboard: vitals time series + skeleton pose + combined alert
4. Produces PNG figures and console output