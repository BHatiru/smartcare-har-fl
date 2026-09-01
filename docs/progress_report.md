# Research Progress Report — April 5, 2026

## Executive Summary

We are building a **privacy-preserving medical action recognition system** using federated learning on skeleton data, targeting patient monitoring and emergency notification in hospital settings. Phase 1 baselines are complete: centralized STGCN++ achieves **89.5% accuracy**, while the best federated method (FedAvg) reaches **77.2%** — a 12.3 pp gap that motivates our ongoing work. We have completed literature surveys and implemented four advanced personalized FL methods (FedProx, Ditto, FedRep, APFL) and a multimodal skeleton–IMU late fusion model, both ready to run on Colab T4.

---

## 1. Project Overview

### 1.1 Problem Statement

Skeleton-based human action recognition (HAR) is increasingly relevant for medical monitoring and emergency notification — recognising actions such as falling, staggering, or pain-related gestures that may indicate a medical emergency. However, deploying such systems across hospitals requires training on patient data that cannot be centralized due to privacy regulations. **Federated learning** (FL) enables collaborative model training while keeping raw skeleton data on each hospital's local infrastructure.

### 1.2 System Architecture

```
 FL Training (Colab T4)                  Real-Time Inference (CPU)
┌──────────────────────────┐            ┌──────────────────────────┐
│  NTU RGB+D 60 dataset    │            │  Webcam / Video input    │
│  ↓ filter 10 classes     │            │  ↓                       │
│  5 hospital clients      │   ONNX     │  YOLOX → RTMPose → STGCN │
│  (non-IID, Dirichlet)    │ ────────►  │  ↓                       │
│  STGCN++ (443K params)   │  export    │  10 medical actions      │
│  50 rounds FL training   │            │  15-20 FPS, CPU only     │
└──────────────────────────┘            └──────────────────────────┘
```

### 1.3 Technical Setup

| Component | Detail |
|-----------|--------|
| **Model** | STGCN++ — 6 GCN blocks, base_channels=64, ~443K params |
| **Skeleton** | COCO-17 (2D, 17 joints), input `(N, 2, 100, 17, 3)` |
| **Dataset** | NTU RGB+D 60 medical subset: train=6,695, test=2,749 |
| **FL Framework** | Flower (simulation mode), 5 clients, Dirichlet α=0.5 |
| **Training** | SGD lr=0.01, warm-start 2 epochs, LR schedule ×0.1@R20, ×0.01@R35 |
| **Hardware** | Google Colab T4 GPU (~75 min per FL method at 50 rounds) |

### 1.4 Medical Classes (10)

| # | Medical Actions (7) | # | Normal Context (3) |
|---|---------------------|---|---------------------|
| 0 | **Falling** | 7 | Standing up |
| 1 | Staggering | 8 | Sitting down |
| 2 | Touch head (headache) | 9 | Walking towards |
| 3 | Touch chest (heart pain) | | |
| 4 | Touch back (backache) | | |
| 5 | Touch neck (neckache) | | |
| 6 | Nausea / vomiting | | |

---

## 2. Phase 1: Baseline Experiments (COMPLETED)

**Notebook:** `train_colab_baselines.ipynb` (29 cells, executed on Colab T4)
**Runtime:** ~3.8 hours total
**Results:** `outputs/phase1_baselines.json`

### 2.1 Results Summary

| Method | Best Acc | Final Acc (R50) | Δ vs Centralized | Worst Client | Training Time |
|--------|----------|-----------------|-------------------|--------------|---------------|
| **Centralized** | **89.49%** | **88.07%** | — (ceiling) | — | 25 min |
| FedAvg | 77.19% | 71.95% | −12.30 pp | 59.91% | 76 min |
| FedBN | 76.36% | 72.64% | −13.13 pp | 59.26% | 76 min |
| Clustered FedBN | 76.83% | 74.61% | −12.66 pp | 65.03% | 76 min |

### 2.2 Client Data Distribution

Non-IID partition via Dirichlet α=0.5 across 5 clients:

| Client | Samples | Share |
|--------|---------|-------|
| 0 | 1,798 | 26.9% |
| 1 | 1,078 | 16.1% |
| 2 | 1,648 | 24.6% |
| 3 | 1,253 | 18.7% |
| 4 | 918 | 13.7% |

![Data Distribution](outputs/medical_data_distribution.png)

### 2.3 Training Dynamics

![Baseline Comparison](outputs/baseline_comparison.png)

**Convergence patterns (sampled at R1, R10, R20, R30, R40, R50):**

| Method | R1 | R10 | R20 | R30 | R40 | R50 |
|--------|----|-----|-----|-----|-----|-----|
| FedAvg | 42.89% | 69.44% | **77.19%** | 73.77% | 71.95% | 71.95% |
| FedBN | 41.72% | 65.88% | 75.66% | 74.06% | 72.57% | 72.64% |
| Clustered FedBN | 44.20% | 53.91% | 63.33% | **76.54%** | 74.61% | 74.61% |

### 2.4 Key Observations

1. **12–13 pp FL gap to centralized** — substantial room for improvement, consistent with literature on non-IID FL.
2. **Accuracy degradation after R20** — all FL methods peak early then decline, suggesting client drift and/or LR schedule mismatch. FedAvg peaks at R20, Clustered FedBN peaks later at R30.
3. **FedAvg slightly outperforms FedBN** — surprising, since keeping BN local is expected to help under distribution shift. Possible explanation: with only 5 clients and moderate heterogeneity (α=0.5), global BN statistics are still informative.
4. **Clustered FedBN has best final accuracy** (74.61%) and best fairness (worst client 65.03%), but clustering collapsed to [0,0,0,0,1] after R10 — client 4 was isolated while all others formed a single group.
5. **Warm-start only reaches 48.7%** after 2 epochs — models are still early in convergence when FL begins.

### 2.5 Missing Metrics

- Per-class F1 scores and confusion matrix (Colab Drive mount failure lost full per-round histories; only sampled at 6 checkpoints)
- Per-class recall/precision for safety-critical actions (falling, staggering) — critical for medical deployment

---

## 3. Phase 3: Personalized FL Methods (RESEARCH COMPLETE — NOT YET RUN)

**Report:** `reports/Report - Personalized Federated Learning Methods.md`
**Notebook:** `train_colab_methods.ipynb` (32 cells, ready to execute)

### 3.1 Motivation

The Phase 1 baseline gap (12–13 pp) and post-R20 degradation suggest that standard FL aggregation is insufficient for heterogeneous medical data. Personalized FL (PFL) methods allow each client to maintain a partially customized model while benefiting from collaborative training.

### 3.2 Literature Survey

We surveyed 7 PFL methods from the FedCFE paper (Liu et al., IEEE IoT Journal 2026). Four were selected for implementation based on effort-to-value ratio:

| Method | Category | Core Idea | Key Hyperparameter |
|--------|----------|-----------|-------------------|
| **FedProx** | Regularization | Proximal term $\frac{\mu}{2}\|w - w_g\|^2$ constrains client drift | μ = 0.01 |
| **Ditto** | Dual model | Personal model with L2 reg toward global | λ = 0.1 |
| **FedRep** | Model splitting | Shared body (440K) + local head (2.6K) | head_ep=1, body_ep=1 |
| **APFL** | Model mixing | Adaptive α interpolation: local ↔ global | α₀ = 0.5 |

**Deprioritized:** Per-FedAvg (2× cost, MAML complexity), FedALA (too many hyperparams), FedCFE (paper's own method — cite, don't reimplement).

### 3.3 Experimental Design

- **Quick scan:** 10 rounds per method (~15 min each on T4)
- **Same partition:** 5 clients, Dirichlet α=0.5
- **Warm-start:** 2 epochs on combined data
- **Metrics:** Global test accuracy, per-client accuracy, Δ vs FedAvg baseline
- **Follow-up:** Top performer(s) → full 50-round run

### 3.4 Expected Outcomes

Based on literature:
- **FedProx:** +0.5–2 pp over FedAvg (stability gains, not necessarily peak accuracy)
- **Ditto:** +2–5 pp on personalized evaluation (dual model allows client specialization)
- **FedRep:** +1–4 pp (exploits natural STGCN++ body/head architecture split)
- **APFL:** Competitive with Ditto; auto-adaptive reduces hyperparam sensitivity

---

## 4. Phase 4: Multimodal Skeleton–IMU Fusion (RESEARCH COMPLETE — NOT YET RUN)

**Report:** `reports/Report - Multimodal Skeleton-Inertial Fusion.md`
**Notebook:** `train_colab_methods.ipynb` (Part B, same notebook)

### 4.1 Motivation

Real-world medical monitoring increasingly combines vision (skeleton) and wearable (IMU) sensors. Skeleton captures spatial joint configuration; IMU captures local dynamics and acceleration — particularly discriminative for fall-related actions.

### 4.2 Dataset Survey

| Dataset | Subjects | Actions | Skeleton | IMU | Medical Relevance |
|---------|----------|---------|----------|-----|-------------------|
| **UTD-MHAD** | 8 | 27 | Kinect 20j | Wrist accel+gyro | Fair |
| **SmartFallMM** | 40 (24 elderly) | 14 | Azure Kinect | Watch + phone | Excellent |
| **MMAct** | 40 | 37 | OpenPose 2D | 9-axis IMU | Good |
| **NTU RGB+D 60** | 40 | 60 | Kinect 25j | **None** (synthetic) | Good (our data) |

### 4.3 Synthetic IMU Generation

Since NTU RGB+D 60 lacks real IMU data, we generate **synthetic accelerometer signals** from 3D skeleton joint trajectories:

$$a(t) = \frac{p(t+1) - 2p(t) + p(t-1)}{\Delta t^2}$$

**Virtual sensors:** Right wrist (NTU joint 11) + waist (joint 0) → 6 channels (2 sensors × 3 axes).
**Post-processing:** 2nd-order Butterworth low-pass filter (~12 Hz), Gaussian noise (σ=0.05 m/s²), normalize to [-1, 1].

Validated by SmartFallMM (Debnath et al., 2025), IMUTube (Kwon et al., IMWUT 2020, 216 citations), and SynHAR (Uhlenberg et al., 2024).

### 4.4 Late Fusion Architecture

```
Skeleton (N,2,100,17,3) → STGCN++ → 256-d feature
                                          ↓
                                    Concat [384-d] → FC(384, 10) → class
                                          ↑
IMU (N, 6, 100)         → 1D-CNN  → 128-d feature
```

- STGCN++ backbone: ~443K params (unchanged)
- 1D-CNN IMU encoder: ~50K params (new)
- Fusion head: ~3.8K params (new)

### 4.5 Planned Experiment

Centralized comparison: skeleton-only vs. late fusion (10 epochs, T4). If synthetic IMU shows measurable improvement, pursue Fusion-GCN (IMU as extra graph nodes, +12.4% F1 reported in literature) and FL + multimodal combination.

---

## 5. Infrastructure & Tooling

### 5.1 Agent System

An automated research workflow was established using GitHub Copilot agent modes:
- **Researcher** — literature surveys, method analysis, experiment design
- **Analyst** — results interpretation, charts, statistical analysis
- **Reporter** — progress reports and presentations (this document)
- **Research loop skill** — end-to-end experiment cycle automation

### 5.2 Experiment Tracking

- `experiments/log.md` — chronological experiment log with roadmap
- `outputs/phase1_baselines.json` — structured results with full config
- `reports/` — detailed literature survey reports
- Notebooks designed for Colab T4 execution with Drive-based checkpointing

### 5.3 Deployment Pipeline

ONNX export verified for Demo 2 real-time inference:
- YOLOX Tiny (19 MB) → RTMPose-M (52 MB) → STGCN++ FL (2 MB)
- 15–20 FPS on CPU, no PyTorch needed at inference time
- Input: `(1, 2, 100, 17, 3)` → Output: `(1, 10)` logits

---

## 6. Notebooks Inventory

| Notebook | Purpose | Status |
|----------|---------|--------|
| `train_colab_baselines.ipynb` | Phase 1: 4 baseline methods (50 rounds) | **Executed** — results in JSON |
| `train_colab_methods.ipynb` | Phase 3+4: PFL methods scan + multimodal | **Created** (32 cells) — not yet run |
| `train_colab.ipynb` | Original FL training notebook | Superseded by baselines |
| `train_colab_medical.ipynb` | Medical subset data prep | Utility |
| `train_colab_ntu60.ipynb` | Full NTU-60 training | Utility |

---

## 7. Current Status & Next Steps

### What's Done
- [x] FL infrastructure (Flower simulation, STGCN++, ONNX export)
- [x] Phase 1 baselines — 4 methods compared, 50 rounds each
- [x] Phase 3 literature survey — 7 PFL methods analyzed, 4 implemented
- [x] Phase 4 literature survey — 7 datasets reviewed, synthetic IMU designed, late fusion implemented
- [x] Agent system and experiment tracking infrastructure

### What's Next (Prioritized)

| Priority | Task | Estimated Colab Time | Expected Outcome |
|----------|------|---------------------|------------------|
| **1** | Run `train_colab_methods.ipynb` Part A — PFL quick scan | ~1.5 hours | Identify best PFL method |
| **2** | Run Part B — multimodal skeleton+IMU comparison | ~30 min | Determine if synthetic IMU helps |
| **3** | Deep dive: 50-round run on top PFL method | ~75 min | Full convergence comparison |
| **4** | Re-run baselines with per-class F1 and confusion matrix | ~4 hours | Fill missing per-class medical HAR metrics |
| **5** | Phase 2: 3D skeleton (NTU 25j) transition | ~1 day | Assess 2D vs 3D for medical HAR |

### Open Questions

1. **Why does FedAvg outperform FedBN?** — Is α=0.5 heterogeneity too mild for BN localization to help?
2. **What causes post-R20 degradation?** — Client drift vs. LR schedule vs. overfitting? FedProx should directly test this.
3. **Do personalized models (Ditto, APFL) close the centralized gap?** — Literature suggests +2–5 pp possible.
4. **Does synthetic IMU provide signal above skeleton-only?** — Proof of concept before pursuing real IMU datasets.

---

## Appendix

### A. Output Files

| File | Description |
|------|-------------|
| `outputs/phase1_baselines.json` | Full Phase 1 results with config and sampled histories |
| `outputs/baseline_comparison.png` | 4-method accuracy comparison chart |
| `outputs/medical_data_distribution.png` | Client data distribution heatmap |
| `outputs/fl_vis.png` | FL training visualization (4-panel) |
| `outputs/fl_train.png` | Training curves |
| `outputs/heatmap.png` | Client × class distribution |
| `outputs/confM.png` | Confusion matrix |
| `outputs/global_model.pt` | Best FL model checkpoint |
| `outputs/stgcnpp_federated.onnx` | ONNX-exported model for deployment |

### B. Key References

| Paper | Venue | Year | Relevance |
|-------|-------|------|-----------|
| FedCFE (Liu et al.) | IEEE IoT Journal | 2026 | PFL comparison methods source |
| FedProx (Li et al.) | MLSys | 2020 | Proximal regularization (4000+ cites) |
| Ditto (Li et al.) | ICML | 2021 | Dual model personalization (800+ cites) |
| FedRep (Collins et al.) | ICML | 2021 | Representation learning (700+ cites) |
| APFL (Deng et al.) | arXiv | 2020 | Adaptive mixing |
| IMUTube (Kwon et al.) | IMWUT | 2020 | Virtual IMU validation (216 cites) |
| SmartFallMM (Debnath et al.) | Sensors | 2025 | Elderly activity monitoring, skeleton→IMU |
| Fusion-GCN (Duhme et al.) | GCPR | 2021 | Skeleton+IMU graph fusion (+12.4% F1) |
