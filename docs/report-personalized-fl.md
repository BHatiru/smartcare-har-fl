# Preliminary Research: Personalized Federated Learning Methods for Skeleton-Based Action Recognition

**Research Area:** Federated Learning for Medical HAR  
**Date:** April 2026  
**Context:** Literature survey conducted as part of Phase 3 (Advanced FL Methods) of the federated skeleton-based action recognition project. This report evaluates personalized FL (PFL) methods identified from the FedCFE paper (Liu et al., IEEE IoT Journal 2026) and broader PFL literature, assessing their suitability for our STGCN++ skeleton HAR pipeline.

---

## 1. Motivation

Our Phase 1 baseline experiments revealed a significant accuracy gap between centralized training (89.5%) and federated methods (FedAvg: 77.2%, FedBN: 76.4%, Clustered FedBN: 76.8%) on a 10-class medical action recognition task. All FL methods showed accuracy degradation after round 20 and a ~12–13 percentage point ceiling gap to centralized performance.

This gap motivates exploring **personalized federated learning** (PFL) methods that allow each client (hospital) to maintain a partially customized model while still benefiting from collaborative training. The FedCFE paper (Liu et al., 2026) benchmarks against several PFL baselines — we survey these methods to identify the most promising candidates for our setup.

### Our FL Setup

| Parameter | Value |
|-----------|-------|
| Model | STGCN++ (~443K params, 6 GCN blocks, base_channels=64) |
| Data | NTU RGB+D 60, 10 medical classes, COCO-17 skeleton |
| Clients | 5 simulated hospitals, Dirichlet α=0.5 (non-IID) |
| Training | SGD lr=0.01, warm-start 2 epochs, 1 epoch/round |
| Hardware | Google Colab T4 GPU (~15 min per 10-round experiment) |

---

## 2. Methods Surveyed

We evaluated 7 PFL methods spanning four categories of personalization strategy:

| Category | Methods | Core Idea |
|----------|---------|-----------|
| **Regularization** | FedProx, Ditto | Constrain local updates toward global model |
| **Model Splitting** | FedRep | Shared representation body + local classifier head |
| **Model Mixing** | APFL | Adaptive interpolation between global and local models |
| **Meta-Learning** | Per-FedAvg | Learn a good initialization via MAML-style training |
| **Aggregation** | FedALA | Element-wise weighted aggregation from global to local |
| **Architecture** | FedCFE | Dual feature extractors + knowledge distillation |

---

## 3. Method Analysis

### 3.1 FedProx (Li et al., MLSys 2020) — Regularization

**Mechanism:** Adds a proximal regularization term to the local training objective:

$$\min_w F_k(w) + \frac{\mu}{2}\|w - w^t\|^2$$

where $w^t$ is the current global model. This penalizes client drift — local updates that diverge too far from the global consensus.

**Relevance to our problem:** Our Phase 1 results showed accuracy degradation after round 20, which is consistent with client drift in non-IID settings. FedProx directly addresses this by constraining local updates.

**Implementation:** Minimal — requires only modifying the local training loop to add the proximal penalty term. Approximately 20 lines of code change to the existing training function. The proximal term can be computed as a sum of squared differences between local and global parameters.

**Key hyperparameter:** $\mu \in \{0.001, 0.01, 0.1, 1.0\}$. Starting value: $\mu = 0.01$.

**Expected impact:** +0.5–2% accuracy over FedAvg. Primary benefit is training stability rather than peak accuracy.

**Citations:** 4000+ citations. Reference implementation available in PFLlib.

---

### 3.2 Ditto (Li et al., ICML 2021) — Dual Model Personalization

**Mechanism:** Each client maintains two separate models:
1. A **global model** $w$ — trained and aggregated via standard FedAvg
2. A **personalized model** $v$ — trained locally with L2 regularization toward the global model:

$$\min_v F_k(v) + \frac{\lambda}{2}\|v - w\|^2$$

Only the global model participates in aggregation. The personalized model is used for inference at each client.

**Relevance to our problem:** In a medical deployment scenario, each hospital would use its personalized model for local inference while contributing to collective learning through the global model. This cleanly separates the collaboration mechanism from the personalization objective.

**Implementation:** Moderate — requires maintaining two model instances per client and running two training phases per round. Approximately 40 additional lines. Memory cost is 2× per client (~900K params total), well within T4 capacity.

**Key hyperparameter:** $\lambda \in \{0.01, 0.1, 1.0\}$. Starting value: $\lambda = 0.1$.

**Expected impact:** +2–5% over FedAvg on personalized evaluation. Particularly strong on non-IID data.

**Citations:** 800+ citations. ICML 2021.

---

### 3.3 FedRep (Collins et al., ICML 2021) — Representation Learning

**Mechanism:** Splits the model into a shared **body** (feature extractor) and a local **head** (classifier). Only the body is aggregated across clients. Training alternates between two phases:
1. **Head phase:** Freeze body, train head locally (adapts to local class distribution)
2. **Body phase:** Freeze head, train body (learns shared representations, aggregated)

**Relevance to our problem:** STGCN++ has a natural architectural split — the 6 GCN blocks + temporal convolutions form the body (~440K params), and the final FC layer (256 → 10) forms the head (~2.6K params). Under Dirichlet α=0.5, clients have heterogeneous class distributions, so local heads can specialize to each client's label mix while sharing motion representation learning.

**Implementation:** Medium complexity — requires separating optimizer groups, alternating freeze/unfreeze phases, and modifying aggregation to exclude head parameters. Approximately 50 lines of additional code.

**Key hyperparameters:** Head training epochs = 1, body training epochs = 1 per round.

**Expected impact:** +1–4% over FedAvg. Strongest when class distributions differ across clients.

**Citations:** 700+ citations. ICML 2021.

---

### 3.4 APFL (Deng et al., arXiv 2020) — Adaptive Mixing

**Mechanism:** Each client maintains a global model copy and a local model. A per-client mixing coefficient $\alpha_k$ adaptively interpolates between them:

$$w_k^{\text{personal}} = \alpha_k \cdot w_k^{\text{local}} + (1 - \alpha_k) \cdot w^{\text{global}}$$

The mixing coefficient $\alpha_k$ is updated based on which model (global vs. local) yields lower loss on local data. Clients that benefit more from global knowledge naturally decrease $\alpha$, while clients with distinctive distributions increase $\alpha$.

**Relevance to our problem:** With 5 clients of varying sizes (918–1798 samples) and different class mixes, adaptive mixing could automatically determine how much each hospital should rely on collaborative vs. local knowledge.

**Implementation:** Low-medium — dual model copy + alpha update rule. Approximately 40 lines. No architecture changes needed.

**Expected impact:** Competitive with Ditto. Auto-adaptive nature reduces hyperparameter sensitivity.

---

### 3.5 Per-FedAvg (Fallah et al., NeurIPS 2020) — Meta-Learning

**Mechanism:** Uses MAML-style meta-learning to find a global initialization that adapts quickly to each client's distribution. Each round performs a two-step inner/outer optimization: inner update on one data batch, meta-gradient computed on a different batch.

**Assessment:** Theoretically elegant but practically challenging. Doubles training cost, requires careful meta-learning rate tuning, and MAML-style second-order gradients interact poorly with batch normalization layers common in GCN architectures. **Lower priority** for implementation.

---

### 3.6 FedALA (Zhang et al., AAAI 2023) — Adaptive Local Aggregation

**Mechanism:** Element-wise weighted aggregation from global to local model. An Adaptive Local Aggregation (ALA) module learns per-parameter weights that control how much of the global update each client absorbs.

**Assessment:** Sophisticated but complex (~100 lines for the ALA module alone) with many hyperparameters ($\eta$, `layer_idx`, `rand_percent`, convergence threshold). The complexity doesn't justify the marginal gain for our 5-client setup. **Low priority.**

---

### 3.7 FedCFE (Liu et al., IEEE IoT Journal 2026) — Composite Feature Extraction

**Mechanism:** Parallel global and local feature extractors, a conditional generator for pseudo-feature synthesis, and knowledge distillation between extractors. Represents the state-of-the-art the paper proposes.

**Assessment:** This is the paper's own contribution — the most complex method surveyed. Requires duplicating the STGCN++ backbone (2× encoder params), implementing a conditional generator for skeleton features, and designing a custom KD loss. Estimated 300+ lines of new code with significant architecture restructuring. **Not planned for reimplementation** — we will cite their reported numbers and focus on implementing the baselines they compared against.

---

## 4. Implementation Plan

Based on the analysis above, we prioritize methods by implementation effort vs. expected research value:

### Phase 1 — Quick Wins (implemented in notebook, ~1 hour each)

| Priority | Method | Rationale |
|----------|--------|-----------|
| **1** | FedProx | Minimal code change, directly addresses client drift, expected baseline in any PFL comparison |
| **2** | Ditto | Clean personalization framework, strong expected gains, easy to explain |

### Phase 2 — Medium Effort (implemented in notebook, ~2 hours each)

| Priority | Method | Rationale |
|----------|--------|-----------|
| **3** | FedRep | Exploits natural STGCN++ body/head split, tests representation sharing hypothesis |
| **4** | APFL | Auto-adaptive, good for heterogeneity analysis across clients |

### Phase 3 — Only If Time Permits

| Priority | Method | Rationale |
|----------|--------|-----------|
| **5** | Per-FedAvg | Theoretically interesting, but 2× cost and MAML complexity |
| **6** | FedALA | Complex, many hyperparameters, better suited for large-scale FL |

### Not Implementing

| Method | Reason |
|--------|--------|
| FedCFE | The paper's own method — cite their numbers, don't reimplement |
| SCAFFOLD | 2× communication cost, overkill for 5 clients |
| FedNova | Addresses heterogeneous local epochs — not our scenario |
| MOON | Contrastive learning — high complexity, marginal benefit |

---

## 5. Experimental Design

All methods are implemented in a single Colab notebook (`train_colab_methods.ipynb`) for quick-scan comparison:

- **10 FL rounds** per method (reduced from 50 to enable ~15 min training per method on T4)
- **Same data partition**: 5 clients, Dirichlet α=0.5, 10 medical classes
- **Warm-start**: 2 epochs on combined data (shared initialization)
- **Evaluation**: Global test accuracy + per-client personalized accuracy where applicable
- **Outputs**: Comparison bar chart, learning curves, summary table with Δ vs. FedAvg

Methods producing promising results in the quick scan will be run at full scale (50 rounds) in subsequent experiments.

---

## 6. Reference Implementation

All surveyed methods (except FedCFE) have reference implementations in **PFLlib** (TsingZ0/PFLlib), a JMLR-published library with Apache 2.0 license. Our implementations are adapted to work with the Flower FL framework and STGCN++ architecture used in this project.

### Key References

| Paper | Venue | Citations |
|-------|-------|-----------|
| FedProx (Li et al., 2020) | MLSys 2020 | 4000+ |
| Ditto (Li et al., 2021) | ICML 2021 | 800+ |
| FedRep (Collins et al., 2021) | ICML 2021 | 700+ |
| APFL (Deng et al., 2020) | arXiv 2020 | — |
| Per-FedAvg (Fallah et al., 2020) | NeurIPS 2020 | — |
| FedALA (Zhang et al., 2023) | AAAI 2023 | — |
| FedCFE (Liu et al., 2026) | IEEE IoT Journal 2026 | — |
| PFLlib (Zhang et al.) | JMLR | — |
